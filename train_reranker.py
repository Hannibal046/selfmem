import json,os,time,argparse,warnings,time,yaml,random
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
## torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
## lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    ModelSummary,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)
## transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
)
## own
from utils.utils import (
    LabelSmoother,
    get_remain_time,
    split_list,
    get_gpu_usage,
)
from utils.metrics_utils import (
    get_rouge_score,
    get_bleu_score,
    get_nltk_bleu_score,
    get_distinct_score,
)
from utils.optim_utils import (
    get_inverse_sqrt_schedule_with_warmup
)

class MemoryDataset(torch.utils.data.Dataset):

    def __init__(self,data,memory=None,):
        super().__init__()
        self.data = data
        if memory is not None:
            assert len(data)==len(memory),(len(data),len(memory))
            for idx in range(len(data)):
                self.data[idx]['memory']=memory[idx]
    
    def __getitem__(self,index):
        return self.data[index]

    def __len__(self,):
        return len(self.data)

def collate_fct(samples,toker,max_src_len,max_trg_len,src='document',trg='summary',num_candidates=None,is_training=False,candidates_sampling=None):
    
    if not is_training:max_trg_len = max_src_len
    batch_size = len(samples)
    src = [d[src] for d in samples]
    trg = [d[trg] for d in samples]
    candidates = [d['candidates'] for d in samples]
    labels = []
    for idx in range(len(candidates)):
        if num_candidates is not None:
            if candidates_sampling == 'sequential':
                candidates[idx]=candidates[idx][:num_candidates]
            elif candidates_sampling == 'random':
                candidates[idx]=random.sample(candidates[idx],num_candidates)
            elif candidates_sampling == 'top_1_plus_bottom':
                candidates[idx].sort(key=lambda x:x[1],reverse=True)
                candidates[idx] = candidates[idx][:1] + candidates[idx][-(num_candidates-1):]
        candidates[idx].sort(key=lambda x:x[1],reverse=True)    
        candidates[idx] = [x[0] for x in candidates[idx]]
        labels.append([x[1] for x in candidates[idx]])
        if is_training:
            candidates[idx].insert(0,trg[idx])
            labels[idx].insert(0,1)
    
    flattened_candidates = [x for y in candidates for x in y]
    flattened_labels = [x for y in labels for x in y]

    tokenized_src = toker(src,return_tensors='pt',padding=True,truncation=True,max_length=max_src_len)
    tokenized_candidates = toker(flattened_candidates,return_tensors='pt',padding=True,truncation=True,max_length=max_trg_len)

    return {
        "src_input_ids":tokenized_src['input_ids'],
        "src_attention_mask":tokenized_src['attention_mask'],
        "candidate_input_ids":tokenized_candidates['input_ids'],
        "candidate_attention_mask":tokenized_candidates['attention_mask'],
        "candidates":candidates,
        "refs":trg,
        "labels":torch.tensor(flattened_labels).view(batch_size,-1)
    }

class RankingModel(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        
        parser = parent_parser.add_argument_group("model_args")
        ## data
        parser.add_argument('--data_dir',)
        parser.add_argument('--config_path',)
        parser.add_argument('--candidate_dir',)
        parser.add_argument('--src')
        parser.add_argument('--trg')
        parser.add_argument('--max_trg_len', type=int)
        parser.add_argument('--max_src_len', type=int)
        parser.add_argument('--pretrained_model_path')
        parser.add_argument("--temperature",type=float)
        parser.add_argument('--lr',type=float)
        parser.add_argument('--warmup_steps',type=int)
        parser.add_argument('--weight_decay',type=float)
        parser.add_argument('--per_device_train_batch_size',type=int)
        parser.add_argument("--num_candidates",type=int)
        parser.add_argument('--per_device_eval_batch_size',type=int)
        parser.add_argument('--logging_steps',type=int)
        parser.add_argument('--eval_metrics')
        parser.add_argument('--seed',type=int)
        parser.add_argument('--cheat',action='store_true')
        parser.add_argument('--contrastive_loss',type=bool)
        parser.add_argument('--simcls_loss',type=bool)
        parser.add_argument('--margin',type=float)
        parser.add_argument('--no_gold',type=bool)
        parser.add_argument('--gold_weight',type=float)
        parser.add_argument('--gold_margin',type=float)
        parser.add_argument('--architecture')
        parser.add_argument('--requires_gold',type=bool)
        parser.add_argument('--candidates_sampling',type=bool)
        
        return parent_parser
    
    def __init__(self,*args,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.configure_model()
        self.train_collate_fct = partial(collate_fct,
                                  toker = self.toker,
                                  max_src_len = self.hparams.max_src_len,
                                  max_trg_len = self.hparams.max_trg_len,
                                  src = self.hparams.src,trg = self.hparams.trg,
                                  num_candidates=self.hparams.num_candidates,
                                  is_training=True,candidates_sampling=self.hparams.candidates_sampling)
        self.test_collate_fct = partial(self.train_collate_fct,is_training=False)
        
        self.contrastive_loss_list = []
        self.simcls_loss_list = []
        self.total_loss_list = []
        self.rank_list = []

    def configure_model(self):

        self.toker = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_path)
        if self.hparams.architecture == 'single_tower':
            self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained_model_path,num_labels=1)
        elif self.hparams.architecture == 'dual_tower':
            self.model = AutoModel.from_pretrained(self.hparams.pretrained_model_path,num_labels=1)

    def eval_generation(self,hyps,refs,stage='valid'):
        if stage == 'valid':
            cnt = self.valid_data_cnt
        elif stage == 'test':
            cnt = self.test_data_cnt
        hyps = hyps[:cnt]
        refs = refs[:cnt]
        r1,r2,rl = get_rouge_score(hyps,refs)
        bleu = get_bleu_score(hyps,refs)
        bleu_1,bleu_2,bleu_3,bleu_4 = get_nltk_bleu_score(hyps,refs)
        distinct_1,distinct_2 = get_distinct_score(hyps)

        metrics_dict = {
                stage+"_rouge1":r1,
                stage+"_rouge2":r2,
                stage+"_rougeL":rl,
                stage+"_bleu":bleu,
                stage+"_bleu1":bleu_1,
                stage+"_bleu2":bleu_2,
                stage+"_bleu3":bleu_3,
                stage+"_bleu4":bleu_4,
                stage+"_distinct_1":distinct_1,
                stage+"_distinct_2":distinct_2,
            }
        self.log_dict(metrics_dict)
        self.print(json.dumps(metrics_dict,indent=4))

    def listwise_kl_loss_fct(self,logits,labels):

        norm_target = labels[:,1:] # gold

        bs = labels.shape[0]
        min_v = torch.min(labels, 1, keepdim=True).values
        max_v = torch.max(labels, 1, keepdim=True).values
        norm_target = (labels - min_v) / (max_v - min_v + torch.finfo(torch.float32).eps)

        target_dist = F.softmax(norm_target / self.hparams.temperature, dim=-1)
        model_dist = F.log_softmax(logits[:,1:], dim=-1)
        loss = -(target_dist * model_dist - target_dist * target_dist.log()).sum()
        return loss

    def listwise_contrastive_loss_fct(self,scores):
        ## score: [bs,num_candidates]
        if not self.hparams.requires_gold:
            scores = scores[:,1:]
        loss = -nn.LogSoftmax(1)(scores/self.hparams.temperature)
        loss = loss[:,0].mean()
        return loss
    
    def pairwise_ranking_loss_fct(self,logits):
        ## logits: [bs,num_candidates]
        logits = torch.nn.functional.normalize(logits,dim=1)
        refs_scores = logits[:,0]
        candidates_scores = logits[:,1:]
        
        ones = torch.ones_like(candidates_scores)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(candidates_scores, candidates_scores, ones)
        # candidate loss
        n = candidates_scores.size(1)
        for i in range(1, n):
            pos_score = candidates_scores[:, :-i]
            neg_score = candidates_scores[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(self.hparams.margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
        if self.hparams.no_gold:
            return TotalLoss
        # gold summary loss
        pos_score = refs_scores.unsqueeze(-1).expand_as(candidates_scores)
        neg_score = candidates_scores
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(self.hparams.gold_margin)
        TotalLoss += self.hparams.gold_weight * loss_func(pos_score, neg_score, ones)
        return TotalLoss

    def get_logits(self,batch):
        
        batch_size = batch['src_input_ids'].shape[0]
        num_candidates = int(batch['candidate_input_ids'].shape[0]/batch_size)
        
        if self.hparams.architecture == 'dual_tower':

            src_embedding = self.model(
                input_ids = batch['src_input_ids'],
                attention_mask = batch['src_attention_mask'],
            ).pooler_output ## bs,d_model

            candidates_embedding = self.model(
                input_ids = batch['candidate_input_ids'],
                attention_mask = batch['candidate_attention_mask'],
            ).pooler_output.view(batch_size,num_candidates,-1) ## bs,num_candidates,d_model

            logits = torch.cosine_similarity(candidates_embedding,src_embedding.unsqueeze(1).expand_as(candidates_embedding),dim=-1)
            
        elif self.hparams.architecture == 'single_tower':

            candidate_input_ids = batch['candidate_input_ids']
            candidate_attention_mask = batch['candidate_attention_mask']
            src_input_ids = batch['src_input_ids']
            src_attention_mask = batch['src_attention_mask']
            src_input_ids = src_input_ids.repeat_interleave(num_candidates,dim=0)
            src_attention_mask = src_attention_mask.repeat_interleave(num_candidates,dim=0)

            logits = self.model(
                input_ids = torch.cat((src_input_ids,candidate_input_ids),dim=1),
                attention_mask = torch.cat((src_attention_mask,candidate_attention_mask),dim=1),
            ).logits.view(batch_size,num_candidates)
        
        self.cur_logits = logits
        return logits

    def get_ranking(self,logits):
        if self.trainer.state.stage == 'train':
            candidates_logits = logits[:,1:]
        else:
            candidates_logits = logits
        bs,num_candidates = candidates_logits.shape
        self.print(num_candidates)
        rank = (candidates_logits.argmax(dim=1)+1)
        return rank.tolist()
        
    def get_ranking_loss(self,batch):
        
        logits = self.get_logits(batch)
        
        total_loss = 0
        if self.hparams.contrastive_loss:
            contrastive_loss = self.listwise_contrastive_loss_fct(logits)
            total_loss += contrastive_loss
            self.contrastive_loss_list.append(contrastive_loss.item())
        if self.hparams.simcls_loss:
            simcls_loss = self.pairwise_ranking_loss_fct(logits)
            total_loss += simcls_loss
            self.simcls_loss_list.append(simcls_loss.item())
        if self.hparams.kl_loss:
            kl_loss = self.listwise_kl_loss_fct(logits,batch['labels'])
            total_loss += kl_loss
            self.kl_loss_list.append(kl_loss.item())

        self.total_loss_list.append(total_loss.item())
        self.rank_list.extend(self.get_ranking(logits))

        return total_loss

    def training_step(self,batch,batch_idx):
        loss = self.get_ranking_loss(batch)
        self.log("train_loss",loss.item())
        return loss
    
    def test_step(self, batch, batch_idx):
        hyps,ranking = self.rank(batch)
        return hyps,batch['refs'],ranking

    def validation_step(self,batch,batch_idx):
        hyps,ranking = self.rank(batch)
        return hyps,batch['refs'],ranking
    
    def rank(self,batch):
        logits = self.get_logits(batch)
        index = torch.argmax(logits,dim=1).tolist()
        hyps = [candidate[i] for candidate,i in zip(batch['candidates'],index)]
        ranking = self.get_ranking(logits)
        return hyps,ranking

    def merge(self,outputs):

        if dist.is_initialized():
            all_rank_outputs = [None for _ in range(dist.get_world_size())]    
            dist.all_gather_object(all_rank_outputs,outputs)
            outputs = [x for y in all_rank_outputs for x in y] ## all_rank_output[i]: i-th batch output
        single_batch_output_cnt = len(outputs[0])
        ret = [[] for _ in range(single_batch_output_cnt)]
        for idx in range(single_batch_output_cnt):
            for batch in outputs:
                ret[idx].append(batch[idx])
        return ret

    def test_epoch_end(self,outputs):
        if self.logger:self.log("v_num",self.logger.version)
        log_dir = str(self.trainer.log_dir) ## Super Important here to save log_dir 
        hyps,refs,rankings = self.merge(outputs)
        hyps = [x for y in hyps for x in y]
        refs = [x for y in refs for x in y]
        rankings = [x for y in rankings for x in y]
        self.print('avg_ranking:',sum(rankings)/len(rankings))
        self.eval_generation(hyps,refs,'test')

        if self.trainer.is_global_zero:
            with open(os.path.join(log_dir,'test_hyps.txt'),'w') as f:
                for h in hyps[:self.test_data_cnt]:f.write(h.replace("\n"," ")+"\n")
            with open(os.path.join(log_dir,'test_refs.txt'),'w') as f:
                for r in refs[:self.test_data_cnt]:f.write(r.replace("\n"," ")+"\n")
            model_type = os.path.basename(self.hparams.pretrained_model_path)
            self.model.save_pretrained(os.path.join(log_dir,model_type))
            self.toker.save_pretrained(os.path.join(log_dir,model_type))
    
    def validation_epoch_end(self,outputs):
        hyps,refs,rankings = self.merge(outputs)
        hyps = [x for y in hyps for x in y]
        refs = [x for y in refs for x in y]
        rankings = [x for y in rankings for x in y]
        self.print('avg_ranking:',sum(rankings)/len(rankings))
        self.eval_generation(hyps,refs,'valid')

    def on_train_start(self) -> None:
        self.train_start_time = time.time()
        self.print(self.hparams)

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:
            
        if self.global_step % self.hparams.logging_steps == 0 and self.global_step != 0 :
            msg  = f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))} "
            msg += f"[{self.trainer.current_epoch+1}|{self.trainer.max_epochs}] "
            msg += f"[{self.global_step:6}|{self.trainer.estimated_stepping_batches}] "
            
            msg += f"Loss:{sum(self.total_loss_list)/len(self.total_loss_list):.4f} "
            self.total_loss_list = []
            
            if self.contrastive_loss_list:
                msg += f"contrastive_loss:{sum(self.contrastive_loss_list)/len(self.contrastive_loss_list):.4f} "
                self.contrastive_loss_list = []
            
            if self.simcls_loss_list:
                msg += f"simcls_loss:{sum(self.simcls_loss_list)/len(self.simcls_loss_list):.4f} "
                self.simcls_loss_list = []
            
            if self.rank_list:
                msg += f"avg_rank:{sum(self.rank_list)/len(self.rank_list):.4f} "
                self.rank_list = []
            msg += f"Logits:{self.cur_logits[0,:10].tolist()} "
            msg += f"GPU Mem:{get_gpu_usage()} "
            msg += f"lr:{optimizer.param_groups[0]['lr']:e} "
            msg += f"remaining:{get_remain_time(self.train_start_time,self.trainer.estimated_stepping_batches,self.global_step)} "
            if 'valid_'+self.hparams.eval_metrics in self.trainer.callback_metrics.keys():
                msg += f"valid_{self.hparams.eval_metrics}:{self.trainer.callback_metrics['valid_'+self.hparams.eval_metrics]:.4f} "
            self.print(msg)
            self.print(f"avg_rank:{sum(self.rank_list)/len(self.rank_list):.4f} ")
            self.rank_list = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer, self.hparams.warmup_steps)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    },
                }
    
    def load_data(self,_split):

        data_path = os.path.join(self.hparams.data_dir,_split+".jsonl")
        data = [json.loads(x) for x in open(data_path).readlines()]
        data_cnt = len(data)
        
        candidate_path = os.path.join(self.hparams.candidate_dir,_split+".candidates")
        candidates = [x.strip() for x in open(candidate_path).readlines()]
        num_candidates = int(len(candidates)/len(data))
        candidates = split_list(candidates,num_candidates)
        
        score_path = os.path.join(self.hparams.candidate_dir,_split+".scores")
        scores = [float(x.strip()) for x in open(score_path).readlines()]
        scores = split_list(scores,num_candidates)

        assert len(scores) == len(data) == len(candidates)
        for idx in range(len(data)):
            data[idx]['candidates'] = [
                [candidate,score] for candidate,score in zip(candidates[idx],scores[idx])
            ]

        dataset = MemoryDataset(
            data = data,
        )
        return data_cnt,dataset
    
    def setup(self,stage):
        if stage == 'fit':
            self.train_data_cnt,self.train_dataset=self.load_data('train')
            if self.hparams.cheat is not None and self.hparams.cheat:
                self.valid_data_cnt,self.valid_dataset=self.load_data('test')
            else:
                self.valid_data_cnt,self.valid_dataset=self.load_data('dev')
        elif stage == 'validate':
            self.valid_data_cnt,self.valid_dataset=self.load_data('dev')
        elif stage == 'test':
            self.test_data_cnt,self.test_dataset=self.load_data('test')
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.per_device_train_batch_size,
                                           shuffle=True,collate_fn=self.train_collate_fct,
                                           num_workers=8, pin_memory=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.hparams.per_device_eval_batch_size,
                                           shuffle=False,collate_fn=self.test_collate_fct,
                                           num_workers=8, pin_memory=True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hparams.per_device_eval_batch_size,
                                           shuffle=False,collate_fn=self.test_collate_fct,
                                           num_workers=8, pin_memory=True)




if __name__ == "__main__":
    
    ## args
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_shot",action='store_true')
    parser.add_argument("--do_not_train",action='store_true')
    parser.add_argument("--early_stop_patience",type=int,default=-1)
    

    parser = pl.Trainer.add_argparse_args(parser)
    parser = RankingModel.add_model_specific_args(parser)
    args = parser.parse_args()
    
    config = yaml.full_load(open(args.config_path))
    for k,v in config.items():
        if getattr(args,k) is None:
            setattr(args,k,v)
    ## seed
    pl.seed_everything(args.seed,workers=True)
    
    ## model
    model = RankingModel(**vars(args))
    
    ## strategy
    strategy = None
    if args.accelerator == 'gpu' and torch.cuda.device_count()>1:strategy = DDPStrategy(find_unused_parameters=False)

    ## callbacks
    monitor = "valid_"+args.eval_metrics
    mode = 'max' if args.eval_metrics != 'ppl' else 'min'
    callbacks = []
    callbacks.append(ModelCheckpoint(save_top_k=1, monitor=monitor,mode=mode))
    if args.early_stop_patience > -1:
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode,patience=args.early_stop_patience))

    ## trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks= callbacks,
        strategy = strategy,
        val_check_interval=args.val_check_interval,
    )

    if args.zero_shot:
        trainer.test(model)
    
    if not args.do_not_train:
        trainer.fit(model)
        trainer.test(ckpt_path='best')
    else:
        trainer.test(model)
    # trainer.test(model)