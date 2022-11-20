import json,os,time,argparse,warnings,time,yaml
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
)
## own
from utils.utils import (
    LabelSmoother,
    get_remain_time,
    split_list,
)
from utils.metrics_utils import (
    get_rouge_score,
    get_bleu_score,
)
from utils.optim_utils import (
    get_inverse_sqrt_schedule_with_warmup
)

class MemoryDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        memory=None,
        ):
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

def collate_fct(samples,toker,max_length,src='document',trg='summary',num_candidates=None,is_training=False):
    
    batch_size = len(samples)
    src = [d[src] for d in samples]
    trg = [d[trg] for d in samples]
    candidates = [d['candidates'] for d in samples]
    for idx in range(len(candidates)):
        candidates[idx].sort(key=lambda x:x[1],reverse=True)
        if num_candidates is not None and is_training:
            candidates[idx] = candidates[idx][:1] + candidates[idx][-(num_candidates-1):]
        candidates[idx] = [x[0] for x in candidates[idx]]
        num_candidates = len(candidates[idx])
    flattened_candidates = [x for y in candidates for x in y]

    expanded_src = [[x] * num_candidates for x in src]
    expanded_src = [x for y in expanded_src for x in y]
    
    input = toker(expanded_src,flattened_candidates,return_tensors='pt',padding=True,truncation='only_first',max_length=max_length)
    input['input_ids'] = input['input_ids'].view(batch_size,num_candidates,-1)
    
    ret = {**input}
    ret['candidates'] = candidates
    ret['refs'] = trg
    
    return ret

class SingleTowerRankingModel(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        
        parser = parent_parser.add_argument_group("model_args")
        ## data
        parser.add_argument('--data_dir',)
        parser.add_argument('--config_path',)
        parser.add_argument('--candidate_dir',)
        parser.add_argument('--src')
        parser.add_argument('--trg')
        parser.add_argument('--max_length', )
        parser.add_argument('--pretrained_model_path')
        parser.add_argument("--temperature")
        parser.add_argument('--lr')
        parser.add_argument('--warmup_steps',)
        parser.add_argument('--weight_decay',)
        parser.add_argument('--per_device_train_batch_size')
        parser.add_argument("--num_candidates")
        parser.add_argument('--per_device_eval_batch_size')
        parser.add_argument('--logging_steps')
        parser.add_argument('--eval_metrics')
        parser.add_argument('--seed')
        
        return parent_parser
    
    def __init__(self,*args,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.configure_model()
        self.train_collate_fct = partial(collate_fct,
                                  toker = self.toker,
                                  max_length = self.hparams.max_length,
                                  src = self.hparams.src,trg = self.hparams.trg,
                                  num_candidates=self.hparams.num_candidates,
                                  is_training=True)
        self.test_collate_fct = partial(self.train_collate_fct,is_training=True)
        
        self.contrastive_losses = []
        self.total_losses = []

    def configure_model(self):
        self.toker = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_path,use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained_model_path,num_labels=1)

    def eval_generation(self,hyps,refs,stage='valid'):
        if stage == 'valid':
            cnt = self.valid_data_cnt
        elif stage == 'test':
            cnt = self.test_data_cnt
        hyps = hyps[:cnt]
        refs = refs[:cnt]
        r1,r2,rl = get_rouge_score(hyps,refs)
        bleu = get_bleu_score(hyps,refs)
        self.log(stage+"_rouge1",r1)
        self.log(stage+"_rouge2",r2)
        self.log(stage+"_rougeL",rl)
        self.log(stage+"_bleu",bleu)

    def contrastive_loss_fct(self,scores):
        ## score: [bs,num_candidates]
        loss = -nn.LogSoftmax(1)(scores/self.hparams.temperature)
        loss = loss[:,0].mean()
        return loss


    def get_ranking_loss(self,batch):
        
        batch_size,num_candidates,_ = batch['input_ids'].shape
        batch['input_ids'] = batch['input_ids'].view(batch_size * num_candidates,-1)
        refs = batch.pop('refs')
        candidates = batch.pop('candidates')

        model_output = self.model(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch else None,
            )
        logits = model_output.logits.view(batch_size,num_candidates)

        
        contrastive_loss = self.contrastive_loss_fct(logits)
        self.contrastive_losses.append(contrastive_loss)

        total_loss = contrastive_loss
        self.total_losses.append(total_loss)
        
        batch['refs']=refs
        batch['candidates']=candidates
        
        return total_loss

    def training_step(self,batch,batch_idx):
        loss = self.get_ranking_loss(batch)
        self.log("train_loss",loss,on_step=True,on_epoch=True,batch_size=batch['input_ids'].shape[0])
        return loss
    
    def test_step(self, batch, batch_idx):
        hyps = self.rank(batch)
        return hyps,batch['refs']

    def validation_step(self,batch,batch_idx):
        hyps = self.rank(batch)
        return hyps,batch['refs']
    
    def rank(self,batch):
        candidates = batch.pop("candidates")
        refs = batch.pop("refs")
        batch_size,num_candidates,_ = batch['input_ids'].shape
        batch['input_ids'] = batch['input_ids'].view(batch_size * num_candidates,-1)
        model_output = self.model(**batch)
        batch['refs']=refs
        batch['candidates']=candidates
        logits = model_output.logits.view(batch_size,num_candidates)
        index = torch.argmax(logits,dim=1).tolist()
        
        hyps = [candidate[i] for candidate,i in zip(candidates,index)]
        return hyps

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
        hyps,refs = self.merge(outputs)
        hyps = [x for y in hyps for x in y]
        refs = [x for y in refs for x in y]
        self.eval_generation(hyps,refs,'test')

        if self.trainer.is_global_zero:
            if self.hparams.do_generation:
                with open(os.path.join(self.trainer.log_dir,'test_hyps.txt'),'w') as f:
                    for h in hyps[:self.test_data_cnt]:f.write(h.replace("\n"," ")+"\n")
                with open(os.path.join(self.trainer.log_dir,'test_refs.txt'),'w') as f:
                    for r in refs[:self.test_data_cnt]:f.write(r.replace("\n"," ")+"\n")
            model_type = os.path.basename(self.hparams.pretrained_model_path)
            self.model.save_pretrained(os.path.join(self.trainer.log_dir,model_type+'_best_ckpt'))
            self.src_toker.save_pretrained(os.path.join(self.trainer.log_dir,model_type+'_best_ckpt'))
            # self.trg_toker.save_pretrained(os.path.join(self.trainer.log_dir,'best_ckpt_huggingface/trg_toker'))
    
    def validation_epoch_end(self,outputs):
        hyps,refs = self.merge(outputs)
        hyps = [x for y in hyps for x in y]
        refs = [x for y in refs for x in y]
        self.eval_generation(hyps,refs,'valid')
        

    def on_train_start(self) -> None:
        self.train_start_time = time.time()
        self.print(self.hparams)

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:
        if self.global_step % self.hparams.logging_steps == 0 and self.global_step !=0:
            msg  = f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))} "
            msg += f"[{self.trainer.current_epoch}|{self.trainer.max_epochs}] "
            msg += f"[{self.global_step:6}|{self.trainer.estimated_stepping_batches}] "
            msg += f"Total Loss:{sum(self.total_losses)/len(self.total_losses):.4f} "
            self.total_losses = []
            # msg += f"Mle Loss:{sum(self.mle_losses)/len(self.mle_losses):.4f} "
            # self.mle_losses = []
            # msg += f"Ranking Loss:{sum(self.ranking_losses)/len(self.ranking_losses):.4f} "
            # self.ranking_losses = []
            msg += f"lr:{optimizer.param_groups[0]['lr']:e} "
            msg += f"remaining:{get_remain_time(self.train_start_time,self.trainer.estimated_stepping_batches,self.global_step)} "
            if 'valid_rouge1' in self.trainer.callback_metrics.keys():
                msg += f"valid_rouge1:{self.trainer.callback_metrics['valid_rouge1']:.4f} "
            if 'valid_ppl' in self.trainer.callback_metrics.keys():
                msg += f"valid_ppl:{self.trainer.callback_metrics['valid_ppl']:.4f} "
            if 'valid_bleu' in self.trainer.callback_metrics.keys():
                msg += f"valid_bleu:{self.trainer.callback_metrics['valid_bleu']:.4f} "
            self.print(msg)

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

    @staticmethod
    def reorder_ddp(all_rank_outputs):
        ## this function can only do with only 1 hyp
        rank_cnt = dist.get_world_size()
        num_data_per_rank = int(len(all_rank_outputs)/rank_cnt)
        output = []
        for idx in range(num_data_per_rank):
            output.extend([all_rank_outputs[i] for i in range(idx,len(all_rank_outputs),num_data_per_rank)])
        return output
    
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
            self.valid_data_cnt,self.valid_dataset=self.load_data('dev')
        # elif stage == 'valid':
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
    parser = SingleTowerRankingModel.add_model_specific_args(parser)
    args = parser.parse_args()
    
    config = yaml.full_load(open(args.config_path))
    for k,v in config.items():
        if getattr(args,k) is None:
            setattr(args,k,v)
    ## seed
    pl.seed_everything(args.seed,workers=True)
    
    ## model
    model = SingleTowerRankingModel(**vars(args))
    
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