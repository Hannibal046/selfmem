import json,os,time,argparse,warnings,time,yaml
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
from utils.ddp_utils import (
    UnevenSequentialDistributedSampler,
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

def collate_fct(samples,toker,max_src_len,max_trg_len,src='document',trg='summary',num_candidates=None,is_training=False):
    
    src = [d[src] for d in samples]
    trg = [d[trg] for d in samples]
    candidates = [d['candidates'] for d in samples]
    for idx in range(len(candidates)):
        if is_training:
            candidates[idx].sort(key=lambda x:x[1],reverse=True)
            if num_candidates is not None:
                candidates[idx] = candidates[idx][:1] + candidates[idx][-(num_candidates-1):]
        candidates[idx] = [x[0] for x in candidates[idx]]
        if is_training:candidates[idx].insert(0,trg[idx])
    flattened_candidates = [x for y in candidates for x in y]

    tokenized_src = toker(src,return_tensors='pt',padding=True,truncation=True,max_length=max_src_len)
    tokenized_candidates = toker(flattened_candidates,return_tensors='pt',padding=True,truncation=True,max_length=max_trg_len)

    return {
        "src_input_ids":tokenized_src['input_ids'],
        "src_attention_mask":tokenized_src['attention_mask'],
        "candidate_input_ids":tokenized_candidates['input_ids'],
        "candidate_attention_mask":tokenized_candidates['attention_mask'],
        "candidates":candidates,
        "refs":trg,
    }

class RankingModel(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        
        parser = parent_parser.add_argument_group("model_args")
        ## data
        parser.add_argument('--data_path',)
        parser.add_argument('--config_path',)
        parser.add_argument('--candidate_path',)
        parser.add_argument('--output_path',)
        parser.add_argument('--src')
        parser.add_argument('--trg')
        parser.add_argument('--max_trg_len', type=int)
        parser.add_argument('--max_src_len', type=int)
        parser.add_argument('--pretrained_model_path')
        parser.add_argument('--per_device_eval_batch_size',type=int)
        parser.add_argument('--eval_metrics')
        parser.add_argument('--seed',type=int)
        parser.add_argument('--architecture')
        

        
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
                                  is_training=True)
        self.test_collate_fct = partial(self.train_collate_fct,is_training=False)
        
        self.contrastive_loss_list = []
        self.simcls_loss_list = []
        self.total_loss_list = []

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
        
        self.total_loss_list.append(total_loss.item())
        
        return total_loss

    def test_step(self, batch, batch_idx):
        hyps = self.rank(batch)
        return hyps,batch['refs']

    def rank(self,batch):
        logits = self.get_logits(batch)
        index = torch.argmax(logits,dim=1).tolist()
        hyps = [candidate[i] for candidate,i in zip(batch['candidates'],index)]
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
        if self.logger:self.log("v_num",self.logger.version)
        hyps,refs = self.merge(outputs)
        hyps = [x for y in hyps for x in y]
        refs = [x for y in refs for x in y]
        self.eval_generation(hyps,refs,'test')

        if self.trainer.is_global_zero:
            with open(self.hparams.output_path,'w') as f:
                for h in hyps[:self.test_data_cnt]:f.write(h.replace("\n"," ")+"\n")
    
    def load_data(self,_split):

        data_path = self.hparams.data_path
        data = [json.loads(x) for x in open(data_path).readlines()]
        data_cnt = len(data)
        
        candidate_path = data_path = self.hparams.candidate_path
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
        assert stage == 'test'
        self.test_data_cnt,self.test_dataset=self.load_data('test')
    
    def test_dataloader(self):
        if self.trainer.num_devices > 1:
            sampler = UnevenSequentialDistributedSampler(self.test_dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hparams.per_device_eval_batch_size,
                                           shuffle=False,collate_fn=self.test_collate_fct,
                                           num_workers=8, pin_memory=True,sampler=sampler)

if __name__ == "__main__":
    
    ## args
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = RankingModel.add_model_specific_args(parser)
    args = parser.parse_args()
    
    config = yaml.full_load(open(args.config_path))
    for k,v in config.items():
        if getattr(args,k) is None:
            setattr(args,k,v)
    ## model
    model = RankingModel(**vars(args))
    
    ## strategy
    strategy = None
    if args.accelerator == 'gpu' and torch.cuda.device_count()>1:strategy = DDPStrategy(find_unused_parameters=False)

    ## trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy = strategy,
    )
    trainer.test(model)