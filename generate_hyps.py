import json,os,time,argparse,warnings,time
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES']='1'
## torch
import torch
import torch.distributed as dist
## lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    ModelSummary,
    ModelCheckpoint,
)
from pytorch_lightning.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)
## transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
## own
from utils.utils import (
    LabelSmoother,
    get_remain_time,
)
from utils.metrics_utils import (
    get_rouge_score,
    get_bleu_score,
)
from utils.optim_utils import (
    get_inverse_sqrt_schedule_with_warmup
)
from utils.ddp_utils import (
    UnevenSequentialDistributedSampler,
)
from summarization import (
    DualEncoderPegasusForConditionalGeneration,
    DualEncoderBartForConditionalGeneration,
)
from train_generator import (
    MemoryDataset,
    collate_fct,
    ConditionalGenerator,
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
            if len(data) != len(memory):
                assert len(memory)%len(data)==0,(len(data),len(memory))
                multiple = int(len(memory)/len(data))
                data = [[x]*multiple for x in data]
                data = [x for y in data for x in y]
            assert len(data)==len(memory),(len(data),len(memory))
            for idx in range(len(data)):
                self.data[idx]['memory']=memory[idx]
    
    def __getitem__(self,index):
        return self.data[index]

    def __len__(self,):
        return len(self.data)

class Generator(ConditionalGenerator):
    @staticmethod
    def add_model_specific_args(parent_parser):
        
        parser = parent_parser.add_argument_group("model_args")
        ## data
        parser.add_argument('--data_path', )
        parser.add_argument('--memory_path')
        parser.add_argument('--output_path')
        parser.add_argument('--memory_encoding')
        parser.add_argument('--src', )
        parser.add_argument('--trg', )
        parser.add_argument('--train_max_src_len',)
        parser.add_argument('--train_max_trg_len',)
        ## model
        parser.add_argument('--pretrained_model_path',)
        ## generation
        parser.add_argument('--num_return_sequences')
        parser.add_argument('--num_beam_groups')
        parser.add_argument('--num_beams')
        parser.add_argument('--length_penalty')
        parser.add_argument('--diversity_penalty')
        parser.add_argument('--gen_max_len')
        parser.add_argument('--gen_min_len')
        parser.add_argument('--no_repeat_ngram_size')
        parser.add_argument('--early_stopping')
        parser.add_argument('--top_p')
        parser.add_argument('--temperature')
        parser.add_argument('--do_sample')
        ## training_parameters
        parser.add_argument('--per_device_eval_batch_size',)
        parser.add_argument('--logging_steps')
        parser.add_argument('--seed')
        
        return parent_parser

    def configure_model(self):
        ## tokenizer
        self.src_toker = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_path)
        self.trg_toker = self.src_toker ## to be compatible with NMT task
        self.vocab_size = self.trg_toker.vocab_size
        ## model
        if self.hparams.memory_path is not None:
            ## retrieval-aug
            if self.hparams.memory_encoding == 'concate':
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.pretrained_model_path)
            elif self.hparams.memory_encoding == 'separate':
                if 'pegasus' in self.hparams.pretrained_model_path:
                    self.model = DualEncoderPegasusForConditionalGeneration.from_pretrained(self.hparams.pretrained_model_path)
                elif 'bart' in self.hparams.pretrained_model_path:
                    self.model = DualEncoderBartForConditionalGeneration.from_pretrained(self.hparams.pretrained_model_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.pretrained_model_path)

    def test_step(self, batch, batch_idx):
        hyps = self.generate(batch)
        return hyps,batch['refs']
    
    def test_epoch_end(self,outputs):
        hyps,refs = self.merge(outputs)
        hyps = [x for y in hyps for x in y]
        refs = [x for y in refs for x in y]
        if len(hyps) == len(refs):
            self.eval_generation(hyps,refs,stage='test')
        if self.trainer.is_global_zero:
            if self.hparams.output_path is not None:
                os.makedirs(os.path.dirname(self.hparams.output_path),exist_ok=True)
                with open(self.hparams.output_path,'w') as f:
                    for h in hyps:
                        f.write(h.replace("\n"," ")+"\n")
    
    def setup(self,stage):
        if stage == 'test':
            data = [json.loads(x) for x in open(self.hparams.data_path).readlines()]
            self.test_data_cnt = len(data)
        memory = None
        if self.hparams.memory_path is not None:
            mem_path = os.path.join(self.hparams.memory_path)
            memory = [x.strip() for x in open(mem_path).readlines()]
        self.test_dataset = MemoryDataset(
            data = data,
            memory=memory,
        )

    def test_dataloader(self):
        if self.trainer.num_devices > 1:
            sampler = UnevenSequentialDistributedSampler(self.test_dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hparams.per_device_eval_batch_size,
                                           shuffle=False,collate_fn=self.collate_fct,
                                           num_workers=8, pin_memory=True,sampler=sampler)
    
    def generate(self,batch):
        hyps = []
        with torch.no_grad():
            batch_size = batch['input_ids'].shape[0]
            additional_kwargs = {}
            if 'memory_input_ids' in batch.keys():
                additional_kwargs['memory_input_ids']=batch['memory_input_ids']
                additional_kwargs['memory_attention_mask']=batch['memory_attention_mask']
            output = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=self.hparams.gen_max_len+2,
                min_length=self.hparams.gen_min_len+1,
                no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                num_beams=self.hparams.num_beams,
                length_penalty=self.hparams.length_penalty,
                early_stopping=self.hparams.early_stopping,
                num_return_sequences=self.hparams.num_return_sequences * int(self.hparams.num_beams/self.hparams.num_beam_groups) if self.hparams.num_beam_groups is not None else self.hparams.num_return_sequences,
                num_beam_groups=self.hparams.num_beam_groups, 
                diversity_penalty=self.hparams.diversity_penalty, 
                **additional_kwargs
            )
            hyps = [self.trg_toker.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output]
            if self.hparams.num_beam_groups > 1:
                hyps = [hyps[i] for i in range(len(hyps)) if i % int(self.hparams.num_beams/self.hparams.num_beam_groups) == 0]
        return hyps


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Generator.add_model_specific_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed,workers=True)
    model = Generator(**vars(args))
    strategy = None
    if args.accelerator == 'gpu' and torch.cuda.device_count()>1:strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy = strategy,
        replace_sampler_ddp=False,
    )
    trainer.test(model)