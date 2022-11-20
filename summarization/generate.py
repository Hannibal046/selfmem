from dataclasses import dataclass
import time
from tqdm import tqdm
import warnings
from typing import List
warnings.filterwarnings("ignore")
import os 
import json
from torch.utils.data import DataLoader
import numpy as np
import torch
import argparse
import torch.distributed as dist
import sys 
sys.path.append("..") 
from utils.metrics_utils import (
    get_bleu_score,
    get_perplexity,
    get_chrf_score,
    get_rouge_score,
    get_ter_score,
)
from utils.ddp_utils import (
    UnevenSequentialDistributedSampler,
    is_main_process,
    set_available_port,
    wait_for_everyone,
)
from utils.utils import (
    move_to_device,
    s2ms,
    debpe,
    dotdict,
    
)
from transformers import set_seed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
@dataclass
class GenerationOutput:
    hyps:List=None
    scores:List=None
    length:List=None

def generate(dataloader,model,trg_tokenizer,device,gen_args=None,progress_bar=False):
    
    model = model.module if hasattr(model,'module') else model
    if not gen_args:gen_args={}
    # gen_args = dict(num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,max_length=118)
    def _generate(dataloader,model,trg_tokenizer,device,gen_args=None,progress_bar=False):
        model.eval()
        hyps = []
        model.eval()
        scores = []
        lens = []
        specal_token_ids = trg_tokenizer.all_special_ids
        dataloader = tqdm(
            dataloader,
            disable=not progress_bar,
            total=len(dataloader),
            leave=False,
        )
        for batch in dataloader:
            batch = move_to_device(batch,device)
            with torch.no_grad():
                output = model.generate(
                        batch['input_ids'],
                        max_length=gen_args.gen_max_len+2,
                        num_beams=gen_args.num_beams,
                        min_length=gen_args.gen_min_len+1,
                        no_repeat_ngram_size=gen_args.no_repeat_ngram_size,
                        early_stopping=gen_args.early_stopping,
                        attention_mask = batch['attention_mask'],
                        output_scores=True,
                        return_dict_in_generate=True,
                        )

            scores.extend(output['sequences_scores'].cpu().tolist())
            generated_tokens = output['sequences'].cpu().tolist()
            generated_token_lens = [sum([1 for x in y if x not in specal_token_ids]) for y in generated_tokens]
            lens.extend(generated_token_lens)
            decoded_hyps = trg_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            hyps.extend(decoded_hyps)
        hyps = [hyp.strip() for hyp in hyps]

        return hyps,scores,lens
    
    if not dist.is_initialized():
        ## single gpu
        hyps,scores,lens = _generate(dataloader,model,trg_tokenizer,device,gen_args,progress_bar=progress_bar)
        return GenerationOutput(
            hyps=hyps,
            scores=scores,
            length=lens,
        )
    else:
        ## multi gpus for training
        outputs = _generate(dataloader,model,trg_tokenizer,device,gen_args,progress_bar=progress_bar)
        all_ranks_outputs = [None for _ in range(dist.get_world_size())]    
        dist.all_gather_object(
            all_ranks_outputs,outputs
            )
        # outputs = [x for y in all_ranks_hyps for x in y]
        wait_for_everyone()
        hyps = [x for y in all_ranks_outputs for x in y[0]]
        scores = [x for y in all_ranks_outputs for x in y[1]]
        length = [x for y in all_ranks_outputs for x in y[2]]
        return GenerationOutput(
            hyps=hyps,scores=scores,length=length
        )
