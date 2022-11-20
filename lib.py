## Built-in Module
import pickle
import os
from os import system as shell
import json
import warnings
warnings.filterwarnings("ignore")
import time
from contextlib import nullcontext
from tqdm import tqdm
# import wandb

## torch
import torch
from torch.utils.data import DataLoader

## torch DDP
import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

## huggingface/transformers
from transformers import set_seed

## own
# from utils.ddp_utils import mprint as print
from utils.ddp_utils import (
    is_main_process,
    UnevenSequentialDistributedSampler,
    wait_for_everyone,
    set_available_port,
)
from utils.metrics_utils import (
    get_rouge_score,
    get_bleu_score,
    get_ter_score,
    get_chrf_score,
    get_sentence_bleu,
)
from utils.utils import (
    get_txt,
    MetricsTracer,
    get_model_parameters,
    move_to_device,
    get_remain_time,
    s2hm,
    s2ms,
    dump_vocab,
    get_current_gpu_usage,
    debpe,
    get_files,
    split_list,
    get_jsonl,
)