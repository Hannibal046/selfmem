import os
from utils.utils import get_txt,get_jsonl
from utils.metrics_utils import (
    get_rouge_score,
    get_nltk_bleu_score,
    get_sentence_bleu,
)
from utils.utils import run_pool
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--refs_path")
parser.add_argument("--candidates_path")
parser.add_argument("--output_path",default=None)
parser.add_argument("--metrics",default=None,required=True,choices=['r1r2','r1r2rl','b1b2','bleu'])
parser.add_argument("--num_workers",default=15,type=int)

def r1r2(hyp,ref):
    r1,r2,rl = get_rouge_score([hyp],[ref])
    if r1+r2 == 0:
        return 0
    score = 2*r1*r2/(r1+r2)
    return score

def r1r2rl(hyp,ref):
    r1,r2,rl = get_rouge_score([hyp],[ref])
    return (r1+r2+rl)/3

def b1b2(hyp,ref):
    bleu_1,bleu_2,bleu_3,bleu_4 = get_nltk_bleu_score([hyp],[ref])
    return (bleu_1 + bleu_2)/2

def bleu(hyp,ref):
    return get_sentence_bleu(hyp,ref)


if __name__ == '__main__':

    args = parser.parse_args()
    candidates = get_txt(args.candidates_path)
    refs = [x['summary'] for x in get_jsonl(args.refs_path)]
    assert len(candidates)%len(refs)==0,(len(candidates),len(refs))
    multiple = int(len(candidates)/len(refs))
    refs = [[x]*multiple for x in refs]
    refs = [x for y in refs for x in y]

    def cal_score(hyp_ref):
        hyp = hyp_ref[0]
        ref = hyp_ref[1]
        score = eval(args.metrics)(hyp,ref)
        return score

    scores = run_pool(list(zip(candidates,refs)),cal_score,num_works=args.num_workers,verbose=True)

    if args.output_path is None:
        _split = os.path.basename(args.candidates_path).split(".")[0]
        args.output_path = os.path.join(os.path.dirname(args.candidates_path),_split+".scores")

    with open(args.output_path,"w") as f:
        for s in scores:
            f.write(str(s)+'\n')
    print(f"writing to {args.output_path} done")