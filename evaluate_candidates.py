from random import choice
import argparse
from lib import *
from utils.metrics_utils import (
    get_nltk_bleu_score,
    get_distinct_score,
    get_rouge_score,
    get_bleu_score,
    
)

parser = argparse.ArgumentParser()
parser.add_argument("--refs_path")
parser.add_argument("--candidates_path")
parser.add_argument("--trg",default='summary')
parser.add_argument("--metrics",default='r1')

def eval_generation(hyps,refs):
    r1,r2,rl = get_rouge_score(hyps,refs)
    bleu = get_bleu_score(hyps,refs)
    bleu_1,bleu_2,bleu_3,bleu_4 = get_nltk_bleu_score(hyps,refs)
    distinct_1,distinct_2 = get_distinct_score(hyps)
    metrics_dict = {
        "rouge1":r1,
        "rouge2":r2,
        "rougeL":rl,
        "bleu":bleu,
        "bleu1":bleu_1,
        "bleu2":bleu_2,
        "bleu3":bleu_3,
        "bleu4":bleu_4,
        "distinct_1":distinct_1,
        "distinct_2":distinct_2,
    }
    return metrics_dict

def evaluate_candidates(candidates,refs,metrics):
    num_candidates = int(len(candidates)/len(refs))
    candidates = split_list(candidates,num_candidates)

    ## random
    random_scores = []
    trial_cnt = 5
    for _ in range(trial_cnt):
        random_hyps = [choice(x) for x in candidates]
        random_scores.append(eval_generation(random_hyps,refs))
    random_results = {}
    for metric in random_scores[0].keys():
        random_results[metric]=sum(x[metric] for x in random_scores)/trial_cnt
    print("Random Results:")
    print(json.dumps(random_results,indent=4))
    print("***"*30)

    ## best and worst
    best = []
    worst = []
    for idx in range(len(refs)):
        _candidates = candidates[idx]
        ref = refs[idx]
        scores = []
        for candidate in _candidates:
            if metrics == 'r1':
                score = get_rouge_score([candidate],[ref])[0]
            elif metrics == 'bleu':
                score = get_bleu_score([candidate],[ref])
            elif metrics == 'b1':
                score = get_nltk_bleu_score([candidate],[ref])[0]
            scores.append(score)
        _candidates = list(zip(_candidates,scores))
        _candidates.sort(key=lambda x:x[1])
        best.append(_candidates[-1][0])
        worst.append(_candidates[0][0])
    
    print("Best Results:")
    print(json.dumps(eval_generation(best,refs),indent=4))
    print("***"*30)

    print("Worst Results:")
    print(json.dumps(eval_generation(worst,refs),indent=4))
    print("***"*30)


if __name__ == '__main__':

    args = parser.parse_args()
    candidates = get_txt(args.candidates_path)
    refs = [x[args.trg] for x in get_jsonl(args.refs_path)]
    assert len(candidates)%len(refs)==0,(len(candidates),len(refs))
    evaluate_candidates(candidates,refs,args.metrics)