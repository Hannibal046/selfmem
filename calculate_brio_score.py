from lib import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--refs_path")
parser.add_argument("--candidates_path")
parser.add_argument("--output_path",default=None)

args = parser.parse_args()

candidates = get_txt(args.candidates_path)
refs = [x['summary'] for x in get_jsonl(args.refs_path)]

assert len(candidates)%len(refs)==0,(len(candidates),len(refs))

multiple = int(len(candidates)/len(refs))

refs = [[x]*multiple for x in refs]
refs = [x for y in refs for x in y]

def cal_score(hyp,ref):
    r1,r2,rl = get_rouge_score([hyp],[ref])
    if r1+r2 == 0:
        return 0
    score = 2*r1*r2/(r1+r2)
    # score = (r1+r2+rl)/3
    return score

scores = [cal_score(hyp,ref) for hyp,ref in zip(candidates,refs)]

if args.output_path is None:
    _split = os.path.basename(args.candidates_path).split(".")[0]
    args.output_path = os.path.join(os.path.dirname(args.candidates_path),_split+".scores")

with open(args.output_path,"w") as f:
    for s in scores:
        f.write(str(s)+'\n')
print(f"writing to {args.output_path} done")