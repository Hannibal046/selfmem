def get_rouge_score(hyps,refs):
    from compare_mt.rouge.rouge_scorer import RougeScorer
    assert len(hyps)==len(refs)
    lens = len(hyps)    
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)    
    rouge1 = rouge2 = rougel = 0.0
    for hyp,ref in zip(hyps,refs):
        score = rouge_scorer.score(ref,hyp)
        rouge1 += score['rouge1'].fmeasure
        rouge2 += score['rouge2'].fmeasure
        rougel += score['rougeLsum'].fmeasure
    rouge1 = rouge1 / lens
    rouge2 = rouge2 / lens
    rougel = rougel / lens
    return rouge1,rouge2,rougel

def get_acc(preds,labels):
    assert len(preds) == len(labels)
    acc = 0
    for p,l in zip(preds,labels):
        if p==l:acc += 1
    return acc/len(preds)

def get_sentence_bleu(hyp,ref):
    import sacrebleu
    return sacrebleu.sentence_bleu(hyp, [ref],  smooth_method='exp').score

# def get_rouge_score(hyps,refs):
#     ## pip install py-rouge
#     import rouge

#     def postprocess_text(preds, labels):
#         import nltk
#         preds = [pred.strip() for pred in preds]
#         labels = [label.strip() for label in labels]

#         # rougeLSum expects newline after each sentence
#         preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#         labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

#         return preds, labels
    
#     hyps,refs = postprocess_text(hyps,refs) # add \n

#     evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
#                             max_n=2,
#                             limit_length=True,
#                             length_limit=100,
#                             length_limit_type='words',
#                             apply_avg=True,
#                             apply_best=False,
#                             alpha=0.5,  # Default F1_score
#                             stemming=True)
#     py_rouge_scores = evaluator.get_scores(hyps, refs)

#     ## *100
#     for k,v in py_rouge_scores.items():
#         for _k,_v in v.items():
#             py_rouge_scores[k][_k] = round(_v*100,4)

#     return py_rouge_scores

def get_bleu_score(hyps,refs,return_signature=False):
    # pip install sacrebleu
    """
    hyps:list of string
    refs:list of string
    """
    assert len(hyps) == len(refs)
    
    import sacrebleu
    scorer = sacrebleu.metrics.BLEU(force=True)
    score = scorer.corpus_score(hyps,[refs]).score
    signature = scorer.get_signature()
    if return_signature:
        return score,str(signature)
    else:
        return score
    # return sacrebleu.corpus_bleu(hyps, [refs],force=True).score

def get_chrf_score(hyps,refs,return_signature=False):
    
    assert len(hyps) == len(refs)
    import sacrebleu
    scorer = sacrebleu.CHRF()
    score = scorer.corpus_score(hyps,[refs]).score
    signature = scorer.get_signature()
    if return_signature:
        return score,str(signature)
    else:
        return score

def get_ter_score(hyps,refs,return_signature=False):
    
    assert len(hyps) == len(refs)
    import sacrebleu
    scorer = sacrebleu.TER()
    score = scorer.corpus_score(hyps,[refs]).score
    signature = scorer.get_signature()
    if return_signature:
        return score,str(signature)
    else:
        return score

def get_perplexity(loss, _round=2, base='e'):
    # from fairseq.logging.meters import safe_round
    import math
    
    def safe_round(number, ndigits):
        import numpy as np
        import torch
        if hasattr(number, "__round__"):
            return round(number, ndigits)
        elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
            return safe_round(number.item(), ndigits)
        elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
            return safe_round(number.item(), ndigits)
        else:
            return number

    if loss is None:
        return 0.0
    try:
        if base=='e':
            return safe_round(math.exp(loss), _round)
        else:
            return safe_round(base**loss,_round)
    except OverflowError:
        return float("inf")

def get_edit_distance(x,y,len_split=False):
    # Here first sentence is the edited sentence and the second sentence is the target sentence
    # If src_sent is the first sentence, then insertion and deletion should be reversed.
    """ Dynamic Programming Version of edit distance and finding unedited words."""
    import editdistance
    edit_distance = editdistance.eval(x.split(), y.split())
    if len_split:
        edit_distance = 1 - edit_distance / max(len(y.split()), len(x.split()))
    else:
        edit_distance = 1 - edit_distance / max(len(y), len(x))
    return edit_distance

def get_ndcg_score(relevance_score,true_relevance):
    from sklearn.metrics import ndcg_score
    import numpy as np
    """
    # Relevance scores in Ideal order
    true_relevance = np.asarray([[3, 2, 1, 0, 0]])

    # Relevance scores in output order
    relevance_score = np.asarray([[3, 2, 0, 0, 1]])
    """
    return ndcg_score(np.asarray([true_relevance]),np.asarray([relevance_score]))
