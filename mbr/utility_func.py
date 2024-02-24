# TODO: Put all the utility functions here.
# import re
import numpy as np
from nltk.tokenize import ToktokTokenizer

from evaluate import load
from transformers import CLIPTextModel, CLIPModel, CLIPTokenizer, CLIPProcessor, AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

def load_similarity(sim):

    if sim == 'bertscore':
        similarity = load(sim)
        def compute_similarity(hyp, ref, src):
            return similarity.compute(predictions=hyp, references=ref, lang='en')['f1']
    elif sim == 'sacrebleu':
        similarity = load(sim)
        def compute_similarity(hyp, ref, src):
            scores = [similarity.compute(predictions=[hyp[i]], references=[ref[i]])['score'] for i in range(len(hyp))]
            return scores
    elif sim == 'unigramf1':
        similarity = ToktokTokenizer()
        def compute_similarity(hyp, ref, src):
            nhyp = len(hyp)
            f1s = []
            for i in range(nhyp):
                h = hyp[i]
                r = ref[i]
                hyp_tok = similarity.tokenize(h)
                ref_tok = similarity.tokenize(r)
                
                if len(hyp_tok) == 0 or len(ref_tok) == 0:
                    f1s.append(0.0)
                else:
                    precision = len([token for token in hyp_tok if token in ref_tok]) / len(hyp_tok)
                    recall = len([token for token in hyp_tok if token in ref_tok]) / len(ref_tok)
                    
                    if precision + recall < 0.0001:
                        # Prevent zero division.
                        f1s.append(0.0)
                    else:
                        f1s.append(2.0 * precision * recall / (precision + recall))
            return f1s
    else:
        assert False

    return compute_similarity, similarity

def load_distance(sim, compute_similarity):
    if sim != 'sacrebleu':
        def compute_distance(hyp, ref, src):
            return [1.0 - sim for sim in compute_similarity(hyp, ref, src)]
    else:
        # sacrebleu ranges (0, 100), so need to normalize it.
        def compute_distance(hyp, ref, src):
            return [1.0 - sim / 100.0 for sim in compute_similarity(hyp, ref, src)]

    return compute_distance


def load_evaluate(eval_func, sim, similarity):
    if eval_func == 'bleurt':
        evaluator = load(eval_func, checkpoint='BLEURT-20')
    else:
        evaluator = load(eval_func)

    if eval_func == 'rouge':
        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[[ref]])['rougeL']
    elif eval_func == 'sacrebleu':
        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[ref])['score']
    elif eval_func == 'sacrebleuzh':
        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[ref], tokenize='zh')['score']
    else:
        assert False

    return compute_evaluate, evaluator

def compute_self_score(hyps, src, compute_evaluate):
    scores = []
    n_samples = 0
    n = len(hyps)
    for i in range(n):
        for j in range(n):
            if i != j:
                score = compute_evaluate(hyps[i], hyps[j], src)
                scores.append(score)
                n_samples += 1
    return sum(scores) / n_samples

