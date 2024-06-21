import sys
from pathlib import Path
from typing import List, Tuple

import sacrebleu
from sacrebleu.metrics.bleu import BLEUScore
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import torch
from bert_score import score as bertscore_score
from bleurt import score as bleurt_score
import warnings

warnings.simplefilter('ignore')


def get_parallel(
    path_to_ref_txt: str,
    path_to_hyp_txt: str,
) -> Tuple[List[str], List[str]]:

    with open(path_to_ref_txt, "r", encoding="utf-8") as f:
        reference = f.read().splitlines()
    with open(path_to_hyp_txt, "r", encoding="utf-8") as f:
        hypothesis = f.read().splitlines()
    assert len(reference) == len(hypothesis)

    return reference, hypothesis


def score_sacrebleu(
    path_to_ref_txt: str,
    path_to_hyp_txt: str,
) -> BLEUScore:

    reference, hypothesis = get_parallel(path_to_ref_txt, path_to_hyp_txt)
    bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
    ter = sacrebleu.corpus_ter(hypothesis, [reference])
    print(bleu)
    print(ter)
    return bleu


def score_sentence_bleu_p1(
    path_to_ref_txt: str,
    path_to_hyp_txt: str,
    path_to_output: str,
) -> List[str]:
    reference, hypothesis = get_parallel(path_to_ref_txt, path_to_hyp_txt)
    scores = []
    for i in range(len(reference)):
        score = sentence_bleu(
            [reference[i]], hypothesis[i],
            smoothing_function=SmoothingFunction().method2,
            weights=(0.25, 0.25, 0.25, 0.25),
        )
        scores.append(str(score))
    with open(path_to_output, mode="w") as f:
        f.write("\n".join(scores))
    return scores


def score_bertscore(
    path_to_ref_txt: str,
    path_to_hyp_txt: str,
    lang: str = "de",
) -> Tuple[float, float, float]:

    reference, hypothesis = get_parallel(path_to_ref_txt, path_to_hyp_txt)
    p, r, f1 = bertscore_score(hypothesis, reference, lang=lang,
        rescale_with_baseline=True, verbose=False)
    p = torch.mean(p).item()
    r = torch.mean(r).item()
    f1 = torch.mean(f1).item()
    print(f"BERTScore (P/R/F1) = {p:.4f}/{r:.4f}/{f1:.4f}")
    return p, r, f1


def score_sentence_bertscore(
    path_to_ref_txt: str,
    path_to_hyp_txt: str,
    path_to_output_pref: str,
    lang: str = "de",
) -> Tuple[List[str], List[str], List[str]]:

    reference, hypothesis = get_parallel(path_to_ref_txt, path_to_hyp_txt)
    p, r, f1 = bertscore_score(hypothesis, reference, lang=lang,
        rescale_with_baseline=True, verbose=False)
    p_l = [str(i) for i in p.numpy()]
    r_l = [str(i) for i in r.numpy()]
    f1_l = [str(i) for i in f1.numpy()]
    with open(str(path_to_output_pref) + ".P", mode="w") as f:
        f.write("\n".join(p_l))
    with open(str(path_to_output_pref) + ".R", mode="w") as f:
        f.write("\n".join(r_l))
    with open(str(path_to_output_pref) + ".F1", mode="w") as f:
        f.write("\n".join(f1_l))
    return p_l, r_l, f1_l


def score_bleurt(
    path_to_ref_txt: str,
    path_to_hyp_txt: str,
    checkpoint: str,
) -> float:

    reference, hypothesis = get_parallel(path_to_ref_txt, path_to_hyp_txt)
    scorer = bleurt_score.BleurtScorer(checkpoint)
    scores = scorer.score(references=reference, candidates=hypothesis)
    score = np.mean(scores)
    
    print(f"BLEURT (Average) = {score:.4f}")

    return score

if __name__ == "__main__":
    import sys
    score_bleurt(sys.argv[1], sys.argv[2], sys.argv[3])
