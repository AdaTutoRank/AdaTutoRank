"""
Open-domain QA answer metrics, each taking the max over all gold aliases.

    exact_match     normalised prediction == normalised gold
    f1/precision/recall  token overlap after normalisation
    contains_match  normalised gold is a substring of the normalised prediction
                    (reported as "EM" in the paper table)
"""

import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple


def normalize_answer(s: str) -> str:
    """Lowercase, drop non-alphanumerics, drop articles, squash whitespace."""
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return 1.0 if any(normalize_answer(g) == pred for g in golds) else 0.0


def contains_match(prediction: str, golds: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return 1.0 if any(normalize_answer(g) in pred for g in golds) else 0.0


def token_scores(prediction: str, golds: Sequence[str]) -> Dict[str, float]:
    """Best token-level f1 / precision / recall over the gold aliases (each maximised separately)."""
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    pred_tokens = normalize_answer(prediction).split()
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best["f1"] = max(best["f1"], f1)
        best["precision"] = max(best["precision"], precision)
        best["recall"] = max(best["recall"], recall)
    return best


def generation_metrics(
    predictions: Sequence[str], golds: Sequence[Sequence[str]]
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Returns:
        summary: corpus averages in percent (exact_match, f1_score, precision,
            recall, contains_match)
        per_sample: per-sample exact_match / f1_score / contains_match in percent
    """
    per_sample = []
    totals = {"exact_match": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "contains_match": 0.0}
    for pred, gold in zip(predictions, golds):
        pred = str(pred)
        em = exact_match(pred, gold)
        tok = token_scores(pred, gold)
        cm = contains_match(pred, gold)
        totals["exact_match"] += em
        totals["f1_score"] += tok["f1"]
        totals["precision"] += tok["precision"]
        totals["recall"] += tok["recall"]
        totals["contains_match"] += cm
        per_sample.append({"exact_match": em * 100, "f1_score": tok["f1"] * 100, "contains_match": cm * 100})
    n = max(len(per_sample), 1)
    summary = {k: v / n * 100 for k, v in totals.items()}
    return summary, per_sample
