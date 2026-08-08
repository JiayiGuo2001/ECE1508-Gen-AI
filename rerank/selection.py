"""
Selection strategies. Each turns a candidate list into ONE caption.

Every selector has the same job, so swapping one for another is the only thing
that changes between experiment rows. That is what makes the comparison fair.

    beam_top1  what the model does without you       -> the baseline to beat
    random     picks a candidate at random           -> is the pool any good?
    clip       highest CLIP cosine                   -> your method
    fusion     CLIP + length-normalized log-prob     -> usually beats pure CLIP
    oracle     best candidate by CIDEr vs references -> the ceiling

The oracle is the most informative row. If oracle barely beats beam_top1, the
candidates are near-duplicates and NO reranker can help; ask your teammate for
diverse beam search before tuning anything.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
def texts(record: dict) -> list[str]:
    return [c["text"] for c in record["candidates"]]


def zscore(x: np.ndarray) -> np.ndarray:
    """Center/scale within one candidate list. Flat lists -> all zeros."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-8 else (x - x.mean()) / sd


def length_normalized_logprobs(record: dict) -> np.ndarray:
    """Raw cumulative log-prob always favours short captions. Divide by tokens."""
    return np.array(
        [c["logprob"] / max(1, len(c["text"].split())) for c in record["candidates"]],
        dtype=float,
    )


# --------------------------------------------------------------------------
# per-record selectors
# --------------------------------------------------------------------------
def select_beam_top1(record: dict) -> str:
    """Highest decoder log-prob. Candidates arrive beam-ordered, but don't
    rely on that -- take the argmax explicitly."""
    lp = [c["logprob"] for c in record["candidates"]]
    return record["candidates"][int(np.argmax(lp))]["text"]


def select_random(record: dict, rng) -> str:
    return rng.choice(texts(record))


def select_clip(record: dict, scorer) -> str:
    sims = scorer.score(record["image_path"], texts(record))
    return texts(record)[int(np.argmax(sims))]


def select_fusion(record: dict, scorer, alpha: float = 0.5) -> str:
    """alpha=1 is pure CLIP, alpha=0 is pure (length-normalized) decoder score.

    Both terms are z-scored within the candidate list first: CLIP cosines sit in
    a narrow band around 0.2-0.35 while log-probs are large negatives, so a raw
    weighted sum would be dominated by whichever has the bigger spread.
    """
    clip_z = zscore(scorer.score(record["image_path"], texts(record)))
    lm_z = zscore(length_normalized_logprobs(record))
    return texts(record)[int(np.argmax(alpha * clip_z + (1 - alpha) * lm_z))]


# --------------------------------------------------------------------------
# oracle (batch: needs the whole corpus for CIDEr's IDF weights)
# --------------------------------------------------------------------------
def select_oracle_all(
    records: list[dict],
    references: dict[str, list[str]],
    metric: str = "cider",
) -> dict[str, str]:
    """-> {image_id: best-scoring candidate}.

    Scores candidate slot j for every image in one pass, K passes total, then
    takes the per-image argmax. Doing it this way (rather than one image at a
    time) keeps CIDEr's IDF statistics computed over the full corpus, which is
    what the reported CIDEr will use.
    """
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    if metric == "cider":
        from pycocoevalcap.cider.cider import Cider

        scorer = Cider()
    else:
        from pycocoevalcap.bleu.bleu import Bleu

        scorer = Bleu(4)

    ids = [r["image_id"] for r in records]
    k_max = max(len(r["candidates"]) for r in records)
    tok = PTBTokenizer()

    gts = tok.tokenize(
        {i: [{"caption": c} for c in references[i]] for i in ids}
    )

    per_slot = []
    for j in range(k_max):
        res = tok.tokenize(
            {
                r["image_id"]: [
                    {"caption": r["candidates"][min(j, len(r["candidates"]) - 1)]["text"]}
                ]
                for r in records
            }
        )
        _, scores = scorer.compute_score(gts, res)
        scores = np.asarray(scores[-1] if isinstance(scores, list) else scores, float)
        per_slot.append(scores)

    mat = np.stack(per_slot, axis=1)  # (N, k_max)
    best = {}
    for n, r in enumerate(records):
        valid = mat[n, : len(r["candidates"])]
        best[r["image_id"]] = r["candidates"][int(np.argmax(valid))]["text"]
    return best


# --------------------------------------------------------------------------
def agreement(a: dict[str, str], b: dict[str, str]) -> float:
    """Fraction of images where two selectors chose the same caption."""
    keys = set(a) & set(b)
    return float(np.mean([a[k] == b[k] for k in keys])) if keys else 0.0
