"""
Run every selector over the same candidate pool and score each one.

    python run_experiment.py --candidates data/mock_candidates.jsonl \\
                             --captions captions.txt --fake-clip
    python run_experiment.py --candidates data/beams.jsonl \\
                             --captions captions.txt --alphas 0,0.25,0.5,0.75,1

--fake-clip swaps in a deterministic random scorer so the pipeline can be
tested with no GPU and no weights. Numbers from a fake-clip run are meaningless
by construction -- it is a plumbing test, not a result.

Writes one CSV row per selector: metrics, agreement with beam top-1, and mean
caption length.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random

import numpy as np

import rerank.selection as sel
from rerank.candidates import load_candidates, load_captions, summarize, validate
from rerank.evaluate import evaluate_captions


class FakeCLIP:
    """Deterministic nonsense scores, for testing the plumbing only."""

    model_name, device = "FAKE", "cpu"

    def score(self, image_path, captions):
        out = []
        for c in captions:
            h = hashlib.md5((image_path + "||" + c).encode()).hexdigest()
            out.append(int(h[:8], 16) / 0xFFFFFFFF * 0.2 + 0.2)  # ~[0.2, 0.4]
        return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="candidates .jsonl")
    ap.add_argument("--captions", required=True, help="ground-truth captions file")
    ap.add_argument("--out", default="results/selectors.csv")
    ap.add_argument("--alphas", default="0.5", help="comma-separated fusion weights")
    ap.add_argument("--metrics", default="bleu,meteor,rouge,cider",
                    help="add 'spice' for the final report; it is slow")
    ap.add_argument("--fake-clip", action="store_true")
    ap.add_argument("--cache", default="cache/clip_img_emb.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 62)
    print("selector comparison")
    print("=" * 62)

    records = load_candidates(args.candidates)
    print("\n" + summarize(records))

    problems = validate(records)
    if problems:
        print(f"\n[WARN] {len(problems)} problem(s) in the candidates file:")
        for p in problems[:10]:
            print("  -", p)

    refs_all = load_captions(args.captions)
    missing = [r["image_id"] for r in records if r["image_id"] not in refs_all]
    if missing:
        print(f"\n[FAIL] {len(missing)} images have no references, e.g. {missing[:3]}")
        return 1
    references = {r["image_id"]: refs_all[r["image_id"]] for r in records}

    if args.fake_clip:
        scorer = FakeCLIP()
        print("\n[WARN] --fake-clip: scores are random. Plumbing test only.")
    else:
        from rerank.clip_reranker import CLIPReranker

        scorer = CLIPReranker(cache_path=args.cache)
        print(f"\n[ OK ] CLIP {scorer.model_name} on {scorer.device}")
        # warm the image cache in one batched pass
        scorer.encode_images([r["image_path"] for r in records])
        scorer.save_cache()

    rng = random.Random(args.seed)

    print("\nRunning selectors...")
    picks: dict[str, dict[str, str]] = {}
    picks["beam_top1"] = {r["image_id"]: sel.select_beam_top1(r) for r in records}
    picks["random"] = {r["image_id"]: sel.select_random(r, rng) for r in records}
    picks["clip"] = {r["image_id"]: sel.select_clip(r, scorer) for r in records}
    for a in [float(x) for x in args.alphas.split(",")]:
        picks[f"fusion_a{a:g}"] = {
            r["image_id"]: sel.select_fusion(r, scorer, a) for r in records
        }
    print("  oracle (this one takes a moment)...")
    picks["oracle"] = sel.select_oracle_all(records, references)

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    rows = []
    for name, chosen in picks.items():
        print(f"\n--- {name} ---")
        scores = evaluate_captions(chosen, references, metrics=metrics, verbose=False)
        row = {"selector": name, **{k: round(v, 4) for k, v in scores.items()}}
        row["agree_with_top1"] = round(sel.agreement(chosen, picks["beam_top1"]), 3)
        row["mean_len"] = round(
            float(np.mean([len(c.split()) for c in chosen.values()])), 2
        )
        rows.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "selector"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = list({k for r in rows for k in r})
    fields.sort(key=lambda k: (k != "selector", k))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")

    # the two comparisons that actually matter
    key = "CIDEr" if "CIDEr" in rows[0] else "Bleu_4"
    by = {r["selector"]: r.get(key, float("nan")) for r in rows}
    print(f"\n--- headroom ({key}) ---")
    print(f"  random     {by.get('random'):.4f}")
    print(f"  beam_top1  {by.get('beam_top1'):.4f}   <- baseline")
    print(f"  clip       {by.get('clip'):.4f}   ({by['clip'] - by['beam_top1']:+.4f})")
    print(f"  oracle     {by.get('oracle'):.4f}   <- ceiling")
    gap = by["oracle"] - by["beam_top1"]
    if gap > 1e-9:
        captured = (by["clip"] - by["beam_top1"]) / gap
        print(f"\n  CLIP captured {captured:.1%} of the available headroom.")
    if gap < 0.02 * max(abs(by["beam_top1"]), 1e-9) + 1e-6:
        print("\n  [WARN] oracle barely beats beam_top1: the candidates are")
        print("         near-duplicates. Ask for diverse beam search or sampling;")
        print("         no reranker can extract signal that isn't there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
