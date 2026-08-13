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
    ap.add_argument("--compare", default=None, metavar="PATH",
                    help="a second model's candidates .jsonl (e.g. BLIP). Adds "
                         "its top-1 as an extra row after the oracle.")
    ap.add_argument("--compare-name", default="blip",
                    help="row label for --compare (default: blip)")
    ap.add_argument("--samples", type=int, default=5,
                    help="side-by-side examples to print when using --compare")
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

    # ---- optional second model (e.g. BLIP) --------------------------------
    # Every other row in this CSV is a *selector over our own candidate pool*.
    # The --compare row is a different axis: a different model entirely. It is
    # kept in the same table because that is the comparison a reader wants, but
    # it is tagged in a `pool` column so the two axes cannot be confused.
    compare_records = None
    if args.compare:
        from baselines.blip_captioner import normalize_style

        compare_records = load_candidates(args.compare)
        print(f"\n[ OK ] {args.compare_name}: {len(compare_records)} images "
              f"({args.compare})")

        # Style-normalize BOTH pools identically, so casing/punctuation cannot
        # show up as a metric difference. Our decoder already emits lowercase
        # unpunctuated text, so this is a no-op on our side -- the existing
        # rows keep the numbers you have already reported.
        for pool in (records, compare_records):
            for r in pool:
                for c in r["candidates"]:
                    c["text"] = normalize_style(c["text"])

        # Both models must be scored on exactly the same images or the rows are
        # not comparable. Restrict to the intersection and say so loudly.
        shared = {r["image_id"] for r in records} & \
                 {r["image_id"] for r in compare_records}
        if not shared:
            print("[FAIL] the two candidate files share no image_ids.")
            return 1
        if len(shared) != len(records):
            print(f"[WARN] restricting ALL rows to the {len(shared)} images "
                  f"both models cover (was {len(records)}). Every number in "
                  f"this CSV is on that subset, including the selector rows, "
                  f"so it will not match a full-set run.")
        records = [r for r in records if r["image_id"] in shared]
        compare_records = [r for r in compare_records if r["image_id"] in shared]

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

    # (name, pool, captions). The compare row goes last, after the oracle.
    entries = [(name, "ours", chosen) for name, chosen in picks.items()]
    if compare_records is not None:
        entries.append((
            f"{args.compare_name}_top1",
            args.compare_name,
            {r["image_id"]: sel.select_beam_top1(r) for r in compare_records},
        ))

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    rows = []
    for name, pool, chosen in entries:
        print(f"\n--- {name} ---")
        scores = evaluate_captions(chosen, references, metrics=metrics, verbose=False)
        row = {"selector": name, "pool": pool,
               **{k: round(v, 4) for k, v in scores.items()}}
        if pool == "ours":
            row["agree_with_top1"] = round(
                sel.agreement(chosen, picks["beam_top1"]), 3)
        # else: left blank on purpose. Agreement measures which candidate a
        # selector picked from a shared pool; against a different model's
        # captions the number would be string overlap, not selector behaviour.
        row["mean_len"] = round(
            float(np.mean([len(c.split()) for c in chosen.values()])), 2
        )
        rows.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "selector"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = list({k for r in rows for k in r})
    fields.sort(key=lambda k: (k != "selector", k != "pool", k))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
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

    # ---- gap to the comparison model --------------------------------------
    if compare_records is not None:
        other = by.get(f"{args.compare_name}_top1")
        best_ours = max(v for k, v in by.items()
                        if k != "oracle" and not k.startswith(args.compare_name))
        print(f"\n--- vs {args.compare_name} ({key}) ---")
        print(f"  our best selector  {best_ours:.4f}")
        print(f"  our oracle         {by['oracle']:.4f}   <- ceiling of OUR pool")
        print(f"  {args.compare_name + '_top1':<18} {other:.4f}")
        if other:
            print(f"\n  We reach {100 * best_ours / other:.1f}% of "
                  f"{args.compare_name}'s {key}.")
            if by["oracle"] > other:
                print(f"  Note: our oracle exceeds {args.compare_name}. A perfect "
                      f"reranker over our\n        existing candidates would "
                      f"close the whole gap -- the deficit is\n        selection, "
                      f"not generation.")
            else:
                print(f"  Note: even our oracle trails {args.compare_name}, so the "
                      f"gap is not something\n        reranking alone can close.")

        if args.samples:
            comp = {r["image_id"]: sel.select_beam_top1(r) for r in compare_records}
            print("\n--- side by side ---")
            for iid in sorted(references)[:args.samples]:
                print(f"\n{iid}")
                print(f"  {'ours':<6} {picks['beam_top1'][iid]}")
                print(f"  {args.compare_name:<6} {comp[iid]}")
                print(f"  {'REF':<6} {references[iid][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())