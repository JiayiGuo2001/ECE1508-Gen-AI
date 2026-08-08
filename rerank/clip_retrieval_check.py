"""
Does CLIP know which caption belongs to which image?

For each of N sampled images, we take one true caption and K-1 distractor
captions stolen from other images, and check whether CLIP ranks the true one
first. Chance is 1/K. If CLIP can't clear this bar, reranking real (much more
similar) beam candidates is hopeless, and the bug is in preprocessing.

Run:
    python -m rerank.clip_retrieval_check --captions captions.txt --images images/

Accepts either Flickr8k caption format:
    captions.txt        image,caption                    (Kaggle CSV)
    Flickr8k.token.txt  image.jpg#0<TAB>caption          (original release)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np

from rerank.clip_reranker import CLIPReranker


# --------------------------------------------------------------------------
def load_captions(path: str) -> dict[str, list[str]]:
    """-> {image_filename: [caption, ...]}. Sniffs Kaggle CSV vs token.txt."""
    refs: dict[str, list[str]] = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        is_csv = "," in first and "\t" not in first

        if is_csv:
            reader = csv.reader(f)
            header = next(reader)
            if header and header[0].strip().lower() not in {"image", "image_name"}:
                f.seek(0)
                reader = csv.reader(f)  # no header after all
            for row in reader:
                if len(row) >= 2 and row[0].strip():
                    refs[row[0].strip()].append(",".join(row[1:]).strip())
        else:
            for line in f:
                if "\t" not in line:
                    continue
                key, cap = line.split("\t", 1)
                refs[key.split("#")[0].strip()].append(cap.strip())

    return dict(refs)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, help="captions.txt or Flickr8k.token.txt")
    ap.add_argument("--images", required=True, help="directory of image files")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--n-candidates", type=int, default=10, help="1 true + K-1 distractors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="cache/clip_img_emb.npz")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("=" * 62)
    print("CLIP retrieval check")
    print("=" * 62)

    refs = load_captions(args.captions)
    print(f"\nLoaded captions for {len(refs)} images from {args.captions}")

    # keep only images that exist on disk
    available = [k for k in sorted(refs) if os.path.exists(os.path.join(args.images, k))]
    if len(available) < args.n_candidates:
        print(f"[FAIL] only {len(available)} images found in {args.images}")
        print("       check --images points at the folder holding the .jpg files")
        return 1
    print(f"[ OK ] {len(available)} of them exist in {args.images}")

    sample = rng.sample(available, min(args.n_images, len(available)))
    print(f"\nTesting {len(sample)} images, {args.n_candidates} candidates each "
          f"(chance = {100 / args.n_candidates:.0f}%)")

    rr = CLIPReranker(cache_path=args.cache)
    print(f"[ OK ] CLIP {rr.model_name} on {rr.device}")

    # Build every candidate list first, then encode everything in two big batches.
    true_caps, cand_lists = [], []
    for img in sample:
        true_cap = rng.choice(refs[img])
        others = [o for o in sample if o != img]
        distractors = [rng.choice(refs[o]) for o in rng.sample(others, args.n_candidates - 1)]
        cands = [true_cap] + distractors
        rng.shuffle(cands)  # true caption must not always sit at index 0
        true_caps.append(true_cap)
        cand_lists.append(cands)

    print("\nEncoding images...")
    img_emb = rr.encode_images([os.path.join(args.images, i) for i in sample])
    rr.save_cache()

    print("Encoding captions...")
    flat = [c for lst in cand_lists for c in lst]
    txt_emb = rr.encode_texts(flat).reshape(len(sample), args.n_candidates, -1)

    # score: (N, K)
    sims = np.einsum("nkd,nd->nk", txt_emb, img_emb)

    ranks, true_scores, best_wrong = [], [], []
    for i, (cands, true_cap) in enumerate(zip(cand_lists, true_caps)):
        t = cands.index(true_cap)
        order = np.argsort(-sims[i])
        ranks.append(int(np.where(order == t)[0][0]) + 1)  # 1 = best
        true_scores.append(float(sims[i][t]))
        wrong = [j for j in order if j != t]
        best_wrong.append(float(sims[i][wrong[0]]))

    ranks = np.array(ranks)
    top1 = float((ranks == 1).mean())
    top3 = float((ranks <= 3).mean())
    mrr = float((1.0 / ranks).mean())

    print("\n--- results ---")
    print(f"  Top-1 accuracy   {top1:.1%}   (chance {1/args.n_candidates:.1%})")
    print(f"  Top-3 accuracy   {top3:.1%}")
    print(f"  Mean recip rank  {mrr:.3f}")
    print(f"  Mean cosine, true caption   {np.mean(true_scores):.3f}")
    print(f"  Mean cosine, best distractor {np.mean(best_wrong):.3f}")
    print(f"  Mean margin                  {np.mean(true_scores) - np.mean(best_wrong):+.3f}")

    # a couple of misses, for intuition about how CLIP fails
    misses = [i for i, r in enumerate(ranks) if r > 1][:3]
    if misses:
        print("\n--- example misses (CLIP preferred a caption from another image) ---")
        for i in misses:
            chosen = cand_lists[i][int(np.argmax(sims[i]))]
            print(f"  {sample[i]}  (true caption ranked {ranks[i]})")
            print(f"    true  : {true_caps[i][:78]}")
            print(f"    picked: {chosen[:78]}")

    print("\n" + "=" * 62)
    if top1 >= 0.80:
        print(f"PASSED - CLIP separates captions cleanly ({top1:.1%} top-1).")
        print("Its scores are trustworthy enough to rerank beam candidates.")
        return 0
    if top1 >= 0.50:
        print(f"MARGINAL - {top1:.1%} top-1, well above chance but weaker")
        print("than expected (~90%). Usable, but check image paths and that you are")
        print("not accidentally feeding grayscale or heavily cropped images.")
        return 0
    print(f"FAILED - {top1:.1%} top-1 is near chance. Something is wired wrong:")
    print("  - are image and caption lists aligned (same order)?")
    print("  - is the similarity matrix transposed?")
    print("  - are embeddings L2-normalized before the dot product?")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
