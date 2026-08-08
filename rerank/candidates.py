"""
The candidate-list contract between the captioning model and the reranker.

One JSON object per line:

    {"image_id": "1000268201_693b08cb0e.jpg",
     "image_path": "Flicker8k_Dataset/1000268201_693b08cb0e.jpg",
     "candidates": [{"text": "a child climbing stairs", "logprob": -8.21},
                    {"text": "a little girl in a playhouse", "logprob": -9.03}]}

The ONLY thing your teammate has to change: return the full n-best beam with
each beam's cumulative log-probability, instead of just top-1.

`make_mock_candidates` fabricates realistic lists from ground-truth captions so
the whole pipeline can be built and tested before their model exists.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict


# --------------------------------------------------------------------------
# captions file
# --------------------------------------------------------------------------
def load_captions(path: str) -> dict[str, list[str]]:
    """-> {image_filename: [caption, ...]}. Sniffs Kaggle CSV vs token.txt."""
    refs: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if "," in first and "\t" not in first:
            reader = csv.reader(f)
            header = next(reader)
            if header and header[0].strip().lower() not in {"image", "image_name"}:
                f.seek(0)
                reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0].strip():
                    refs[row[0].strip()].append(",".join(row[1:]).strip())
        else:
            for line in f:
                if "\t" in line:
                    key, cap = line.split("\t", 1)
                    refs[key.split("#")[0].strip()].append(cap.strip())
    return dict(refs)


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def load_candidates(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def save_candidates(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def validate(records: list[dict], images_dir: str | None = None) -> list[str]:
    """-> list of human-readable problems. Empty list means the file is usable."""
    problems = []
    seen = set()
    for i, r in enumerate(records):
        where = f"record {i} ({r.get('image_id', '?')})"
        for key in ("image_id", "candidates"):
            if key not in r:
                problems.append(f"{where}: missing '{key}'")
        if r.get("image_id") in seen:
            problems.append(f"{where}: duplicate image_id")
        seen.add(r.get("image_id"))

        cands = r.get("candidates") or []
        if not cands:
            problems.append(f"{where}: empty candidate list")
        for c in cands:
            if not c.get("text", "").strip():
                problems.append(f"{where}: candidate with empty text")
            if "logprob" not in c:
                problems.append(f"{where}: candidate missing 'logprob' "
                                "(needed by the fusion selector)")
                break
        if len({c.get("text") for c in cands}) == 1 and len(cands) > 1:
            problems.append(f"{where}: all candidates identical - "
                            "beam search is not producing diversity")
        if images_dir and r.get("image_id"):
            p = r.get("image_path") or os.path.join(images_dir, r["image_id"])
            if not os.path.exists(p):
                problems.append(f"{where}: image not found at {p}")
    return problems


def summarize(records: list[dict]) -> str:
    ks = [len(r["candidates"]) for r in records]
    uniq = [len({c["text"] for c in r["candidates"]}) for r in records]
    lens = [len(c["text"].split()) for r in records for c in r["candidates"]]
    return (
        f"{len(records)} images | candidates/image {min(ks)}-{max(ks)} "
        f"(mean {sum(ks)/len(ks):.1f}) | unique/image mean {sum(uniq)/len(uniq):.1f} "
        f"| caption length mean {sum(lens)/len(lens):.1f} words"
    )


# --------------------------------------------------------------------------
# mock generator
# --------------------------------------------------------------------------
def _drop_word(words, rng):
    if len(words) < 4:
        return words
    i = rng.randrange(1, len(words))
    return words[:i] + words[i + 1 :]


def _truncate(words, rng):
    return words[: max(3, len(words) - rng.randint(2, 4))]


def _swap_word(words, donor_words, rng):
    if len(words) < 3 or len(donor_words) < 3:
        return words
    out = list(words)
    out[rng.randrange(len(out))] = donor_words[rng.randrange(len(donor_words))]
    return out


def make_mock_candidates(
    refs: dict[str, list[str]],
    images_dir: str,
    n_images: int = 200,
    k: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Fabricate candidate lists that behave like beam search output.

    Each list mixes: a ground-truth caption, mild corruptions of it, a second
    ground-truth caption, and one caption lifted from another image. Log-probs
    are length-driven and noisy, so beam top-1 is NOT the best candidate --
    which is exactly the headroom the oracle selector should reveal.
    """
    rng = random.Random(seed)
    keys = [key for key in sorted(refs) if refs[key]]
    if images_dir:
        exist = [key for key in keys if os.path.exists(os.path.join(images_dir, key))]
        keys = exist or keys
    keys = rng.sample(keys, min(n_images, len(keys)))

    records = []
    for key in keys:
        pool = refs[key]
        good = rng.choice(pool)
        other_key = rng.choice([o for o in keys if o != key])
        foreign = rng.choice(refs[other_key])

        variants = [good.split()]
        variants.append(_drop_word(good.split(), rng))
        variants.append(_truncate(good.split(), rng))
        if len(pool) > 1:
            variants.append(rng.choice([p for p in pool if p != good]).split())
        variants.append(_swap_word(good.split(), foreign.split(), rng))
        variants.append(foreign.split())

        rng.shuffle(variants)
        texts, seen = [], set()
        for v in variants:
            t = " ".join(v).strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                texts.append(t)
            if len(texts) == k:
                break

        cands = [
            {
                "text": t,
                # longer captions cost more log-prob; noise keeps ordering imperfect
                "logprob": round(-0.85 * len(t.split()) + rng.gauss(0, 1.2), 3),
            }
            for t in texts
        ]
        cands.sort(key=lambda c: -c["logprob"])  # beam order: best first
        records.append(
            {
                "image_id": key,
                "image_path": os.path.join(images_dir, key) if images_dir else key,
                "candidates": cands,
            }
        )
    return records


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate or inspect a candidates file.")
    ap.add_argument("--captions", help="captions.txt / Flickr8k.token.txt (to generate)")
    ap.add_argument("--images", default="", help="image directory")
    ap.add_argument("--out", default="data/mock_candidates.jsonl")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", help="validate an existing candidates file instead")
    args = ap.parse_args()

    if args.check:
        recs = load_candidates(args.check)
        print(summarize(recs))
        probs = validate(recs, args.images or None)
        if probs:
            print(f"\n{len(probs)} problem(s):")
            for p in probs[:20]:
                print("  -", p)
            raise SystemExit(1)
        print("\nno problems found")
    else:
        if not args.captions:
            ap.error("--captions is required unless --check is given")
        refs = load_captions(args.captions)
        recs = make_mock_candidates(refs, args.images, args.n_images, args.k, args.seed)
        save_candidates(recs, args.out)
        print(f"wrote {args.out}")
        print(summarize(recs))
