"""
Convert a Karpathy-format caption JSON (dataset_coco.json, dataset_flickr8k.json,
dataset_flickr30k.json) into the CSV captions file the evaluator expects.

    python -m rerank.convert_karpathy --json data/dataset_coco.json --out-dir data/coco

Writes two files, covering every split:

    captions.txt   image,caption          (up to 5 rows per image, CSV-quoted)
    manifest.tsv   image <TAB> path <TAB> split

One captions file for the whole dataset is deliberate: the evaluator looks up
references only for the image_ids present in the candidates file, so extra rows
are simply never read. Split membership lives in manifest.tsv, so filtering is a
lookup rather than a different file.

`path` is the image location relative to the image root -- COCO filenames are
bare but the files sit in train2014/ and val2014/, so this is what populates
`image_path` in the candidates JSONL.

Why this conversion is not a straight dump:
  - some captions contain literal newlines and tabs, which would break any
    line-based reader downstream;
  - captions contain commas and quotes, so the CSV must be properly quoted;
  - a few images carry 6-7 reference captions instead of 5, and the extras are
    discarded for consistency across datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter

WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    """Collapse all whitespace (including embedded newlines/tabs) into spaces."""
    return WHITESPACE.sub(" ", text).strip()


def load_karpathy(path: str) -> tuple[list[dict], str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "images" not in data:
        raise ValueError(
            f"{path} has no 'images' key - is this a Karpathy split file? "
            "Expected dataset_coco.json / dataset_flickr8k.json / dataset_flickr30k.json"
        )
    return data["images"], data.get("dataset", "unknown")


def convert(
    images: list[dict],
    splits: set[str] | None = None,
    max_captions: int = 5,
    use_tokens: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]], Counter]:
    """-> (caption rows, manifest rows, stats).

    splits=None keeps every split. Pass a set to restrict.
    """
    caption_rows: list[tuple[str, str]] = []
    manifest_rows: list[tuple[str, str, str]] = []
    stats: Counter = Counter()

    for im in images:
        split = im.get("split", "")
        if splits is not None and split not in splits:
            continue

        name = im["filename"]
        rel = os.path.join(im["filepath"], name) if im.get("filepath") else name
        manifest_rows.append((name, rel, split))
        stats["images"] += 1
        stats[f"split:{split}"] += 1

        sents = im.get("sentences", [])
        if len(sents) > max_captions:
            stats["images_trimmed"] += 1

        kept = 0
        for s in sents[:max_captions]:
            raw = s.get("raw", "")
            text = clean(" ".join(s["tokens"]) if use_tokens else raw)
            if not text:
                stats["empty_skipped"] += 1
                continue
            if "\n" in raw or "\r" in raw or "\t" in raw:
                stats["whitespace_fixed"] += 1
            caption_rows.append((name, text))
            kept += 1

        if kept == 0:
            stats["images_with_no_captions"] += 1
        stats["captions"] += kept

    return caption_rows, manifest_rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Karpathy dataset_*.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument(
        "--splits",
        default="",
        help="optional comma-separated filter, e.g. 'val,test'. "
        "Default: keep every split in one file.",
    )
    ap.add_argument("--max-captions", type=int, default=5)
    ap.add_argument(
        "--use-tokens",
        action="store_true",
        help="use the pre-tokenized lowercased text instead of the raw caption. "
        "Default is raw - the PTB tokenizer in evaluate.py handles the rest.",
    )
    args = ap.parse_args()

    images, dataset = load_karpathy(args.json)
    print(f"{args.json}: {len(images)} images, dataset={dataset}")
    print("splits present:", dict(Counter(i.get("split") for i in images)))

    wanted = (
        {s.strip() for s in args.splits.split(",") if s.strip()}
        if args.splits
        else None
    )
    caps, manifest, stats = convert(images, wanted, args.max_captions, args.use_tokens)
    if not caps:
        print("\n[FAIL] no captions produced - check --splits")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    cap_path = os.path.join(args.out_dir, "captions.txt")
    man_path = os.path.join(args.out_dir, "manifest.tsv")

    with open(cap_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["image", "caption"])
        w.writerows(caps)

    with open(man_path, "w", encoding="utf-8") as f:
        f.write("image\tpath\tsplit\n")
        for row in manifest:
            f.write("\t".join(row) + "\n")

    print(f"\n{stats['images']} images, {stats['captions']} captions "
          f"({stats['captions'] / max(1, stats['images']):.2f} per image)")
    per_split = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("split:")}
    print("  by split:", per_split)
    if stats["images_trimmed"]:
        print(f"  trimmed to {args.max_captions} refs on "
              f"{stats['images_trimmed']} images")
    if stats["whitespace_fixed"]:
        print(f"  collapsed embedded newlines/tabs in "
              f"{stats['whitespace_fixed']} captions")
    if stats["empty_skipped"]:
        print(f"  skipped {stats['empty_skipped']} empty captions")
    if stats["images_with_no_captions"]:
        print(f"  [WARN] {stats['images_with_no_captions']} images have no captions")

    print(f"\nwrote {cap_path}")
    print(f"wrote {man_path}")
    print("\nUse with:")
    print(f"  python -m rerank.run_experiment --captions {cap_path} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())