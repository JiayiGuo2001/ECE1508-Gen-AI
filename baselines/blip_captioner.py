"""
BLIP baseline captioner.

Generates candidate captions with BLIP (COCO-finetuned) and writes them in
exactly the JSONL contract from docs/candidate_format.md, so the output flows
through rerank/ and baselines/compare.py unchanged:

    {"image_id": ..., "image_path": ..., "candidates": [{"text": ..., "logprob": ...}]}

The image set comes from an existing candidates file (--like), which guarantees
we score both models on identical images -- the easiest thing to get wrong in a
model-vs-model table.

Runs on CUDA, Apple silicon (MPS) or CPU -- detected automatically.

Usage
-----
    # CUDA box
    python -m baselines.blip_captioner \
        --like beams/coco_beams.jsonl \
        --out  beams/coco_blip.jsonl \
        --fp16

    # MacBook Pro (MPS). Leave --fp16 off; see the dtype note in pick_dtype().
    python -m baselines.blip_captioner \
        --like beams/coco_beams.jsonl \
        --out  beams/coco_blip.jsonl \
        --batch-size 8

    # cheaper: swap in the base checkpoint, or cap the image count
    python -m baselines.blip_captioner --like beams/flicker30k_beams.jsonl \
        --out beams/flickr30k_blip.jsonl \
        --checkpoint Salesforce/blip-image-captioning-base --limit 1000

Requires:  pip install transformers accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must be set BEFORE torch is imported (torch reads it at import time). Lets
# any op without an MPS kernel fall back to CPU instead of hard-crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_CHECKPOINT = "Salesforce/blip-image-captioning-large"

# BLIP sometimes emits these artifacts from its pretraining data.
BLIP_ARTIFACTS = ("arafed ", "araffe ", "araffes ", "a photography of ",
                  "a photo of ", "an image of ", "a picture of ")


def normalize_style(text: str) -> str:
    """COCO/Flickr references are lowercase, unpunctuated fragments; BLIP adds
    capitals, periods and the occasional boilerplate prefix. Normalizing keeps
    the comparison about content rather than typography.

    compare.py applies this to OUR captions too, so it cannot tilt the result
    in either direction.
    """
    t = " ".join(text.strip().split()).lower()
    for prefix in BLIP_ARTIFACTS:
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    return t.rstrip(" .").strip()


def pick_device(requested: str | None = None) -> str:
    """cuda > mps > cpu. Mirrors rerank.clip_reranker.pick_device, so the
    captioner and the reranker land on the same backend."""
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, fp16: bool):
    """float16 is a real speedup on CUDA. On MPS it is a coin flip: several
    ops still fall back to CPU in fp16, and beam search over fp16 logits on
    MPS has produced NaN scores (which silently become empty captions). So
    --fp16 is honoured on MPS but warned about, and float32 is the default.
    """
    import torch

    if not fp16:
        return torch.float32
    if device == "cuda":
        return torch.float16
    if device == "mps":
        print("[WARN] --fp16 on MPS: if captions come out empty or log-probs "
              "are nan, re-run without it.")
        return torch.float16
    print("[WARN] --fp16 ignored on CPU (float16 CPU matmul is slower).")
    return torch.float32


def cumulative_logprobs(model, out):
    """Raw (un-normalized, natural-log) sequence log-probs, one per returned
    beam. The fusion selector length-normalizes itself and so wants the
    cumulative sum -- see docs/candidate_format.md. `sequences_scores` is
    already divided by length, hence compute_transition_scores instead.
    """
    import torch

    try:
        scores = model.compute_transition_scores(
            sequences=out.sequences,
            scores=out.scores,
            beam_indices=out.beam_indices,
            normalize_logits=False,  # beam-search scores are already log-softmax
        )
        return torch.nan_to_num(scores, neginf=0.0, nan=0.0).sum(-1).tolist()
    except Exception as e:  # noqa: BLE001 -- transformers version differences
        print(f"[WARN] falling back to sequences_scores: {e}")
        lengths = (out.sequences != model.config.text_config.pad_token_id) \
            .sum(-1).clamp(min=1)
        return (out.sequences_scores * lengths).tolist()


def generate(records, checkpoint, beam_size, batch_size, max_new_tokens,
             diversity_penalty, fp16, device):
    """Caption every record (needs image_id + image_path); return new records."""
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = pick_device(device)
    dtype = pick_dtype(device, fp16)

    print(f"[..] loading {checkpoint} on {device} ({dtype})")
    if device == "mps":
        print("     Apple silicon: expect ~2-4x slower than a T4. If you run "
              "out of memory, drop --batch-size to 4 or use the base checkpoint.")
    processor = BlipProcessor.from_pretrained(checkpoint)
    model = BlipForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=dtype).to(device).eval()

    gen_kwargs = dict(
        num_beams=beam_size,
        num_return_sequences=beam_size,
        max_new_tokens=max_new_tokens,
        length_penalty=1.0,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if diversity_penalty > 0:
        # Diverse beam search (k groups of 1). Needed if you also want to run
        # the reranking selectors on BLIP's candidates -- plain beam search
        # returns five near-identical strings and there is nothing to select.
        gen_kwargs.update(num_beam_groups=beam_size,
                          diversity_penalty=diversity_penalty)

    out_records = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        images, kept = [], []
        for r in batch:
            try:
                images.append(Image.open(r["image_path"]).convert("RGB"))
                kept.append(r)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] skipping {r['image_id']}: {e}")
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        logprobs = cumulative_logprobs(model, out)
        decoded = processor.batch_decode(out.sequences, skip_special_tokens=True)

        for i, r in enumerate(kept):
            candidates, seen = [], set()
            for j in range(beam_size):
                idx = i * beam_size + j
                text = normalize_style(decoded[idx])
                if text and text not in seen:
                    seen.add(text)
                    candidates.append({"text": text,
                                       "logprob": float(logprobs[idx])})
            if candidates:
                out_records.append({"image_id": r["image_id"],
                                    "image_path": r["image_path"],
                                    "candidates": candidates})
            else:
                print(f"[WARN] {r['image_id']}: no usable candidates")

        if device == "mps" and hasattr(torch, "mps"):
            # MPS shares system RAM and its allocator holds on to freed blocks;
            # on a long run this creeps until the machine starts swapping.
            # (torch.mps landed in 2.0; guarded so older installs still run.)
            torch.mps.empty_cache()

        done = min(start + batch_size, len(records))
        print(f"  {done}/{len(records)} images", end="\r", flush=True)

    print()
    return out_records


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--like", required=True,
                    help="existing candidates .jsonl -- defines the image set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--images-root", default="",
                    help="prefix for image_path if the images have moved")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--diversity-penalty", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--fp16", action="store_true",
                    help="half precision; recommended on CUDA, risky on MPS")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="default: auto-detect (cuda > mps > cpu)")
    args = ap.parse_args()

    with open(args.like, encoding="utf-8") as f:
        source = [json.loads(ln) for ln in f if ln.strip()]
    if args.limit:
        source = source[:args.limit]

    records = [
        {"image_id": r["image_id"],
         "image_path": os.path.join(args.images_root, r["image_path"])
         if args.images_root else r["image_path"]}
        for r in source
    ]
    print(f"[ OK ] {len(records)} images from {args.like}")

    missing = [r for r in records if not os.path.exists(r["image_path"])]
    if missing:
        print(f"[FAIL] {len(missing)} images not found, e.g. "
              f"{missing[0]['image_path']}. Use --images-root.")
        return 1

    out_records = generate(records, args.checkpoint, args.beam_size,
                           args.batch_size, args.max_new_tokens,
                           args.diversity_penalty, args.fp16, args.device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}  ({len(out_records)} records)")

    mean_unique = sum(len({c["text"] for c in r["candidates"]})
                      for r in out_records) / max(len(out_records), 1)
    print(f"mean unique candidates/image: {mean_unique:.2f}")
    if mean_unique < 1.5 and args.beam_size > 1:
        print("[WARN] near-duplicate candidates. Fine for a top-1 comparison; "
              "use --diversity-penalty 0.5 if you want to rerank these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())