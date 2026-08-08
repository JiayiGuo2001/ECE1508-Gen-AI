"""
Verify the environment before running anything else.

    python -m rerank.check_env

Checks, in order:
  1. Java is present and old enough for SPICE.
  2. Every caption metric returns a value on hand-written toy captions.
  3. CLIP loads and can tell three unambiguous images apart.

Metric scores here are meaningless (5 toy images); only pass/fail matters.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from rerank.evaluate import check_java, evaluate_captions

# --------------------------------------------------------------------------
# 1-2. metrics
# --------------------------------------------------------------------------
REFERENCES = {
    "img1": [
        "a child in a pink dress is climbing up a set of stairs",
        "a little girl in a pink dress going into a wooden cabin",
        "a girl going into a wooden building",
        "a little girl climbing into a wooden playhouse",
        "a little girl climbing the stairs to her playhouse",
    ],
    "img2": [
        "a black dog is running through the snow",
        "a dog runs across the snowy field",
        "a black dog running in deep snow",
        "a dog is running in the snow",
        "a black dog runs through a snow covered field",
    ],
    "img3": [
        "two men are playing guitars on a stage",
        "a pair of musicians perform on stage with guitars",
        "two guitarists playing at a concert",
        "two men play guitar in front of a crowd",
        "two male musicians playing guitars on stage",
    ],
    "img4": [
        "a group of people sitting on a boat in the water",
        "several people ride in a small green boat",
        "people in a boat on a lake",
        "a group of people riding a boat",
        "some people sitting in a boat on the water",
    ],
    "img5": [
        "a stop sign on a road with mountains in the background",
        "a red stop sign near a green field",
        "a stop sign beside a rural road",
        "a stop sign with hills behind it",
        "a red octagonal stop sign on a country road",
    ],
}

# img5 is deliberately wrong and must score far below the rest.
PREDICTIONS = {
    "img1": "a little girl climbing into a wooden playhouse",
    "img2": "a black dog is running through the snow",
    "img3": "two men playing guitars on a stage",
    "img4": "people riding in a boat on the water",
    "img5": "a man is eating a sandwich in a kitchen",
}


def check_metrics() -> bool:
    print("\n--- metrics ---")
    scores = evaluate_captions(PREDICTIONS, REFERENCES, verbose=False)
    for k in ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"):
        print(f"  {k:<9} {scores[k]:.4f}" if k in scores else f"  {k:<9} MISSING")

    missing = {"Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"} - set(scores)
    if missing:
        print(f"\n[WARN] no value from: {sorted(missing)}")
        print("       SPICE needs Java <= 15 and ~1GB of Stanford CoreNLP on first")
        print("       run. The other metrics are still usable without it.")

    good = evaluate_captions(
        {"img1": PREDICTIONS["img1"]}, {"img1": REFERENCES["img1"]},
        metrics=("bleu", "rouge"), verbose=False,
    )
    bad = evaluate_captions(
        {"img5": PREDICTIONS["img5"]}, {"img5": REFERENCES["img5"]},
        metrics=("bleu", "rouge"), verbose=False,
    )
    ok = good["ROUGE_L"] > bad["ROUGE_L"] + 0.3
    print(f"\n  exact-match caption  ROUGE_L={good['ROUGE_L']:.3f}")
    print(f"  wrong caption        ROUGE_L={bad['ROUGE_L']:.3f}")
    print(f"[{'  OK' if ok else 'FAIL'}] scorers separate good captions from bad")
    return ok


# --------------------------------------------------------------------------
# 3. CLIP
# --------------------------------------------------------------------------
SHAPES = {
    "red_square": "a red square",
    "blue_circle": "a blue circle",
    "green_triangle": "a green triangle",
}


def make_test_images(outdir: str) -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    specs = [
        ("red_square", "rectangle", [50, 50, 174, 174], "red"),
        ("blue_circle", "ellipse", [50, 50, 174, 174], "blue"),
        ("green_triangle", "polygon", [(112, 45), (180, 180), (44, 180)], "green"),
    ]
    for name, kind, coords, colour in specs:
        img = Image.new("RGB", (224, 224), "white")
        getattr(ImageDraw.Draw(img), kind)(coords, fill=colour)
        paths[name] = os.path.join(outdir, f"{name}.png")
        img.save(paths[name])
    return paths


def check_clip() -> bool:
    print("\n--- CLIP ---")
    try:
        from rerank.clip_reranker import CLIPReranker
    except ImportError as e:
        print(f"[FAIL] {e}")
        print("       pip install git+https://github.com/openai/CLIP.git")
        return False

    names = list(SHAPES)
    paths = make_test_images(tempfile.mkdtemp(prefix="clip_check_"))

    try:
        rr = CLIPReranker()
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] could not load CLIP: {e}")
        return False
    print(f"[ OK ] {rr.model_name} on {rr.device}")

    img_emb = rr.encode_images([paths[n] for n in names])
    txt_emb = rr.encode_texts([SHAPES[n] for n in names])
    sims = img_emb @ txt_emb.T

    norms = np.concatenate(
        [np.linalg.norm(img_emb, axis=1), np.linalg.norm(txt_emb, axis=1)]
    )
    unit = bool(np.allclose(norms, 1.0, atol=1e-3))
    print(f"[{'  OK' if unit else 'FAIL'}] embeddings unit norm "
          f"(dot product == cosine)")

    correct = int((sims.argmax(axis=1) == np.arange(len(names))).sum())
    print(f"[{'  OK' if correct == len(names) else 'FAIL'}] "
          f"{correct}/{len(names)} images matched their own caption")
    print(f"  cosine range {sims.min():.3f} to {sims.max():.3f}")

    nan_free = not (np.isnan(img_emb).any() or np.isnan(txt_emb).any())
    if not nan_free:
        print("[FAIL] NaNs in embeddings")
    return unit and correct == len(names) and nan_free


# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 62)
    print("environment check")
    print("=" * 62)

    if not check_java():
        print("\nSTOP: install java. METEOR, SPICE and the tokenizer need it.")
        return 1

    ok = check_metrics()
    ok = check_clip() and ok

    print("\n" + "=" * 62)
    print("ENVIRONMENT OK" if ok else "ENVIRONMENT NOT READY - see failures above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
