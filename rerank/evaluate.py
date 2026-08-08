"""
Caption evaluation wrapper over pycocoevalcap.

Everything in the pipeline feeds into `evaluate_captions`.

Contract:
    predictions: {image_id: "one caption string"}
    references:  {image_id: ["ref 1", "ref 2", ...]}
Both dicts must have the same keys.

Returns a flat dict, e.g.
    {"Bleu_1": .., "Bleu_4": .., "METEOR": .., "ROUGE_L": .., "CIDEr": .., "SPICE": ..}
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings

DEFAULT_METRICS = ("bleu", "meteor", "rouge", "cider", "spice")

# SPICE's serialization library (FST) reflects into private JDK internals.
# Java <=15 allowed this; Java 16+ made it a hard error. These flags reopen
# the packages FST touches. Best-effort only -- see java_major() note below.
_SPICE_ADD_OPENS = " ".join(
    f"--add-opens java.base/{pkg}=ALL-UNNAMED"
    for pkg in (
        "java.lang",
        "java.lang.reflect",
        "java.lang.invoke",
        "java.util",
        "java.util.concurrent",
        "java.io",
        "java.net",
        "java.math",
        "java.text",
        "java.time",
    )
)


# --------------------------------------------------------------------------
# environment check
# --------------------------------------------------------------------------
def check_java(verbose: bool = True) -> bool:
    """METEOR, SPICE and the PTB tokenizer all shell out to java."""
    if shutil.which("java") is None:
        if verbose:
            print("[FAIL] java not found on PATH.")
            print("       Colab:  !apt-get -qq install -y default-jre")
        return False
    try:
        out = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=30
        )
        lines = ((out.stderr or "") + (out.stdout or "")).strip().splitlines()
        version_line = next(
            (ln for ln in lines if 'version "' in ln), lines[0] if lines else "?"
        )
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[FAIL] java present but not runnable: {e}")
        return False
    if verbose:
        print(f"[ OK ] java: {version_line}")
        major = java_major()
        if major is not None and major >= 16:
            print(f"[WARN] Java {major} enforces strong module encapsulation; SPICE")
            print("       (but not BLEU/METEOR/ROUGE/CIDEr) will likely crash.")
            print("       Reliable fix, scoped to your conda env:")
            print("         conda install -c conda-forge openjdk=11")
    return True


def java_major() -> int | None:
    """8 for '1.8.0_392', 26 for '26.0.2'. None if undetectable."""
    try:
        out = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=30
        )
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r'version "(\d+)(?:\.(\d+))?', (out.stderr or "") + (out.stdout or ""))
    if not m:
        return None
    first, second = int(m.group(1)), m.group(2)
    return int(second) if first == 1 and second else first


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class _SubprocessShim:
    """Stands in for the `subprocess` module inside pycocoevalcap.spice only.

    Strips `-cache <dir>` from the SPICE java command. SPICE uses LMDB for that
    cache via a bundled native library that ships x86_64 only -- on Apple
    silicon it fails with UnsatisfiedLinkError / 'incompatible architecture'.
    Without -cache, SPICE parses captions fresh (slower, same scores).
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def check_call(self, cmd, *args, **kwargs):
        if isinstance(cmd, list) and "-cache" in cmd:
            i = cmd.index("-cache")
            cmd = cmd[:i] + cmd[i + 2:]
        return self._real.check_call(cmd, *args, **kwargs)


@contextlib.contextmanager
def _spice_env(drop_cache: bool | None = None):
    """Scoped compatibility shims for SPICE. Two independent problems:

    1. Java 16+ blocks the reflection SPICE's serializer needs -> --add-opens.
       Set only around the SPICE call; leaking it makes every later java call
       (PTB tokenizer, METEOR) print a notice and breaks version parsing.
       Best-effort. The reliable fix is `conda install -c conda-forge openjdk=11`.

    2. SPICE's LMDB cache uses an x86_64-only native lib -> drop `-cache` on
       Apple silicon.
    """
    import subprocess as _sp

    if drop_cache is None:
        drop_cache = _is_apple_silicon() or os.environ.get("SPICE_NO_CACHE") == "1"

    major = java_major()
    prev = os.environ.get("JDK_JAVA_OPTIONS")
    if major is not None and major >= 16 and "--add-opens" not in (prev or ""):
        os.environ["JDK_JAVA_OPTIONS"] = ((prev or "") + " " + _SPICE_ADD_OPENS).strip()

    patched_mod = None
    if drop_cache:
        from pycocoevalcap.spice import spice as patched_mod

        patched_mod.subprocess = _SubprocessShim(_sp)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("JDK_JAVA_OPTIONS", None)
        else:
            os.environ["JDK_JAVA_OPTIONS"] = prev
        if patched_mod is not None:
            patched_mod.subprocess = _sp


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def evaluate_captions(
    predictions: dict[str, str],
    references: dict[str, list[str]],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    tokenize: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """Score `predictions` against `references`. Missing scorers are skipped
    with a warning rather than killing the whole run."""

    pred_ids, ref_ids = set(predictions), set(references)
    if pred_ids != ref_ids:
        missing = ref_ids - pred_ids
        extra = pred_ids - ref_ids
        raise ValueError(
            f"key mismatch: {len(missing)} ids missing from predictions "
            f"(e.g. {list(missing)[:3]}), {len(extra)} unexpected ids "
            f"(e.g. {list(extra)[:3]})"
        )
    if not predictions:
        raise ValueError("nothing to evaluate: predictions is empty")

    # pycocoevalcap wants {id: [{'caption': str}, ...]}
    gts = {k: [{"caption": c} for c in v] for k, v in references.items()}
    res = {k: [{"caption": v}] for k, v in predictions.items()}

    if tokenize:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
    else:
        gts = {k: [d["caption"] for d in v] for k, v in gts.items()}
        res = {k: [d["caption"] for d in v] for k, v in res.items()}

    # Built lazily inside the loop: constructing Spice() triggers a ~1GB
    # CoreNLP download, which must not take the other metrics down with it.
    def _bleu():
        from pycocoevalcap.bleu.bleu import Bleu
        return Bleu(4)

    def _meteor():
        from pycocoevalcap.meteor.meteor import Meteor
        return Meteor()

    def _rouge():
        from pycocoevalcap.rouge.rouge import Rouge
        return Rouge()

    def _cider():
        from pycocoevalcap.cider.cider import Cider
        return Cider()

    def _spice():
        from pycocoevalcap.spice.spice import Spice
        return Spice()

    registry = {
        "bleu": (_bleu, ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        "meteor": (_meteor, "METEOR"),
        "rouge": (_rouge, "ROUGE_L"),
        "cider": (_cider, "CIDEr"),
        "spice": (_spice, "SPICE"),
    }
    scorers = [(k, *registry[k]) for k in metrics if k in registry]

    out: dict[str, float] = {}
    for key, factory, names in scorers:
        label = names if isinstance(names, str) else "BLEU"
        try:
            if key == "spice":
                with _spice_env():
                    score, _ = factory().compute_score(gts, res)
            else:
                score, _ = factory().compute_score(gts, res)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{label} failed and was skipped: {e}", stacklevel=2)
            continue
        if isinstance(names, list):
            for n, s_ in zip(names, score):
                out[n] = float(s_)
        else:
            out[names] = float(score)
        if verbose:
            print(f"[ OK ] {label}")

    return out


def format_scores(scores: dict[str, float]) -> str:
    return "\n".join(f"  {k:<10} {v:.4f}" for k, v in scores.items())


if __name__ == "__main__":
    ok = check_java()
    sys.exit(0 if ok else 1)
