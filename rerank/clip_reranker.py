"""
CLIP scoring for caption reranking.

    from rerank.clip_reranker import CLIPReranker
    rr = CLIPReranker()
    scores = rr.score("images/foo.jpg", ["a dog on grass", "a cat on a mat"])
    best   = rr.rerank("images/foo.jpg", ["a dog on grass", "a cat on a mat"])

Scores are raw cosine similarities in [-1, 1] (usually 0.15-0.40 for real
image/caption pairs). NOT softmax probabilities: softmax over a candidate list
normalizes away absolute match quality, so an image whose candidates are all
bad would still produce probabilities summing to 1.

Install:  pip install git+https://github.com/openai/CLIP.git
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CLIPReranker:
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str | None = None,
        cache_path: str | None = None,
    ):
        import clip

        self.device = device or pick_device()
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.model_name = model_name

        # image embedding cache: {image_path: (D,) float32}
        self.cache_path = cache_path
        self._cache: dict[str, np.ndarray] = {}
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            self._cache = {k: data[k] for k in data.files}

    # ------------------------------------------------------------------
    def encode_images(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        """-> (N, D) float32, L2-normalized. Uses/fills the cache."""
        import torch

        todo = [p for p in paths if p not in self._cache]
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            batch = torch.stack(
                [self.preprocess(Image.open(p).convert("RGB")) for p in chunk]
            ).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for p, v in zip(chunk, feats.cpu().numpy()):
                self._cache[p] = v

        return np.stack([self._cache[p] for p in paths])

    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """-> (M, D) float32, L2-normalized. Batched; one-at-a-time is ~20x slower."""
        import clip
        import torch

        out = []
        for i in range(0, len(texts), batch_size):
            tokens = clip.tokenize(texts[i : i + batch_size], truncate=True).to(
                self.device
            )
            with torch.no_grad():
                feats = self.model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)

    # ------------------------------------------------------------------
    def score(self, image_path: str, captions: list[str]) -> np.ndarray:
        """Cosine similarity of one image against each caption. -> (len(captions),)"""
        img = self.encode_images([image_path])[0]
        txt = self.encode_texts(captions)
        return txt @ img

    def rerank(self, image_path: str, captions: list[str]) -> str:
        """The candidate CLIP likes best."""
        return captions[int(np.argmax(self.score(image_path, captions)))]

    # ------------------------------------------------------------------
    def save_cache(self) -> None:
        if self.cache_path and self._cache:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            np.savez(self.cache_path, **self._cache)
