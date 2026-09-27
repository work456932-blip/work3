from __future__ import annotations

import hashlib
import numpy as np


class TextEncoder:
    def __init__(self, name: str, device: str = "cpu", dim: int = 64) -> None:
        self.name = name
        self.device = device
        self.dim = int(dim)

        self._model = None
        self._tokenizer = None
        self._init_model()

    def _init_model(self) -> None:
        if self.name != "biomedclip_text":
            return
        try:
            import open_clip

            model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            model, _ = open_clip.create_model_from_pretrained(model_id)
            tokenizer = open_clip.get_tokenizer(model_id)

            model = model.to(self.device)
            model.eval()

            self._model = model
            self._tokenizer = tokenizer
        except Exception:
            self._model = None
            self._tokenizer = None

    def _encode_one_hash(self, text: str) -> np.ndarray:
        text = text or ""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def _encode_biomedclip(self, texts):
        try:
            import torch

            tokens = self._tokenizer(texts)
            if isinstance(tokens, dict):
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
            else:
                tokens = tokens.to(self.device)

            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats = feats.detach().cpu().numpy().astype(np.float32)

            if feats.shape[1] < self.dim:
                pad = np.zeros((feats.shape[0], self.dim - feats.shape[1]), dtype=np.float32)
                feats = np.concatenate([feats, pad], axis=1)
            elif feats.shape[1] > self.dim:
                feats = feats[:, : self.dim]

            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            feats = feats / norms
            return feats.astype(np.float32)
        except Exception:
            return None

    def encode(self, texts):
        if self._model is not None and self._tokenizer is not None:
            feats = self._encode_biomedclip(texts)
            if feats is not None:
                return feats
        return np.stack([self._encode_one_hash(t) for t in texts], axis=0)
