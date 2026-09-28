"""Disk cache for encoder embeddings, so re-running analyses never re-encodes.

One .npz per (backend, model, dtype); keys are sha1(text). Only used by the
calibration benchmark: the latency benchmark always encodes live.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str):
        self.path = path
        self.index: dict[str, int] = {}
        self.vecs: list[np.ndarray] = []
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            keys, mat = z["keys"], z["vecs"]
            self.index = {str(k): i for i, k in enumerate(keys)}
            self.vecs = list(mat)

    def get_many(self, embedder, texts: list[str], batch: int = 256, label: str = "") -> np.ndarray:
        uniq = list(dict.fromkeys(texts))
        missing = [t for t in uniq if _key(t) not in self.index]
        if missing:
            print(f"[embed] {label}: {len(missing)} new texts ({len(uniq) - len(missing)} cached)", flush=True)
            for i in range(0, len(missing), batch):
                chunk = missing[i:i + batch]
                out = embedder.embed(chunk)
                for t, v in zip(chunk, out):
                    self.index[_key(t)] = len(self.vecs)
                    self.vecs.append(v.astype(np.float32))
                done = min(i + batch, len(missing))
                print(f"[embed] {label}: {done}/{len(missing)}", flush=True)
                self.save()
        return np.stack([self.vecs[self.index[_key(t)]] for t in texts])

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        keys = np.array(sorted(self.index, key=self.index.get))
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, keys=keys, vecs=np.stack(self.vecs) if self.vecs else np.zeros((0, 4096), np.float32))
        os.replace(tmp, self.path)
