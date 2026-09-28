"""Encoder backends that produce the embeddings CLM's projection heads expect.

CLM-8B's heads were trained on Qwen3-8B **last-token pooled**, **L2-normalised**
final hidden states (vLLM ``--runner pooling`` defaults, max 2048 tokens).  Every
backend here returns exactly that: ``embed(texts) -> float32 [n, 4096]``, rows
L2-normalised.

* ``hf``   – Hugging Face transformers on MPS / CUDA / CPU.  The only option on a Mac
             (vLLM ships Linux wheels only).  Left padding so position -1 is always the
             real last token; ``AutoModel`` output already has the final RMSNorm applied,
             matching what vLLM pools.
* ``vllm`` – an OpenAI-compatible ``/v1/embeddings`` endpoint (the reference setup),
             via CLM's own ``clm.embedder.Embedder`` with its client-side LRU disabled so
             "uncached" timings really are uncached.
* ``fake`` – deterministic hash-seeded random vectors.  Plumbing tests only; its
             numbers mean nothing.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

HIDDEN = 4096
DEFAULT_HF_MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 2048


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Block until queued GPU work finishes, so wall-clock timings are honest."""
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


@dataclass
class EmbedStats:
    calls: int = 0
    texts: int = 0
    tokens: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


class BaseEmbedder:
    name = "base"
    device = "cpu"

    def __init__(self) -> None:
        self.stats = EmbedStats()

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, HIDDEN), np.float32)
        t0 = time.perf_counter()
        out, ntok = self._embed(texts)
        self.stats.seconds += time.perf_counter() - t0
        self.stats.calls += 1
        self.stats.texts += len(texts)
        self.stats.tokens += ntok
        return out

    def describe(self) -> dict:
        return {"backend": self.name, "device": self.device}


class HFEmbedder(BaseEmbedder):
    name = "hf"

    def __init__(self, model_id: str = DEFAULT_HF_MODEL, device: str | None = None,
                 dtype: str = "bfloat16", batch_size: int = 16, max_tokens: int = MAX_TOKENS):
        super().__init__()
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id, self.batch_size, self.max_tokens = model_id, batch_size, max_tokens
        self.device = pick_device(device)
        self.dtype_name = dtype
        torch_dtype = getattr(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"          # position -1 = real last token for every row
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        t0 = time.perf_counter()
        self.model = AutoModel.from_pretrained(model_id, dtype=torch_dtype).to(self.device).eval()
        self.load_seconds = time.perf_counter() - t0
        hidden = self.model.config.hidden_size
        if hidden != HIDDEN:
            raise ValueError(f"{model_id} has hidden size {hidden}; CLM-8B heads need {HIDDEN} (Qwen3-8B)")

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        torch = self.torch
        outs, ntok = [], 0
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i:i + self.batch_size]
                enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_tokens,
                               return_tensors="pt").to(self.device)
                ntok += int(enc["attention_mask"].sum())
                h = self.model(**enc).last_hidden_state[:, -1, :]
                outs.append(h.float().cpu().numpy())
        return l2(np.concatenate(outs).astype(np.float32)), ntok

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model_id, "device": self.device,
                "dtype": self.dtype_name, "batch_size": self.batch_size,
                "load_seconds": round(self.load_seconds, 1)}


class VLLMEmbedder(BaseEmbedder):
    name = "vllm"

    def __init__(self, url: str = "http://127.0.0.1:8090/v1/embeddings", model: str = "qwen3-8b",
                 batch_size: int = 64, max_tokens: int = MAX_TOKENS):
        super().__init__()
        from clm.embedder import Embedder
        # cache_size=0 would still insert then evict; use 1 and bypass the cache explicitly below.
        self.inner = Embedder(url=url, model=model, max_tokens=max_tokens, batch=batch_size, cache_size=1)
        self.url, self.model, self.batch_size = url, model, batch_size
        self.device = "remote"
        if not self.inner.healthy():
            raise RuntimeError(f"no vLLM embeddings server answering at {url}")

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        outs, ntok = [], 0
        for i in range(0, len(texts), self.batch_size):
            vecs, tk = self.inner._fetch(texts[i:i + self.batch_size])   # uncached on purpose
            outs.extend(vecs)
            ntok += tk
        return np.stack(outs).astype(np.float32), ntok

    def describe(self) -> dict:
        return {"backend": self.name, "url": self.url, "model": self.model, "batch_size": self.batch_size}


class FakeEmbedder(BaseEmbedder):
    """Deterministic pseudo-embeddings for plumbing tests. Results are meaningless."""
    name = "fake"

    def __init__(self, device: str | None = None, **_):
        super().__init__()
        self.device = device or "cpu"

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            rows.append(np.random.default_rng(seed).standard_normal(HIDDEN).astype(np.float32))
        return l2(np.stack(rows)), sum(len(t.split()) for t in texts)


def make_embedder(backend: str, **kw) -> BaseEmbedder:
    if backend == "hf":
        return HFEmbedder(model_id=kw.get("model") or DEFAULT_HF_MODEL, device=kw.get("device"),
                          dtype=kw.get("dtype", "bfloat16"), batch_size=kw.get("batch_size", 16))
    if backend == "vllm":
        return VLLMEmbedder(url=kw.get("vllm_url") or "http://127.0.0.1:8090/v1/embeddings",
                            batch_size=kw.get("batch_size", 64))
    if backend == "fake":
        return FakeEmbedder(device=kw.get("device"))
    raise ValueError(f"unknown backend {backend!r}")
