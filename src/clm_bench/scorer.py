"""CLM scoring on top of precomputed encoder embeddings.

logits(state, candidates) = scale * cos(state_head(e_s), action_head(e_c)),
exactly as ``clm.engine.Engine.answer`` computes them, minus the HTTP layer and the
arena, so that we control what is cached and what is timed.
"""
from __future__ import annotations

import numpy as np

RAW_SCALE = 100.0      # clm.engine.RAW_SCALE: the no-head ablation


class CLMScorer:
    def __init__(self, checkpoint: str | None = None, device: str = "cpu", random_heads: bool = False):
        import torch
        self.torch, self.device = torch, device
        if random_heads:
            from clm.heads import make_head
            torch.manual_seed(0)
            self.state_head = make_head(1536, 3, layernorm=True).eval().to(device)
            self.action_head = make_head(1536, 3, layernorm=True).eval().to(device)
            self.scale, self.cfg, self.checkpoint = 20.0, {"random": True}, "random-init (plumbing test)"
        else:
            from clm.heads import HeadPair, download
            path = checkpoint or download()
            hp = HeadPair("clm-latest", path, device).ensure()
            self.state_head, self.action_head = hp.state_head, hp.action_head
            self.scale, self.cfg, self.checkpoint = hp.scale, hp.cfg, path
        self.n_params = sum(p.numel() for h in (self.state_head, self.action_head) for p in h.parameters())

    def _project(self, head, emb: np.ndarray):
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32)).to(self.device)
            return torch.nn.functional.normalize(head(x), dim=-1)

    def project_states(self, emb: np.ndarray):
        return self._project(self.state_head, emb)

    def project_actions(self, emb: np.ndarray):
        return self._project(self.action_head, emb)

    def logits_from_projected(self, zs, za) -> np.ndarray:
        """zs [1|n, d], za [k, d] device tensors -> [n, k] numpy logits."""
        with self.torch.inference_mode():
            return (self.scale * (zs @ za.T)).float().cpu().numpy()

    def logits(self, state_emb: np.ndarray, cand_emb: np.ndarray) -> np.ndarray:
        return self.logits_from_projected(self.project_states(state_emb), self.project_actions(cand_emb))

    @staticmethod
    def raw_logits(state_emb: np.ndarray, cand_emb: np.ndarray) -> np.ndarray:
        """Ablation: cosine in Qwen3-8B's own embedding space, no heads."""
        return RAW_SCALE * (state_emb @ cand_emb.T)

    def describe(self) -> dict:
        return {"checkpoint": self.checkpoint, "logit_scale": round(float(self.scale), 4),
                "head_params": self.n_params, "cfg": self.cfg}
