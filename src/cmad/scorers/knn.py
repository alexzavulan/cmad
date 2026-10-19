from __future__ import annotations

import torch

from cmad.features import FeatureBatch

from .base import ScoreOutput

METRICS = ("cosine", "euclidean", "sqeuclidean")


def nn_distance(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k: int = 1,
    metric: str = "cosine",
    chunk: int = 65_536,
    query_chunk: int = 1_024,
) -> torch.Tensor:
    """Exact k-NN distance of queries [M x D] against bank [N x D], mean over the k neighbours.

    cosine assumes L2-normalised inputs (distance = 1 - similarity); euclidean works on any
    vectors and is always computed in fp32; sqeuclidean is the same without the square root
    (what FAISS's IndexFlatL2 returns, hence what the official PatchCore scores with).
    Chunked over both sides so peak memory is query_chunk x chunk, never M x N.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    out = []
    for q0 in range(0, queries.shape[0], query_chunk):
        q = queries[q0 : q0 + query_chunk]
        if metric == "cosine":
            best = torch.full((q.shape[0], k), -2.0, device=q.device, dtype=q.dtype)
            for s in range(0, bank.shape[0], chunk):
                sims = q @ bank[s : s + chunk].T
                best = torch.cat([best, sims], dim=1).topk(k, dim=1).values
            out.append((1.0 - best.float()).mean(dim=1))
        else:
            q = q.float()
            q2 = (q * q).sum(dim=1, keepdim=True)
            best = torch.full((q.shape[0], k), float("inf"), device=q.device)
            for s in range(0, bank.shape[0], chunk):
                b = bank[s : s + chunk].float()
                d2 = q2 + (b * b).sum(dim=1)[None, :] - 2.0 * (q @ b.T)
                best = torch.cat([best, d2], dim=1).topk(k, dim=1, largest=False).values
            d2 = best.clamp_min(0)
            out.append((d2 if metric == "sqeuclidean" else d2.sqrt()).mean(dim=1))
    return torch.cat(out)


def greedy_coreset(
    x: torch.Tensor, n_select: int, proj_dim: int = 128, seed: int = 0
) -> torch.Tensor:
    """PatchCore's k-centre greedy on a seeded random projection; returns indices into x."""
    n = x.shape[0]
    if n_select >= n:
        return torch.arange(n, device=x.device)
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(x.shape[1], proj_dim, generator=g).to(x.device) / proj_dim**0.5
    z = x.float() @ proj
    first = int(torch.randint(n, (1,), generator=g))
    selected = torch.empty(n_select, dtype=torch.long, device=x.device)
    selected[0] = first
    min_d = ((z - z[first]) ** 2).sum(dim=1)
    for i in range(1, n_select):
        idx = torch.argmax(min_d)
        selected[i] = idx
        min_d = torch.minimum(min_d, ((z - z[idx]) ** 2).sum(dim=1))
    return selected


class KNNScorer:
    def __init__(
        self,
        k: int = 1,
        coreset_fraction: float = 1.0,
        coreset_proj_dim: int = 128,
        image_agg: str = "max",
        metric: str = "cosine",
        device: str = "cpu",
        seed: int = 0,
        normalize: bool = False,
    ) -> None:
        if image_agg not in ("max", "top1pct"):
            raise ValueError(f"unknown image_agg {image_agg!r}")
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}")
        self.k, self.fraction, self.proj_dim = k, coreset_fraction, coreset_proj_dim
        self.image_agg, self.metric, self.device, self.seed = (
            image_agg,
            metric,
            device,
            seed,
        )
        self.normalize = (
            normalize  # L2-normalise patches on the way in (cosine on raw features)
        )
        self.bank: torch.Tensor | None = None

    def _dtype(self) -> torch.dtype:
        # fp16 similarity only for cosine on CUDA (spec §9.1); euclidean needs the precision
        if self.metric == "cosine" and self.device.startswith("cuda"):
            return torch.float16
        return torch.float32

    def _stack(self, fb: FeatureBatch) -> torch.Tensor:
        x = torch.cat([fb.patches[layer] for layer in fb.patches], dim=-1)  # N x P x D
        return torch.nn.functional.normalize(x.float(), dim=-1) if self.normalize else x

    def fit(
        self,
        train: FeatureBatch,
        roi: torch.Tensor | None = None,
        support: dict[str, FeatureBatch] | None = None,
    ) -> None:
        """Build the memory bank from the normal training patches (optionally ROI-filtered, then coreset-subsampled)."""
        x = self._stack(train)
        if roi is not None:
            x = x[roi.bool()]
        x = x.reshape(-1, x.shape[-1]).to(self.device).float()
        if self.fraction < 1.0:
            idx = greedy_coreset(
                x, int(len(x) * self.fraction), self.proj_dim, self.seed
            )
            x = x[idx]
        self.bank = x.to(self._dtype())

    def _patch_distance(
        self, feats: FeatureBatch, bank: torch.Tensor, k: int
    ) -> torch.Tensor:
        """B x P distance of every patch to its k nearest bank patches."""
        x = self._stack(feats)
        b, p, d = x.shape
        q = x.reshape(-1, d).to(self.device, self._dtype())
        return nn_distance(q, bank, k, self.metric).float().reshape(b, p).cpu()

    def _aggregate(self, dist: torch.Tensor) -> torch.Tensor:
        if self.image_agg == "max":
            return dist.max(dim=1).values
        return dist.topk(max(1, dist.shape[1] // 100), dim=1).values.mean(dim=1)

    def score(
        self, feats: FeatureBatch, roi: torch.Tensor | None = None
    ) -> ScoreOutput:
        """Score each patch by its distance to the nearest bank patches; the image score aggregates them (max or top 1 %)."""
        if self.bank is None:
            raise RuntimeError("KNNScorer.score called before fit")
        dist = self._patch_distance(feats, self.bank, self.k)
        b = dist.shape[0]
        if roi is not None:
            dist = dist.masked_fill(~roi.reshape(b, -1).bool(), 0.0)
        return ScoreOutput(self._aggregate(dist), dist.reshape(b, *feats.grid), {})

    def memory_bytes(self) -> int:
        if self.bank is None:
            return 0
        return self.bank.numel() * self.bank.element_size()
