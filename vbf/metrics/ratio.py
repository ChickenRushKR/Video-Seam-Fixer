"""Ratio metric: frame-to-frame change in global colour statistics.

Captures exposure / colour-balance / contrast shifts that flicker (per-pixel)
misses: each frame is summarised by per-channel statistics (mean, std), and we
penalise abrupt changes in those statistics across the seam. Uses symmetric
log-ratio so a 2x jump up and down are penalised equally.
"""

from __future__ import annotations

import torch

from vbf.metrics import _seam
from vbf.metrics.base import Metric, register

_EPS = 1e-4


@register("ratio")
class RatioMetric(Metric):
    def _frame_stats(self, seq: torch.Tensor) -> torch.Tensor:
        """Per-frame statistic vector ``[T, C*nstats]``."""
        stats = list(getattr(self.cfg, "stats", ["mean", "std"]))
        flat = seq.flatten(2)  # [T, 3, H*W]
        parts = []
        if "mean" in stats:
            parts.append(flat.mean(dim=2))
        if "std" in stats:
            parts.append(flat.std(dim=2))
        return torch.cat(parts, dim=1)  # [T, 3*len(stats)]

    def _pair_logratio(self, seq: torch.Tensor) -> torch.Tensor:
        s = self._frame_stats(seq).clamp_min(_EPS)  # [T, D]
        logr = torch.log(s[1:]) - torch.log(s[:-1])  # [T-1, D]
        return logr.abs()

    def loss(self, seq: torch.Tensor, boundary_index: int) -> torch.Tensor:
        return _seam.seam_loss(self._pair_logratio(seq), boundary_index)

    def observe(self, seq: torch.Tensor, boundary_index: int) -> dict[str, float]:
        seam, baseline, mean = _seam.seam_stats(self._pair_logratio(seq), boundary_index)
        return {
            "ratio/seam_logratio": seam,
            "ratio/baseline_logratio": baseline,
            "ratio/mean_logratio": mean,
        }
