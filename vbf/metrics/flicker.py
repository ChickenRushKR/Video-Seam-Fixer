"""Flicker metric: frame-to-frame colour change.

Measures the per-pixel colour L2 between adjacent frames. A clean boundary has a
seam colour-jump comparable to the motion-driven change within A and within B;
flicker shows up as a spike at the seam.
"""

from __future__ import annotations

import torch

from vbf.metrics import _seam
from vbf.metrics.base import Metric, register


def _rgb_to_lab_like(x: torch.Tensor) -> torch.Tensor:
    """Cheap perceptual-ish transform: luma + two opponent-colour channels.

    Not a colour-accurate CIELAB (which needs gamma + XYZ), but differentiable
    and emphasises luminance the way Lab does. ``x`` is ``[T,3,H,W]`` in [0,1].
    """
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    op1 = r - g
    op2 = 0.5 * (r + g) - b
    return torch.stack([luma, op1, op2], dim=1)


@register("flicker")
class FlickerMetric(Metric):
    def _prep(self, seq: torch.Tensor) -> torch.Tensor:
        space = getattr(self.cfg, "color_space", "rgb")
        return _rgb_to_lab_like(seq) if space == "lab" else seq

    def _pair_diffs(self, seq: torch.Tensor) -> torch.Tensor:
        x = self._prep(seq)
        diff = (x[1:] - x[:-1]) ** 2            # [T-1, 3, H, W]
        return diff.flatten(1).mean(dim=1)       # [T-1]

    def loss(self, seq: torch.Tensor, boundary_index: int) -> torch.Tensor:
        return _seam.seam_loss(self._pair_diffs(seq), boundary_index)

    def observe(self, seq: torch.Tensor, boundary_index: int) -> dict[str, float]:
        seam, baseline, mean = _seam.seam_stats(self._pair_diffs(seq), boundary_index)
        return {
            "flicker/seam_diff": seam,
            "flicker/baseline_diff": baseline,
            "flicker/mean_diff": mean,
        }
