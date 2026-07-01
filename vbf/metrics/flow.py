"""Optical-flow continuity metric (RAFT).

Backpropagating *through* RAFT (iterative, correlation volumes) over several
high-res pairs is extremely heavy in VRAM/time. Instead we use RAFT only as a
**flow estimator under ``no_grad``** (recomputed every call, so it always tracks
the current B), and make the *differentiable* signal a ``grid_sample`` based
**warp consistency**: warping frame ``t+1`` back by the (detached) flow should
reconstruct frame ``t``; gradients flow to the optimizable pixels through the
sampler, not through RAFT. This is orders of magnitude cheaper and scales.

Flow is computed only for pairs within ``radius`` of the seam, so cost is bounded
regardless of B length. The flow-magnitude continuity is logged as an
observation (RAFT being non-differentiable here, it is not used as a loss).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vbf.metrics import _seam
from vbf.metrics.base import Metric, register


def _pad_to_multiple(x: torch.Tensor, m: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)


def _backward_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample ``img`` ([N,3,H,W]) at identity+flow ([N,2,H,W]); returns [N,3,H,W]."""
    n, _, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    base = torch.stack([xs, ys], dim=0).unsqueeze(0)  # [1,2,H,W] (x,y)
    coords = base + flow                               # [N,2,H,W]
    x = 2 * coords[:, 0] / max(w - 1, 1) - 1
    y = 2 * coords[:, 1] / max(h - 1, 1) - 1
    grid = torch.stack([x, y], dim=-1)                 # [N,H,W,2]
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


@register("flow")
class FlowMetric(Metric):
    def __init__(self, cfg: object, device):
        super().__init__(cfg, device)
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        self.radius = int(getattr(cfg, "radius", 3))
        self.num_flow_updates = int(getattr(cfg, "num_flow_updates", 6))
        fr = getattr(cfg, "flow_resolution", None)
        self.flow_resolution = tuple(fr) if fr else None
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights, progress=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _pair_range(self, boundary_index: int, num_pairs: int) -> tuple[int, int]:
        seam = boundary_index - 1
        lo = max(0, seam - self.radius)
        hi = min(num_pairs, seam + self.radius + 1)
        return lo, hi

    @torch.no_grad()
    def _estimate_flow(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Estimate flow img1->img2 at working res. Detached (no RAFT backward)."""
        h, w = img1.shape[-2:]
        a_in, b_in = img1, img2
        if self.flow_resolution is not None:
            a_in = F.interpolate(a_in, size=self.flow_resolution, mode="bilinear", align_corners=False)
            b_in = F.interpolate(b_in, size=self.flow_resolution, mode="bilinear", align_corners=False)
        a, (eh, ew) = _pad_to_multiple(a_in * 2 - 1)
        b, _ = _pad_to_multiple(b_in * 2 - 1)
        flow = self.model(a, b, num_flow_updates=self.num_flow_updates)[-1]
        flow = flow[..., :eh, :ew]
        if (eh, ew) != (h, w):  # rescale flow vectors back to working res
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            flow[:, 0] *= w / ew
            flow[:, 1] *= h / eh
        return flow.detach()

    def _pair_signals(self, seq: torch.Tensor, boundary_index: int):
        num_pairs = seq.shape[0] - 1
        lo, hi = self._pair_range(boundary_index, num_pairs)
        img1 = seq[lo:hi]              # grad-carrying
        img2 = seq[lo + 1 : hi + 1]    # grad-carrying
        flow = self._estimate_flow(img1.detach(), img2.detach())  # detached

        # differentiable warp-consistency (gradients flow to pixels via grid_sample)
        warped = _backward_warp(img2, flow)
        warp_err = (warped - img1).pow(2).flatten(1).mean(dim=1)   # [P]

        # flow magnitude: observation only (no grad)
        mag = flow.pow(2).sum(1).clamp_min(1e-12).sqrt().flatten(1).mean(dim=1)  # [P]

        local_boundary = (boundary_index - 1 - lo) + 1
        return warp_err, mag, local_boundary

    def loss(self, seq: torch.Tensor, boundary_index: int) -> torch.Tensor:
        warp_err, _, lb = self._pair_signals(seq, boundary_index)
        return _seam.seam_loss(warp_err, lb)

    def observe(self, seq: torch.Tensor, boundary_index: int) -> dict[str, float]:
        warp_err, mag, lb = self._pair_signals(seq, boundary_index)
        ws, wb, wm = _seam.seam_stats(warp_err, lb)
        ms, mb, mm = _seam.seam_stats(mag, lb)
        return {
            "flow/seam_warp_err": ws,
            "flow/baseline_warp_err": wb,
            "flow/mean_warp_err": wm,
            "flow/seam_mag": ms,
            "flow/baseline_mag": mb,
            "flow/mean_mag": mm,
        }
