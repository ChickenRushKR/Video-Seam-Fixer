"""Flow-based frame interpolation — no extra dependencies.

Estimates bidirectional optical flow (RAFT, reused from ``vbf.metrics.flow``) between two
frames and backward-warps each toward the intermediate time ``t``, then blends with an
occlusion-aware weight derived from forward/backward flow consistency. Where the two flows
disagree (dis-occlusion / unreliable motion) the blend falls back toward a straight
cross-fade, which avoids the torn "ghost" a naive warp would produce.

Sign convention (validated empirically by the k,k+2 -> k+1 self-test in
``scripts/interp_experiment.py``): with ``F01 = flow(f0->f1)`` and ``F10 = flow(f1->f0)``,
the intermediate is
    I_t ≈ w0 · warp(f0,  t · F01) + w1 · warp(f1, (1-t) · F10)
where ``warp(img, d)[p] = img[p + d[p]]`` (backward sampling).
"""

from __future__ import annotations

import torch

from vbf.metrics.flow import _backward_warp


class FlowInterpolator:
    backend = "flow"

    def __init__(self, device=None, flow_resolution=(960, 544), occlusion=True):
        import torch as _t

        self.device = device or ("cuda" if _t.cuda.is_available() else "cpu")
        self.occlusion = occlusion
        self._fm = None
        self._flow_resolution = flow_resolution

    def _flow_metric(self):
        if self._fm is None:
            from vbf.config import FlowMetricConfig
            from vbf.metrics.flow import FlowMetric

            cfg = FlowMetricConfig()
            cfg.flow_resolution = self._flow_resolution
            self._fm = FlowMetric(cfg, self.device)
        return self._fm

    @torch.no_grad()
    def _flow(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._flow_metric()._estimate_flow(a, b)  # [1,2,H,W], detached

    @torch.no_grad()
    def _consistency(self, Fab: torch.Tensor, Fba: torch.Tensor) -> torch.Tensor:
        """Per-pixel forward/backward flow inconsistency on a's grid: |Fab + Fba(warped by Fab)|.
        ~0 where motion is reliable, large at occlusions. Returns [1,1,H,W]."""
        Fba_at_a = _backward_warp(Fba, Fab)         # sample Fba at where a maps to
        err = (Fab + Fba_at_a).pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt()
        return err

    @torch.no_grad()
    def interpolate(self, f0: torch.Tensor, f1: torch.Tensor, t: float) -> torch.Tensor:
        """Interpolate the frame at time ``t`` in (0,1) between ``f0`` and ``f1`` ([3,H,W])."""
        a = f0.unsqueeze(0).to(self.device).float()
        b = f1.unsqueeze(0).to(self.device).float()
        F01 = self._flow(a, b)
        F10 = self._flow(b, a)
        # backward sampling: I_t[q] = f0[q - t*F01[q]] and f1[q - (1-t)*F10[q]] (negative disp)
        w0 = _backward_warp(a, -t * F01)             # f0 pulled toward t
        w1 = _backward_warp(b, -(1.0 - t) * F10)     # f1 pulled toward t

        if not self.occlusion:
            out = (1.0 - t) * w0 + t * w1
            return out[0].clamp(0, 1).cpu()

        # occlusion-aware: trust the endpoint whose flow is locally consistent
        e0 = self._consistency(F01, F10)             # [1,1,H,W] on f0 grid
        e1 = self._consistency(F10, F01)
        s = 6.0                                      # sharpness of the reliability weighting
        r0 = (1.0 - t) * torch.exp(-s * e0)
        r1 = t * torch.exp(-s * e1)
        cf = 1e-3                                    # tiny cross-fade floor so weights never vanish
        w = (r0 + cf) + (r1 + cf)
        out = ((r0 + cf) * w0 + (r1 + cf) * w1) / w
        return out[0].clamp(0, 1).cpu()

    @torch.no_grad()
    def occlusion_score(self, f0: torch.Tensor, f1: torch.Tensor) -> float:
        """Mean fwd/bwd flow inconsistency (ghosting-risk proxy) between two frames."""
        a = f0.unsqueeze(0).to(self.device).float()
        b = f1.unsqueeze(0).to(self.device).float()
        return float(self._consistency(self._flow(a, b), self._flow(b, a)).mean())
