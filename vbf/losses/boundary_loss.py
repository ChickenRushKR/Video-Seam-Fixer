"""Composite boundary loss: weighted metric losses + fidelity regularization.

``total = Σ w_m · metric_m.loss(seq) + w_fidelity · ‖B_opt − B_orig‖²``

The fidelity term keeps the optimized B frames close to the originals so the
optimizer smooths the seam without hallucinating the whole clip away.
"""

from __future__ import annotations

import torch

from vbf.config import Config
from vbf.data.window import BoundaryWindow
from vbf.metrics.base import Metric, build_metric


class BoundaryLoss:
    def __init__(self, config: Config, device: torch.device | str):
        self.config = config
        self.device = device
        self.weights = dict(config.loss.weights)
        metric_cfgs = config.loss.metrics
        self.metrics: dict[str, Metric] = {}
        for name in ("flicker", "ratio", "flow"):
            if self.weights.get(name, 0.0) > 0:
                self.metrics[name] = build_metric(name, getattr(metric_cfgs, name), device)

    def fidelity(self, window: BoundaryWindow, b_opt: torch.Tensor) -> torch.Tensor:
        ref = window.b_original[: window.optimize_count].to(b_opt.device)
        return (b_opt - ref).pow(2).mean()

    def metric_terms(
        self, window: BoundaryWindow, seq: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted sum of metric losses on ``seq`` (no fidelity term)."""
        components: dict[str, float] = {}
        total = seq.new_zeros(())
        for name, metric in self.metrics.items():
            term = metric.loss(seq, window.boundary_index)
            total = total + self.weights[name] * term
            components[f"loss/{name}"] = float(term.detach())
        return total, components

    def compute(
        self, window: BoundaryWindow, seq: torch.Tensor, b_opt: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total, components = self.metric_terms(window, seq)
        w_fid = self.weights.get("fidelity", 0.0)
        if w_fid > 0:
            fid = self.fidelity(window, b_opt)
            total = total + w_fid * fid
            components["loss/fidelity"] = float(fid.detach())
        components["loss/total"] = float(total.detach())
        return total, components

    def observe(self, window: BoundaryWindow, seq: torch.Tensor) -> dict[str, float]:
        out: dict[str, float] = {}
        for metric in self.metrics.values():
            out.update(metric.observe_for(window, seq))
        return out
