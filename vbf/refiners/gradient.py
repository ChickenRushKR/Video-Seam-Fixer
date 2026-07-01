"""GradientRefiner — free-pixel prototype backend.

The optimizable B frames are a learnable parameter (initialized from the
original B). A's tail frames are fixed anchors, and any B frames beyond the
optimize window are carried along fixed. Each step minimizes the composite
boundary loss with Adam/SGD and logs metrics + a boundary snapshot.

Note: optimizing raw pixels under temporal-smoothness losses tends to *blend*
content (ghosting). For sharp results prefer the ``warp`` backend.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vbf.data.window import BoundaryWindow
from vbf.refiners.base import BoundaryRefiner, Correction, register_refiner


class GradientCorrection(Correction):
    """Additive residual learned at working res, upsampled and added natively."""

    def __init__(self, delta_work: torch.Tensor):
        self.delta_work = delta_work  # [M,3,hW,wW]
        self.optimize_count = delta_work.shape[0]

    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        h, w = frame_native.shape[-2:]
        d = F.interpolate(
            self.delta_work[i : i + 1].to(frame_native.device),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        return (frame_native + d).clamp(0, 1)


@register_refiner("gradient")
class GradientRefiner(BoundaryRefiner):
    def refine(self, window: BoundaryWindow) -> tuple[BoundaryWindow, Correction]:
        cfg = self.config.refiner
        window.to(self.device)

        m = window.optimize_count
        param = window.b[:m].clone().requires_grad_(True)
        fixed_tail = window.b[m:].detach()

        if cfg.optimizer == "sgd":
            opt = torch.optim.SGD([param], lr=cfg.lr, momentum=0.9)
        else:
            opt = torch.optim.Adam([param], lr=cfg.lr)

        for step in range(cfg.num_steps + 1):
            b = torch.cat([param, fixed_tail], dim=0) if fixed_tail.numel() else param
            seq = torch.cat([window.anchor, b], dim=0)

            total, components = self.loss_fn.compute(window, seq, param)

            if self.logger is not None and (step % cfg.log_every == 0 or step == cfg.num_steps):
                with torch.no_grad():
                    metrics = self.loss_fn.observe(window, seq.detach())
                self.logger.log_step(
                    step=step,
                    losses=components,
                    metrics=metrics,
                    window=self._snapshot_window(window, param, fixed_tail),
                )

            if step == cfg.num_steps:
                break

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            with torch.no_grad():
                param.clamp_(0.0, 1.0)

        with torch.no_grad():
            delta_work = (param.detach() - window.b_original[:m]).contiguous()
            refined_b = (
                torch.cat([param, fixed_tail], dim=0) if fixed_tail.numel() else param
            ).clamp(0, 1)
        window.b = refined_b.detach()
        return window, GradientCorrection(delta_work)

    @staticmethod
    def _snapshot_window(
        window: BoundaryWindow, param: torch.Tensor, fixed_tail: torch.Tensor
    ) -> BoundaryWindow:
        with torch.no_grad():
            b = torch.cat([param, fixed_tail], dim=0) if fixed_tail.numel() else param
            snap = BoundaryWindow(
                anchor=window.anchor,
                b=b.detach().clamp(0, 1),
                b_original=window.b_original,
                optimize_count=window.optimize_count,
            )
        return snap
