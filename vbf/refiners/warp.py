"""WarpRefiner / WarpColorRefiner — flow-TARGET geometric warp (+ optional color).

Warps the *original sharp* B frames toward a FIXED target estimated once from
optical flow, so it is stable (no RAFT feedback loop), ghost-free (resamples
sharp frames), and (for low-DOF models) rigid enough to avoid background wobble.

The geometric warp is a polynomial coordinate map in normalized coords, selected
by ``warp.model``:

* ``translation`` — global shift only (1 DOF/axis)
* ``affine``      — degree-1: shift + rotation + scale + shear (3 DOF/axis), fit
                    by *iterative* registration (captures scale/rotation)
* ``polynomial``  — degree-2: adds spatially-varying scale/rotation (6 DOF/axis),
                    a SMOOTH non-rigid warp that catches local pose drift a single
                    global affine cannot, without the per-pixel "jelly" of a dense
                    field. Single-shot fit (composition of polynomials is unstable).

Coefficients are in normalized coords → resolution independent: the same warp is
re-applied to the native-res B frame at output time.

Sign (grid_sample, align_corners=True): content at A_last(p) sits at B0(p+fs(p));
for continuity it should sit at q=p+vA(p) in B0', so B0'(q)=B0(q+(fs-vA)(q)) and
the target sample coord at q is ``q + excess(q)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vbf.data.window import BoundaryWindow
from vbf.refiners.base import (
    BoundaryRefiner,
    ColorCorrection,
    CompositeCorrection,
    Correction,
    register_refiner,
)

_DEGREE = {"translation": 1, "affine": 1, "polynomial": 2}
_NCOEF = {1: 3, 2: 6}


# ------------------------- geometric primitives -------------------------

def _normalized_base_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)  # [H,W,2] (x,y)


def _basis(coords: torch.Tensor, degree: int) -> torch.Tensor:
    """Polynomial basis of ``coords`` [...,2] -> [...,K]."""
    x, y = coords[..., 0], coords[..., 1]
    one = torch.ones_like(x)
    cols = [one, x, y]
    if degree >= 2:
        cols += [x * x, x * y, y * y]
    return torch.stack(cols, dim=-1)


def _identity_coeffs(degree: int, device, dtype) -> torch.Tensor:
    """Coeffs [K,2] such that ``basis @ coeffs == coords`` (identity warp)."""
    c = torch.zeros(_NCOEF[degree], 2, device=device, dtype=dtype)
    c[1, 0] = 1.0  # x term -> x output
    c[2, 1] = 1.0  # y term -> y output
    return c


def _warp_coeffs(frames: torch.Tensor, coeffs: torch.Tensor, degree: int) -> torch.Tensor:
    """Warp ``frames`` [N,3,H,W] by per-frame poly ``coeffs`` [N,K,2]."""
    n, _, h, w = frames.shape
    base = _normalized_base_grid(h, w, frames.device, frames.dtype)  # [H,W,2]
    b = _basis(base, degree)                                          # [H,W,K]
    grid = torch.einsum("hwk,nkc->nhwc", b, coeffs)                  # [N,H,W,2]
    return F.grid_sample(frames, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _fit_coeffs(ctrl_coords: torch.Tensor, target: torch.Tensor, degree: int) -> torch.Tensor:
    """Least-squares poly coeffs [K,2] mapping ctrl_coords -> target."""
    b = _basis(ctrl_coords, degree)  # [Nc,K]
    return torch.linalg.lstsq(b, target).solution  # [K,2]


def _robust_fit_coeffs(
    ctrl_coords: torch.Tensor, target: torch.Tensor, degree: int, iters: int = 5
) -> torch.Tensor:
    """IRLS robust poly fit: downweights outliers (the moving subject) so the fit
    captures only the GLOBAL background drift, not the dancer's local motion."""
    b = _basis(ctrl_coords, degree)            # [N,K]
    k = b.shape[1]
    eye = torch.eye(k, device=b.device, dtype=b.dtype)
    w = torch.ones(b.shape[0], device=b.device, dtype=b.dtype)
    c = torch.linalg.lstsq(b, target).solution
    for _ in range(iters):
        wb = b * w.unsqueeze(1)
        c = torch.linalg.solve(wb.T @ b + 1e-6 * eye, wb.T @ target)  # [K,2]
        resid = (b @ c - target).norm(dim=1)                          # [N]
        sigma = resid.median() + 1e-6
        w = 1.0 / (1.0 + (resid / (2.0 * sigma)) ** 2)                # soft outlier rejection
    return c


def _robust_translation(ctrl_coords: torch.Tensor, target: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """IRLS robust global translation (x, y): the background's common shift."""
    disp = target - ctrl_coords                # [N,2]
    w = torch.ones(disp.shape[0], device=disp.device, dtype=disp.dtype)
    m = disp.mean(0)
    for _ in range(iters):
        m = (w.unsqueeze(1) * disp).sum(0) / w.sum().clamp_min(1e-6)
        resid = (disp - m).norm(dim=1)
        sigma = resid.median() + 1e-6
        w = 1.0 / (1.0 + (resid / (2.0 * sigma)) ** 2)
    return m


def _coeffs_to_theta(c: torch.Tensor) -> torch.Tensor:
    """degree-1 coeffs [3,2] -> affine theta [2,3]."""
    return torch.stack(
        [torch.stack([c[1, 0], c[2, 0], c[0, 0]]), torch.stack([c[1, 1], c[2, 1], c[0, 1]])]
    )


def _theta_to_coeffs(t: torch.Tensor) -> torch.Tensor:
    """affine theta [2,3] -> degree-1 coeffs [3,2]."""
    c = torch.zeros(3, 2, device=t.device, dtype=t.dtype)
    c[:, 0] = torch.stack([t[0, 2], t[0, 0], t[0, 1]])
    c[:, 1] = torch.stack([t[1, 2], t[1, 0], t[1, 1]])
    return c


# ----------------------------- corrections -----------------------------

class WarpCorrection(Correction):
    """Per-frame polynomial warp coeffs [M,K,2] (resolution-independent)."""

    def __init__(self, coeffs: torch.Tensor, degree: int):
        self.coeffs = coeffs
        self.degree = degree
        self.optimize_count = coeffs.shape[0]

    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        c = self.coeffs[i : i + 1].to(frame_native.device)
        return _warp_coeffs(frame_native, c, self.degree).clamp(0, 1)


# ------------------------------- refiner --------------------------------

@register_refiner("warp")
class WarpRefiner(BoundaryRefiner):
    def _flow_estimator(self):
        est = self.loss_fn.metrics.get("flow")
        if est is None:
            from vbf.metrics.flow import FlowMetric

            est = FlowMetric(self.config.loss.metrics.flow, self.device)
        return est

    @torch.no_grad()
    def _robust_flow_coeffs(self, flow, degree, wcfg) -> torch.Tensor:
        """Robustly fit a GLOBAL transform (coeffs [K,2]) to a flow field's
        *background* component, downweighting the moving subject. The subject is
        excluded HERE (before any subtraction), so a fast dancer never inflates
        the estimate."""
        h, w = flow.shape[-2:]
        dev, dt = flow.device, flow.dtype
        disp = torch.empty_like(flow)
        disp[:, 0] = flow[:, 0] * (2.0 / max(w - 1, 1))   # px -> normalized
        disp[:, 1] = flow[:, 1] * (2.0 / max(h - 1, 1))
        disp = disp.clamp(-wcfg.max_disp, wcfg.max_disp)
        n = wcfg.fit_size
        ctrl = F.adaptive_avg_pool2d(disp, (n, n))
        p = _normalized_base_grid(n, n, dev, dt).reshape(-1, 2)
        target = p + ctrl[0].permute(1, 2, 0).reshape(-1, 2)
        identity = _identity_coeffs(degree, dev, dt)
        if wcfg.model == "translation":
            c = identity.clone()
            c[0] = _robust_translation(p, target)
            return c
        return _robust_fit_coeffs(p, target, degree)

    @torch.no_grad()
    def _fit_geometry(self, window: BoundaryWindow) -> tuple[torch.Tensor, int]:
        """Estimate the global geometric DRIFT (B's canvas vs A's), returns (coeffs, degree).

        Uses temporal-MEDIAN background plates: the median over A's last ~K frames
        and B's first ~K frames averages out the moving subject, leaving the static
        background. Registering those two plates yields the pure canvas drift
        (scale/position), cleanly separated from both the dancer AND instantaneous
        motion — so fast clips no longer over-warp.
        """
        wcfg = self.config.refiner.warp
        degree = _DEGREE[wcfg.model]
        est = self._flow_estimator()
        anchor = window.anchor
        dev, dt = anchor.device, anchor.dtype
        identity = _identity_coeffs(degree, dev, dt)

        k = min(anchor.shape[0], wcfg.plate_frames)
        kb = min(window.b_original.shape[0], wcfg.plate_frames)
        a_plate = anchor[-k:].median(dim=0).values.unsqueeze(0)            # [1,3,H,W] bg of A
        b_plate = window.b_original[:kb].median(dim=0).values.unsqueeze(0)  # bg of B

        flow = est._estimate_flow(a_plate, b_plate)                        # A canvas -> B canvas
        drift = self._robust_flow_coeffs(flow, degree, wcfg)               # global drift transform
        coeffs = identity + wcfg.strength * (drift - identity)
        return coeffs, degree

    def _decay(self, m, dev, dt):
        """Per-frame correction weight: uniform (constant drift) or linear decay (seam jump)."""
        if self.config.refiner.constant:
            return torch.ones(m, device=dev, dtype=dt)
        idx = torch.arange(m, device=dev, dtype=dt)
        return (1.0 - idx / max(m - 1, 1)).clamp_min(0.0)

    def _per_frame_targets(self, coeffs_fit, degree, m, dev, dt):
        identity = _identity_coeffs(degree, dev, dt)
        decay = self._decay(m, dev, dt).view(m, 1, 1)
        return identity + decay * (coeffs_fit - identity), identity  # [M,K,2], [K,2]

    def refine(self, window: BoundaryWindow) -> tuple[BoundaryWindow, Correction]:
        cfg = self.config.refiner
        window.to(self.device)
        m = window.optimize_count
        b_orig = window.b_original[:m].to(self.device)
        fixed_tail = window.b[m:].detach()
        dev, dt = b_orig.device, b_orig.dtype

        coeffs_fit, degree = self._fit_geometry(window)
        target, identity = self._per_frame_targets(coeffs_fit, degree, m, dev, dt)
        coeffs = identity.expand(m, *identity.shape).clone().requires_grad_(True)

        opt = torch.optim.Adam([coeffs], lr=max(cfg.lr, 0.05))
        for step in range(cfg.num_steps + 1):
            total = ((coeffs - target) ** 2).mean()
            if self.logger is not None and (step % cfg.log_every == 0 or step == cfg.num_steps):
                self._log(window, b_orig, fixed_tail, coeffs, degree, step, total, {})
            if step == cfg.num_steps:
                break
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

        with torch.no_grad():
            warped = _warp_coeffs(b_orig, coeffs, degree).clamp(0, 1)
            refined_b = torch.cat([warped, fixed_tail], dim=0) if fixed_tail.numel() else warped
        window.b = refined_b.detach()
        return window, WarpCorrection(coeffs.detach(), degree)

    def _log(self, window, b_orig, fixed_tail, coeffs, degree, step, total, extra_metrics):
        with torch.no_grad():
            warped = _warp_coeffs(b_orig, coeffs, degree)
            b = torch.cat([warped, fixed_tail], dim=0) if fixed_tail.numel() else warped
            seq = torch.cat([window.anchor, b], dim=0)
            metrics = self.loss_fn.observe(window, seq)
            metrics["warp/coeff_dev"] = float((coeffs - _identity_coeffs(degree, coeffs.device, coeffs.dtype)).abs().mean())
            metrics.update(extra_metrics)
        self.logger.log_step(
            step=step,
            losses={"loss/total": float(total.detach())},
            metrics=metrics,
            window=self._snapshot_window(window, warped.detach().clamp(0, 1), fixed_tail),
        )

    @staticmethod
    def _snapshot_window(window, warped, fixed_tail):
        b = torch.cat([warped, fixed_tail], dim=0) if fixed_tail.numel() else warped
        return BoundaryWindow(
            anchor=window.anchor, b=b.clamp(0, 1),
            b_original=window.b_original, optimize_count=window.optimize_count,
        )


@torch.no_grad()
def _fit_color(anchor: torch.Tensor, b0: torch.Tensor, ccfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel gain/bias mapping B0's color stats (mean, optional std) to A_last."""
    eps = 1e-5
    a = anchor[-1].flatten(1)
    b = b0[0].flatten(1)
    ma, mb = a.mean(1), b.mean(1)
    if ccfg.match_std:
        gain = (a.std(1) / (b.std(1) + eps)).clamp(1.0 / ccfg.gain_clamp, ccfg.gain_clamp)
    else:
        gain = torch.ones_like(ma)
    bias = ma - gain * mb
    gain = 1.0 + ccfg.strength * (gain - 1.0)
    bias = ccfg.strength * bias
    return gain, bias


@register_refiner("warpcolor")
class WarpColorRefiner(WarpRefiner):
    """Geometric warp (motion/pose) + per-channel color/exposure correction."""

    def refine(self, window: BoundaryWindow) -> tuple[BoundaryWindow, Correction]:
        cfg = self.config.refiner
        window.to(self.device)
        m = window.optimize_count
        b_orig = window.b_original[:m].to(self.device)
        fixed_tail = window.b[m:].detach()
        dev, dt = b_orig.device, b_orig.dtype

        coeffs_fit, degree = self._fit_geometry(window)
        gain_fit, bias_fit = _fit_color(window.anchor, window.b_original[:1], cfg.color)

        coeff_target, identity = self._per_frame_targets(coeffs_fit, degree, m, dev, dt)
        decay = self._decay(m, dev, dt)
        gain_target = 1.0 + decay.view(m, 1) * (gain_fit.view(1, 3) - 1.0)
        bias_target = decay.view(m, 1) * bias_fit.view(1, 3)

        coeffs = identity.expand(m, *identity.shape).clone().requires_grad_(True)
        gain = torch.ones(m, 3, device=dev, dtype=dt, requires_grad=True)
        bias = torch.zeros(m, 3, device=dev, dtype=dt, requires_grad=True)
        opt = torch.optim.Adam([coeffs, gain, bias], lr=max(cfg.lr, 0.05))

        def transform(b):
            return _warp_coeffs(b, coeffs, degree) * gain.view(m, 3, 1, 1) + bias.view(m, 3, 1, 1)

        for step in range(cfg.num_steps + 1):
            total = (
                ((coeffs - coeff_target) ** 2).mean()
                + ((gain - gain_target) ** 2).mean()
                + ((bias - bias_target) ** 2).mean()
            )
            if self.logger is not None and (step % cfg.log_every == 0 or step == cfg.num_steps):
                with torch.no_grad():
                    corrected = transform(b_orig).clamp(0, 1)
                    b = torch.cat([corrected, fixed_tail], dim=0) if fixed_tail.numel() else corrected
                    seq = torch.cat([window.anchor, b], dim=0)
                    metrics = self.loss_fn.observe(window, seq)
                    metrics["color/gain_mean"] = float(gain[0].mean())
                    metrics["color/bias_mean"] = float(bias[0].mean())
                    metrics["warp/coeff_dev"] = float((coeffs - identity).abs().mean())
                self.logger.log_step(
                    step=step,
                    losses={"loss/total": float(total.detach())},
                    metrics=metrics,
                    window=self._snapshot_window(window, transform(b_orig).detach().clamp(0, 1), fixed_tail),
                )
            if step == cfg.num_steps:
                break
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

        with torch.no_grad():
            corrected = transform(b_orig).clamp(0, 1)
            refined_b = torch.cat([corrected, fixed_tail], dim=0) if fixed_tail.numel() else corrected
        window.b = refined_b.detach()
        correction = CompositeCorrection(
            [WarpCorrection(coeffs.detach(), degree), ColorCorrection(gain.detach(), bias.detach())]
        )
        return window, correction
