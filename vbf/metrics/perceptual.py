"""Perceptual / structural boundary metrics — pure-torch, no extra dependencies.

These are *evaluation* metrics (and the objective for the conservative
strength-search): unlike the raw mean-|diff| seam gap, they separate the two
things that actually matter at a seam —

* **structure** (SSIM) — invariant to a uniform brightness/colour offset, so a
  small global exposure difference between two clips does not read as "broken".
* **perceptual colour** (Lab ΔE76) — measures the colour shift a viewer sees,
  so a correction that *introduces* a tint (e.g. video3's red cast) is penalised
  even when it lowers the raw pixel L1.

Plus two temporal quantities that judge a seam *relative to the clip's own
motion* rather than against an ideal of zero:

* **motion baseline** — the clip's normal adjacent-frame change; a seam smaller
  than this needs no correction.
* **temporal instability** — 2nd-difference spike of the adjacent-frame-diff
  series across the seam (a step or flicker introduced by over-correction).

All inputs are ``[N,3,H,W]`` or ``[3,H,W]`` float tensors in [0,1].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# SSIM (gaussian-windowed, structural — brightness/colour-offset tolerant)
# --------------------------------------------------------------------------- #
def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]           # [size, size]
    return w.expand(3, 1, size, size)     # depthwise over 3 channels


def ssim(a: torch.Tensor, b: torch.Tensor, size: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM between two frames ``[3,H,W]`` (or ``[1,3,H,W]``) in [0,1]."""
    if a.dim() == 3:
        a = a.unsqueeze(0)
    if b.dim() == 3:
        b = b.unsqueeze(0)
    a = a.float()
    b = b.float()
    win = _gaussian_window(size, sigma, a.device, a.dtype)
    pad = size // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=3)
    mu_b = F.conv2d(b, win, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = F.conv2d(a * a, win, padding=pad, groups=3) - mu_a2
    var_b = F.conv2d(b * b, win, padding=pad, groups=3) - mu_b2
    cov = F.conv2d(a * b, win, padding=pad, groups=3) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_ab + c1) * (2 * cov + c2)) / ((mu_a2 + mu_b2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


# --------------------------------------------------------------------------- #
# CIE Lab + ΔE76 (perceptual colour distance)
# --------------------------------------------------------------------------- #
def rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """sRGB ``[...,3,H,W]`` in [0,1] -> CIE Lab (D65). Returns same spatial shape, 3 chans."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img.float().clamp(0, 1)
    # sRGB -> linear
    lin = torch.where(img > 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    # linear RGB -> XYZ (D65)
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    # normalise by D65 white
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883

    def f(t):
        return torch.where(t > 0.008856, t.clamp_min(1e-8) ** (1 / 3), 7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return torch.stack([L, a, bb], dim=1)   # [N,3,H,W]


def delta_e76(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean CIE76 colour difference (ΔE) between two frames in [0,1]."""
    la, lb = rgb_to_lab(a), rgb_to_lab(b)
    de = torch.sqrt(((la - lb) ** 2).sum(dim=1) + 1e-12)   # per-pixel ΔE
    return float(de.mean())


# --------------------------------------------------------------------------- #
# Temporal quantities (seam judged relative to the clip's own motion)
# --------------------------------------------------------------------------- #
def _pair_diffs(frames: torch.Tensor) -> torch.Tensor:
    """Per-adjacent-pair mean |diff| for ``[N,3,H,W]`` -> ``[N-1]``."""
    return (frames[1:] - frames[:-1]).abs().flatten(1).mean(dim=1)


def motion_baseline(clip: torch.Tensor, last_n: int = 20) -> float:
    """Normal adjacent-frame change over the last ``last_n`` frames of a clip."""
    tail = clip[-(last_n + 1):] if clip.shape[0] > last_n + 1 else clip
    d = _pair_diffs(tail)
    return float(d.mean()) if d.numel() else 0.0


def seam_l1(prev_last: torch.Tensor, next_first: torch.Tensor) -> float:
    """Mean |diff| across the seam (single frame pair)."""
    a = prev_last.unsqueeze(0) if prev_last.dim() == 3 else prev_last
    b = next_first.unsqueeze(0) if next_first.dim() == 3 else next_first
    return float((b - a).abs().mean())


def temporal_instability(seq: torch.Tensor, seam_index: int, radius: int = 4) -> float:
    """2nd-difference energy of the adjacent-frame-diff series around the seam.

    ``seq`` is the stitched sequence [..A_tail.., ..B_head..]; ``seam_index`` is
    the position of the first B frame in it. A clean transition has a smooth
    diff series (≈0); a step or oscillation introduced by over-correction spikes.
    """
    lo = max(0, seam_index - radius - 1)
    hi = min(seq.shape[0], seam_index + radius + 1)
    d = _pair_diffs(seq[lo:hi])
    if d.shape[0] < 3:
        return 0.0
    return float(((d[2:] - 2 * d[1:-1] + d[:-2]) ** 2).mean())


def seam_panel(
    prev_clip: torch.Tensor,
    next_raw: torch.Tensor,
    next_corr: torch.Tensor | None = None,
    b_orig: torch.Tensor | None = None,
    b_corr: torch.Tensor | None = None,
) -> dict:
    """Full metric panel for one boundary.

    ``prev_clip`` full previous clip (for motion baseline + tail); ``next_raw``
    the raw next clip; ``next_corr`` the corrected next clip (optional).
    ``b_orig``/``b_corr`` optional full next-clip original/corrected tensors for
    correction-magnitude + downstream-integrity (tail untouched) checks.
    """
    prev_last = prev_clip[-1]
    base = motion_baseline(prev_clip)
    raw_l1 = seam_l1(prev_last, next_raw[0])
    out = {
        "motion_baseline": base,
        "seam_l1_raw": raw_l1,
        "seam_ratio_raw": (raw_l1 / base) if base else float("nan"),
        "ssim_raw": ssim(prev_last, next_raw[0]),
        "deltaE_raw": delta_e76(prev_last, next_raw[0]),
    }
    if next_corr is not None:
        cor_l1 = seam_l1(prev_last, next_corr[0])
        out.update(
            seam_l1_corr=cor_l1,
            seam_ratio_corr=(cor_l1 / base) if base else float("nan"),
            ssim_corr=ssim(prev_last, next_corr[0]),
            deltaE_corr=delta_e76(prev_last, next_corr[0]),
        )
    if b_orig is not None and b_corr is not None:
        n = min(b_orig.shape[0], b_corr.shape[0])
        out["correction_magnitude"] = float((b_corr[:n] - b_orig[:n]).abs().mean())
        out["downstream_integrity"] = float((b_corr[n - 1] - b_orig[n - 1]).abs().mean())
    return out
