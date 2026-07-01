"""Shared helpers for turning a per-adjacent-pair quantity into a seam loss.

Given ``d`` of shape ``[T-1, ...]`` (some non-negative discontinuity measure for
each adjacent frame pair), the seam is the pair at index ``boundary_index - 1``
(between the last anchor frame and the first B frame). We want that pair's value
to look like the surrounding pairs — i.e. minimize the *excess* of the seam over
the local baseline, plus overall smoothness of the ``d`` sequence so the seam
does not stand out as a spike.
"""

from __future__ import annotations

import torch


def reduce_pairs(d: torch.Tensor) -> torch.Tensor:
    """Reduce a ``[T-1, ...]`` per-pair tensor to a scalar-per-pair ``[T-1]``."""
    if d.dim() == 1:
        return d
    return d.flatten(1).mean(dim=1)


def seam_loss(d: torch.Tensor, boundary_index: int) -> torch.Tensor:
    """Excess-over-baseline at the seam + smoothness of the per-pair sequence.

    The baseline is **detached**: otherwise the optimizer minimizes
    ``relu(seam - baseline)`` by *inflating the baseline* (degrading neighbor
    frames) instead of lowering the seam. Detaching removes that gradient path,
    so the only way to shrink the excess is to actually improve the seam pair.
    """
    d = reduce_pairs(d)
    seam = boundary_index - 1
    seam = max(0, min(seam, d.shape[0] - 1))

    seam_val = d[seam]
    mask = torch.ones_like(d, dtype=torch.bool)
    mask[seam] = False
    baseline = (d[mask].mean() if mask.any() else d.mean()).detach()

    excess = torch.relu(seam_val - baseline)
    # smoothness: penalize large second differences (a spike at the seam shows up here)
    if d.shape[0] >= 3:
        smooth = ((d[2:] - 2 * d[1:-1] + d[:-2]) ** 2).mean()
    else:
        smooth = d.new_zeros(())
    return excess + smooth


def seam_stats(d: torch.Tensor, boundary_index: int) -> tuple[float, float, float]:
    """(seam value, baseline mean, full mean) for logging."""
    d = reduce_pairs(d)
    seam = boundary_index - 1
    seam = max(0, min(seam, d.shape[0] - 1))
    mask = torch.ones_like(d, dtype=torch.bool)
    mask[seam] = False
    baseline = d[mask].mean() if mask.any() else d.mean()
    return float(d[seam]), float(baseline), float(d.mean())
