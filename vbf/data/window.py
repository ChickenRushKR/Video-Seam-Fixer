"""BoundaryWindow: the unit of work passed to refiners and metrics.

A window is the concatenation ``[A_tail (anchor) | B]`` over which boundary
continuity is measured. The last ``K`` frames of A are *anchors* (held fixed);
the first ``M`` frames of B are *optimizable* (their pixels are the learnable
variables); any remaining B frames are carried along fixed.

Layout of ``full_sequence()`` (indices)::

    [ 0 .. K-1 ]      anchor (A tail, fixed)
    [ K .. K+M-1 ]    optimizable B frames
    [ K+M .. end ]    fixed B tail
                ^ boundary is between index K-1 and K
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BoundaryWindow:
    anchor: torch.Tensor      # [K, 3, H, W] last K frames of A (fixed)
    b: torch.Tensor           # [Tb, 3, H, W] current B (mutable across steps)
    b_original: torch.Tensor  # [Tb, 3, H, W] original B (for fidelity regularization)
    optimize_count: int       # M: first M frames of B are optimizable

    @classmethod
    def build(
        cls,
        a_frames: torch.Tensor,
        b_frames: torch.Tensor,
        anchor_frames: int,
        optimize_b_frames: int | None,
    ) -> "BoundaryWindow":
        k = max(1, min(anchor_frames, a_frames.shape[0]))
        anchor = a_frames[-k:].clone()
        tb = b_frames.shape[0]
        m = tb if optimize_b_frames is None else max(1, min(optimize_b_frames, tb))
        return cls(
            anchor=anchor,
            b=b_frames.clone(),
            b_original=b_frames.clone(),
            optimize_count=m,
        )

    @property
    def boundary_index(self) -> int:
        """Index in ``full_sequence`` of the first B frame (right side of the seam)."""
        return self.anchor.shape[0]

    def assemble(self, b: torch.Tensor | None = None) -> torch.Tensor:
        """Concatenate anchor + (given or current) B into ``[T, 3, H, W]``."""
        b = self.b if b is None else b
        return torch.cat([self.anchor, b], dim=0)

    def full_sequence(self) -> torch.Tensor:
        return self.assemble()

    def to(self, device: torch.device | str) -> "BoundaryWindow":
        self.anchor = self.anchor.to(device)
        self.b = self.b.to(device)
        self.b_original = self.b_original.to(device)
        return self

    def preview(self, radius: int = 3) -> torch.Tensor:
        """Frames straddling the seam: ``radius`` from A_tail and ``radius`` from B."""
        bi = self.boundary_index
        seq = self.full_sequence()
        lo = max(0, bi - radius)
        hi = min(seq.shape[0], bi + radius)
        return seq[lo:hi]
