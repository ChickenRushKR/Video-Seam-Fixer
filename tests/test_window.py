"""BoundaryWindow layout and assembly."""

import torch

from vbf.data.window import BoundaryWindow


def test_build_and_assemble():
    a = torch.zeros(10, 3, 8, 8)
    b = torch.ones(6, 3, 8, 8)
    win = BoundaryWindow.build(a, b, anchor_frames=3, optimize_b_frames=4)

    assert win.anchor.shape[0] == 3
    assert win.boundary_index == 3
    assert win.optimize_count == 4

    seq = win.full_sequence()
    assert seq.shape[0] == 3 + 6
    # anchors are A's tail (zeros), B side is ones
    assert torch.allclose(seq[: win.boundary_index], torch.zeros_like(seq[: win.boundary_index]))
    assert torch.allclose(seq[win.boundary_index :], torch.ones_like(seq[win.boundary_index :]))


def test_optimize_count_clamped_to_b_length():
    a = torch.zeros(4, 3, 8, 8)
    b = torch.ones(5, 3, 8, 8)
    win = BoundaryWindow.build(a, b, anchor_frames=2, optimize_b_frames=999)
    assert win.optimize_count == 5
