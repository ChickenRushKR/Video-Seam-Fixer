"""video_io round-trip: save then reload should preserve shape and rough values."""

import torch

from vbf.io.video_io import load_video, save_video, to_float


def test_roundtrip(tmp_path):
    t, h, w = 12, 64, 48
    # smooth structured content (real video compresses well; pure noise does not)
    ys = torch.linspace(0, 1, h).view(1, 1, h, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, 1, w)
    tt = torch.linspace(0, 1, t).view(t, 1, 1, 1)
    frames = (0.5 + 0.5 * torch.sin(6.28 * (xs + ys + tt))).expand(t, 3, h, w).contiguous()
    path = tmp_path / "rt.mp4"
    save_video(path, frames, fps=24.0)

    vid = load_video(path)
    assert vid.fps == 24.0 or abs(vid.fps - 24.0) < 1e-3
    assert vid.frames.dtype == torch.uint8
    assert vid.frames.shape[1:] == (3, h, w)
    assert vid.num_frames == t
    # h264 yuv420p is lossy; just check it's not wildly off on average
    n = min(vid.num_frames, t)
    assert (to_float(vid.frames[:n]) - frames[:n]).abs().mean() < 0.1
