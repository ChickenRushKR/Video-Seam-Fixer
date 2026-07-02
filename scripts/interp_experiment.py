"""v2 experiment: does flow-based frame interpolation smooth the seam motion step?

Runs on the v1 outputs (experiments/FINAL/video{2,3}_normalized_tight.mp4) so the seam
frames are already aligned. Three parts:

1. SELF-TEST  — interpolate frame[k] & frame[k+2] at t=0.5, compare to the real frame[k+1]
   (PSNR). Validates the interpolator + sign convention, and shows flow beats a plain blend.
2. PER-SEAM   — synth the midpoint between the two seam frames; report the ghosting proxy
   (fwd/bwd flow inconsistency) and the local temporal 2nd-difference with vs without the
   midpoint; save prev|mid|next|occlusion PNG.
3. COMPARE    — one slowed video per clip: local seam windows, LEFT = hold (freeze duplicate),
   RIGHT = interpolated midpoint (same length, synced) so the freeze-vs-smooth is directly visible.

    python scripts/interp_experiment.py
"""

import math
import os

import numpy as np
import torch
from PIL import Image

from vbf.interp import get_interpolator
from vbf.io.video_io import load_video, save_video_stream, to_float, to_uint8

OUT = "analysis/interp"
os.makedirs(OUT, exist_ok=True)
VIDS = {"2": "experiments/FINAL/video2_normalized_tight.mp4",
        "3": "experiments/FINAL/video3_normalized_tight.mp4"}
SEAMS = [241, 529, 673]           # v1-tight seam positions (frame index of next[start])
R = 5                             # local window radius for the comparison / temporal metric


def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return 99.0 if mse < 1e-9 else -10 * math.log10(mse)


def diff_series(frames):          # [T,3,H,W] -> [T-1] adjacent |diff|
    return (frames[1:] - frames[:-1]).abs().flatten(1).mean(1)


def second_diff_rms(series):      # smoothness: RMS of 2nd difference
    if series.shape[0] < 3:
        return 0.0
    return float(((series[2:] - 2 * series[1:-1] + series[:-2]) ** 2).mean().sqrt())


def to_img(t):
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def main():
    interp = get_interpolator("flow")

    # ---- 1. SELF-TEST (validate interpolator) ----
    print("=== self-test: interp(k, k+2) vs real k+1 ===")
    fr = to_float(load_video(VIDS["3"]).frames)
    ks = [120, 300, 700, 850]
    flow_ps, blend_ps = [], []
    for k in ks:
        mid = interp.interpolate(fr[k], fr[k + 2], 0.5)
        blend = (fr[k] + fr[k + 2]) / 2
        flow_ps.append(psnr(mid, fr[k + 1]))
        blend_ps.append(psnr(blend, fr[k + 1]))
    print(f"  flow-interp PSNR  = {np.mean(flow_ps):.2f} dB  (per-k {[round(x,1) for x in flow_ps]})")
    print(f"  plain-blend PSNR  = {np.mean(blend_ps):.2f} dB  (interp should be higher)")

    # ---- 2 + 3. PER-SEAM score + comparison video ----
    for v, path in VIDS.items():
        frames = to_float(load_video(path).frames)
        fps = load_video(path).fps
        print(f"\n=== video{v} seams ===")
        hold_win, interp_win = [], []     # frames for the LEFT (hold) / RIGHT (interp) comparison
        for s in SEAMS:
            prev, nxt = frames[s - 1], frames[s]
            mid = interp.interpolate(prev, nxt, 0.5)
            occ = interp.occlusion_score(prev, nxt)

            # temporal smoothness of the local window, hold(dup) vs interp(mid)
            base = frames[s - R:s]                       # ..prev
            tail = frames[s:s + R]                        # next..
            hold = torch.cat([base, prev.unsqueeze(0), tail], 0)      # freeze duplicate at seam
            withm = torch.cat([base, mid.unsqueeze(0), tail], 0)      # interpolated midpoint
            sd_hold = second_diff_rms(diff_series(hold))
            sd_mid = second_diff_rms(diff_series(withm))
            step_pn = float((nxt - prev).abs().mean())
            step_pm = float((mid - prev).abs().mean())
            step_mn = float((nxt - mid).abs().mean())
            print(f"  seam@{s}: step prev->next={step_pn:.4f} -> [prev->mid={step_pm:.4f}, mid->next={step_mn:.4f}] "
                  f"| 2nd-diff hold={sd_hold:.4f} interp={sd_mid:.4f} | occ(ghost proxy)={occ:.4f}")

            # PNG: prev | mid | next | occlusion heat
            a = prev.unsqueeze(0).to(interp.device).float(); b = nxt.unsqueeze(0).to(interp.device).float()
            occmap = interp._consistency(interp._flow(a, b), interp._flow(b, a))[0].cpu()
            oh = (occmap / (occmap.max() + 1e-6)).repeat(3, 1, 1)
            g = np.full((prev.shape[1], 12, 3), 40, np.uint8)
            row = np.concatenate([to_img(prev), g, to_img(mid), g, to_img(nxt), g, to_img(oh)], 1)
            Image.fromarray(row).resize((row.shape[1] // 4, prev.shape[1] // 4)).save(f"{OUT}/v{v}_seam{s}.png")

            hold_win.append(hold); interp_win.append(withm)

        # comparison video: LEFT hold(freeze) | RIGHT interp, all seam windows, 4x slow
        L = torch.cat(hold_win, 0); Rr = torch.cat(interp_win, 0)
        comp = torch.cat([L, Rr], dim=3)                 # side-by-side (concat on width)
        def gen():
            for i in range(comp.shape[0]):
                for _ in range(4):                        # 4x slow (repeat frames)
                    yield to_uint8(comp[i])
        outp = f"{OUT}/v{v}_hold_vs_interp_slow.mp4"
        n = save_video_stream(outp, gen(), fps=fps)
        print(f"  -> {outp} ({n}f)  [LEFT=hold/freeze, RIGHT=interpolated]")


if __name__ == "__main__":
    main()
