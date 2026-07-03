"""In-process seam-quality evaluation for an arbitrary clip chain.

Unlike ``eval_boundaries.py`` (which reads pre-rendered result mp4s for the fixed
sample chains), this measures the ESTIMATOR directly and fast: for a chain it runs
the current ``normalize_chain`` estimation, reconstructs each seam's corrected
``prev_last`` / ``next_first`` (post overlap-drop), and reports raw vs corrected
seam metrics — no full render needed. Used as the before/after yardstick when
hardening the algorithm.

Usage:
    python scripts/eval_chain.py samples/video2-A.mp4 ... [--mode tight]
    python scripts/eval_chain.py 2                     # sample-2 shortcut
    python scripts/eval_chain.py dance                 # Desktop/dance_extension
    python scripts/eval_chain.py josujin               # Downloads/josujin_trim/out
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from vbf.io.video_io import load_video, to_float, to_uint8
from vbf.metrics.perceptual import delta_e76, ssim
from vbf.normalize import chain as C


def _resolve(spec):
    if len(spec) == 1 and spec[0] in ("2", "3"):
        return [f"samples/video{spec[0]}-{c}.mp4" for c in "ABCD"]
    if len(spec) == 1 and spec[0] == "dance":
        d = r"C:\Users\HYEONJIN\Desktop\dance_extension"
        return sorted(glob.glob(os.path.join(d, "*.mp4")),
                      key=lambda p: int(os.path.basename(p).split("-")[0]))
    if len(spec) == 1 and spec[0] == "josujin":
        d = r"C:\Users\HYEONJIN\Downloads\josujin_trim\out"
        return sorted(glob.glob(os.path.join(d, "*.mp4")),
                      key=lambda p: int(os.path.basename(p)[0]))
    return spec


def _edge_frames(paths, ref_hw, dmax=40):
    """Return (heads, tails) as float [.,3,H,W] edge buffers per clip, at ref resolution."""
    heads, tails, lengths = [], [], []
    for p in paths:
        fr = load_video(p).frames
        if (fr.shape[2], fr.shape[3]) != ref_hw:
            fr = to_uint8(F.interpolate(to_float(fr), size=ref_hw, mode="bilinear", align_corners=False))
        lengths.append(fr.shape[0])
        heads.append(to_float(fr[:dmax]))
        tails.append(to_float(fr[-dmax:]))
    return heads, tails, lengths


def evaluate(paths, mode="tight", overlap="auto"):
    v0 = load_video(paths[0])
    ref_hw = (v0.frames.shape[2], v0.frames.shape[3])
    Hh, Ww = ref_hw
    heads, tails, lengths = _edge_frames(paths, ref_hw)
    N = len(paths)

    auto = str(overlap).lower() == "auto"
    ov = [0] * N
    for i in range(1, N):
        ov[i] = C.detect_overlap(tails[i - 1], heads[i]) if auto else int(overlap)
        ov[i] = min(ov[i], lengths[i] - 1)

    # pass 1: estimate + gate each seam (mirrors chain.normalize_chain, local-style anchor)
    est = []
    for i in range(1, N):
        prev_last = tails[i - 1][-1]
        k = ov[i]
        ai = min(max(k - 1, 0), heads[i].shape[0] - 1)
        hs = min(k, heads[i].shape[0] - 1)
        align_f = heads[i][ai]
        next_first = heads[i][hs]
        M = C.seam_affine(prev_last, align_f)[:2].astype(np.float32)
        g, b, sh = C.color_sharp_fit(tails[i - 1], heads[i][hs:hs + C.WIN])
        L = C.lighting_gain(prev_last, C.apply_frame(align_f, M, g, b, sh, Ww, Hh)).clamp(0.80, 1.25)
        def _build(aa, M=M, g=g, b=b, sh=sh, L=L, sf=next_first):
            Mw, gw, bw, sw, Lw = C.scale_strength(M, g, b, sh, L, aa)
            return C.apply_frame(sf, Mw, gw, bw, sw, Ww, Hh, Lw)
        a = C.gate_strength(prev_last, _build)
        if a > 0:                                        # crop-cap guard (mirror of normalize_chain)
            Me = C.scale_strength(M, g, b, sh, L, a)[0]
            if C.crop_zoom_for([Me], Ww, Hh)[0, 0] > C.CROP_CAP:
                a = 0.0
        base = float((tails[i - 1][1:] - tails[i - 1][:-1]).abs().flatten(1).mean(1).mean())
        est.append((i, k, prev_last, next_first, M, g, b, sh, L, a, base))

    eff = [C.scale_strength(e[4], e[5], e[6], e[7], e[8], e[9])[0] for e in est]
    crop = C.crop_zoom_for(eff, Ww, Hh)

    ident = (np.eye(2, 3, dtype=np.float32), np.ones(3), np.zeros(3), 1.0, None)
    rows = []
    for (i, k, prev_last, next_first, M, g, b, sh, L, a, base) in est:
        # do-nothing baseline: literal concat, no processing on either side
        raw_l1 = float((next_first - prev_last).abs().mean())
        # our output's seam: BOTH sides rendered with the same uniform crop (common-mode) —
        # prev with identity correction (its body), next with the gated correction
        prev_r = C.apply_frame(prev_last, *ident[:4], Ww, Hh, ident[4], crop=crop)
        Mw, gw, bw, sw, Lw = C.scale_strength(M, g, b, sh, L, a)
        corr_first = C.apply_frame(next_first, Mw, gw, bw, sw, Ww, Hh, Lw, crop=crop)
        cor_l1 = float((corr_first - prev_r).abs().mean())
        rows.append({
            "seam": f"{i}->{i+1}", "overlap": k, "base": base, "alpha": a,
            "raw_l1": raw_l1, "raw_ratio": raw_l1 / base if base else float("nan"),
            "raw_ssim": ssim(prev_last, next_first), "raw_dE": delta_e76(prev_last, next_first),
            "cor_l1": cor_l1, "cor_ratio": cor_l1 / base if base else float("nan"),
            "cor_ssim": ssim(prev_r, corr_first), "cor_dE": delta_e76(prev_r, corr_first),
        })
    return rows


def _print(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'seam':7} {'ov':3} {'a':>4} {'base':>7} | {'raw_L1':>7} {'ratio':>6} {'ssim':>6} {'dE':>6} "
          f"| {'cor_L1':>7} {'ratio':>6} {'ssim':>6} {'dE':>6}")
    for r in rows:
        print(f"{r['seam']:7} {r['overlap']:3d} {r.get('alpha',1):4.2f} {r['base']:7.4f} | "
              f"{r['raw_l1']:7.4f} {r['raw_ratio']:6.2f} {r['raw_ssim']:6.3f} {r['raw_dE']:6.2f} | "
              f"{r['cor_l1']:7.4f} {r['cor_ratio']:6.2f} {r['cor_ssim']:6.3f} {r['cor_dE']:6.2f}")
    # aggregate: mean corrected/raw improvement
    imp_l1 = np.mean([r["raw_l1"] - r["cor_l1"] for r in rows])
    imp_ss = np.mean([r["cor_ssim"] - r["raw_ssim"] for r in rows])
    imp_de = np.mean([r["raw_dE"] - r["cor_dE"] for r in rows])
    print(f"mean improvement: L1 {imp_l1:+.4f}  SSIM {imp_ss:+.4f}  dE {imp_de:+.3f}  "
          f"(positive = better)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="+")
    ap.add_argument("--mode", default="tight")
    ap.add_argument("--overlap", default="auto")
    a = ap.parse_args()
    paths = _resolve(a.spec)
    _print(evaluate(paths, a.mode, a.overlap), f"{a.spec} mode={a.mode}")
