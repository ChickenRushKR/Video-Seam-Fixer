"""Chain boundary-normalization core (v1).

Make the cut between consecutively-generated clips invisible. Each next clip is
generated slightly zoomed / shifted / brighter / sharper than the previous one
(and its first frame is ~= the previous clip's last frame — a 1-frame freeze).
We estimate that per-seam offset, compose it across the whole chain so every clip
lands in the first clip's reference space, apply it to each whole clip, and drop
the duplicate seam frame. Because both sides of every seam map to the same space,
the seam matches and no downstream seam is broken.

Pipeline per clip (all composed cumulatively to the reference):
  1. geometry   — full affine (6-DOF, anisotropic x/y), fit on the STATIC background
  2. colour/exposure — per-channel gain+bias (mean+std)
  3. sharpness  — high-freq energy ratio (blur/unsharp)
  4. lighting   — spatially-varying low-freq gain (subject-masked, very smooth)
  5. drop the duplicate first frame of each next clip (freeze -> continuous motion)

Memory: clips are processed ONE AT A TIME (never all in RAM at once). Estimation
reads only each clip's head/tail; rendering streams frame-by-frame to the writer.

The only heavy dependency is OpenCV (ORB + affine) + torch; ffmpeg is used for the
raw concat and the side-by-side slow-motion comparison.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vbf.io.video_io import load_video, save_video_stream, to_float, to_uint8

WIN = 8               # frames each side of a seam for colour/exposure/sharpness stats
GRID = (24, 14)       # lighting-gain grid (gh, gw) — small => very smooth, cannot form a blob
SLOW_RADIUS = 15      # frames each side of a seam shown in the slow-motion comparison
SLOW_FACTOR = 4       # slow-motion factor for the boundary comparison


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _np(t: torch.Tensor) -> np.ndarray:
    """torch [3,H,W] float -> uint8 HxWx3 RGB."""
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _blur(t: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Gaussian low-pass of [N,3,H,W] or [3,H,W]."""
    x = t.unsqueeze(0) if t.dim() == 3 else t
    ax = torch.arange(k) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 3.0) ** 2))
    g = (g / g.sum()).to(x)
    w = (g[:, None] * g[None, :]).expand(3, 1, k, k)
    return F.conv2d(x, w, padding=k // 2, groups=3)


def _bg_mask(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Static-background mask (255 where two frames match, 0 on the moving subject)."""
    d = np.abs(A.astype(np.int32) - B.astype(np.int32)).sum(2)
    thr = max(np.percentile(d, 70), 30)
    bg = (d <= thr).astype(np.uint8) * 255
    return cv2.erode(bg, np.ones((21, 21), np.uint8))


def seam_affine(prev_last: torch.Tensor, next_first: torch.Tensor) -> np.ndarray:
    """FULL affine (6-DOF: independent x/y scale + shear + rot + shift) mapping
    ``next_first -> prev_last``, fit on STATIC-BACKGROUND features only (the moving
    subject would otherwise bias the global transform). 3x3; identity on failure."""
    A, B = _np(prev_last), _np(next_first)
    ga, gb = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(6000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def fit(mask):
        ka, da = orb.detectAndCompute(ga, mask)
        kb, db = orb.detectAndCompute(gb, mask)
        if da is None or db is None or len(da) < 20 or len(db) < 20:
            return None
        m = sorted(bf.match(db, da), key=lambda x: x.distance)[:600]
        if len(m) < 20:
            return None
        src = np.float32([kb[x.queryIdx].pt for x in m])
        dst = np.float32([ka[x.trainIdx].pt for x in m])
        M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
        return M if M is not None and int(inl.sum()) >= 15 else None

    M = fit(_bg_mask(A, B))          # background-locked fit
    if M is None:                    # fall back to whole-frame if too few background matches
        M = fit(None)
    H = np.eye(3, dtype=np.float64)
    if M is not None:
        H[:2] = M
    return H


def axis_scales(H: np.ndarray) -> tuple[float, float]:
    """(sx, sy) axis scale factors of a 3x3 affine (column norms)."""
    L = H[:2, :2]
    return float(np.hypot(L[0, 0], L[1, 0])), float(np.hypot(L[0, 1], L[1, 1]))


def color_sharp_fit(prev_win: torch.Tensor, next_win: torch.Tensor):
    """Match next_win -> prev_win: per-channel colour gain/bias (mean+std) and a scalar
    sharpness (high-freq energy) ratio. Returns (gain[3], bias[3], sharp)."""
    pm, ps = prev_win.mean((0, 2, 3)), prev_win.std((0, 2, 3))
    nm, ns = next_win.mean((0, 2, 3)), next_win.std((0, 2, 3))
    g = (ps / (ns + 1e-5)).clamp(0.85, 1.18)
    b = pm - g * nm
    hp = (prev_win - _blur(prev_win)).std()
    hn = (next_win - _blur(next_win)).std()
    sharp = float((hp / (hn + 1e-5)).clamp(0.75, 1.4))
    return g.numpy(), b.numpy(), sharp


def lighting_gain(pf: torch.Tensor, nf: torch.Tensor) -> torch.Tensor:
    """Smooth low-freq gain (grid [3,gh,gw]) mapping nf's lighting field -> pf's, on the
    STATIC background only (subject region set to 1 so it is never relit)."""
    lp, ln = _blur(pf.unsqueeze(0))[0], _blur(nf.unsqueeze(0))[0]
    gmap = (lp / (ln + 1e-3)).clamp(0.85, 1.20)
    diff = (pf - nf).abs().mean(0, keepdim=True)
    subj = (diff > torch.quantile(diff.flatten(), 0.70)).float()
    gmap = gmap * (1 - subj) + subj
    return F.adaptive_avg_pool2d(gmap, GRID)


def apply_frame(f, M, g, b, sharp, Ww, Hh, L=None):
    """geometry -> colour(gain,bias) -> sharpness -> optional low-freq lighting, on [3,H,W]."""
    img = _np(f)
    w = cv2.warpAffine(img, M, (Ww, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    x = torch.from_numpy(w).permute(2, 0, 1).float() / 255.0
    x = x * torch.from_numpy(g).float().view(3, 1, 1) + torch.from_numpy(b).float().view(3, 1, 1)
    if abs(sharp - 1.0) > 1e-3:
        lo = _blur(x)[0]
        x = lo + sharp * (x - lo)
    if L is not None:
        gm = F.interpolate(L.unsqueeze(0), size=(x.shape[1], x.shape[2]), mode="bilinear", align_corners=False)[0]
        x = x * gm
    return x.clamp(0, 1)


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #
@dataclass
class ChainResult:
    full_path: str
    slow_path: str | None
    num_frames: int
    fps: float
    mode: str
    seams: list[dict] = field(default_factory=list)   # per-seam metrics
    seconds: float = 0.0
    transforms: dict = field(default_factory=dict)     # cumulative scale/colour/lighting per clip
    overlap: int = 1                                   # duplicate frames dropped per seam
    interpolate: int = 0                               # midpoints inserted per seam (0 = off)
    interp_backend: str = ""                           # backend actually used


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def normalize_chain(
    clip_paths: list[str],
    out_dir: str,
    mode: str = "tight",
    drop_dup: bool = True,
    overlap: int | None = None,
    make_slow: bool = True,
    interpolate: int = 0,
    interp_backend: str = "rife",
    progress=None,
) -> ChainResult:
    """Normalize a chain of clips so the cuts are invisible; write the full result and a
    side-by-side (raw|fixed) boundary slow-motion.

    ``overlap`` (int): how many frames each next clip **duplicates** from the previous clip's
    tail (``next[0..overlap-1] ≈ prev[-overlap..-1]``). Those duplicates are dropped, and the
    per-seam transform is estimated from the true corresponding pair ``(prev[-1], next[overlap-1])``
    — a pure-drift pair (no motion), so alignment is cleaner. ``overlap=1`` is the common
    single-frame conditioning case (default). ``overlap=0`` = clips are already continuous
    (nothing dropped). If ``overlap`` is None it derives from ``drop_dup`` (True→1, False→0).

    ``interpolate`` (v2): insert this many synthesized in-between frames at each seam to smooth
    the residual motion step (0 = off). ``interp_backend``: ``rife`` (learned, handles occlusion;
    falls back to ``flow`` if ccvfi is unavailable) or ``flow`` (RAFT warp, no extra deps).

    ``progress(stage: str, frac: float, message: str)`` is called throughout (frac 0..1).
    """
    prog = progress or (lambda *a, **k: None)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    if len(clip_paths) < 2:
        raise ValueError("need at least 2 clips")
    N = len(clip_paths)
    ov = overlap if overlap is not None else (1 if drop_dup else 0)   # duplicate frames per seam
    ov = max(0, ov)

    # ---- PHASE 1: read each clip's seam neighbourhood + stats (one clip at a time) ----
    # align_f[i] = the next-clip frame that DUPLICATES prev[-1] (index overlap-1); heads skip the
    # overlapping duplicates so colour/sharpness stats use genuine post-seam content.
    align_f, last_f, heads, tails, lengths, stats = [], [], [], [], [], []
    fps = 24.0
    ref_hw = None
    for i, p in enumerate(clip_paths):
        prog("analyze", 0.05 + 0.25 * i / N, f"Analyzing clip {i + 1}/{N}")
        v = load_video(p)
        if i == 0:
            fps = v.fps
            ref_hw = (v.frames.shape[2], v.frames.shape[3])
        fr = v.frames
        if (fr.shape[2], fr.shape[3]) != ref_hw:            # keep the chain a single canvas size
            fr = to_uint8(F.interpolate(to_float(fr), size=ref_hw, mode="bilinear", align_corners=False))
        T = fr.shape[0]
        lengths.append(T)
        ai = min(max(ov - 1, 0), T - 1)               # frame duplicating prev[-1]
        hs = min(ov, T - 1)                            # post-overlap head start
        align_f.append(to_float(fr[ai:ai + 1])[0])
        last_f.append(to_float(fr[-1:])[0])
        heads.append(to_float(fr[hs:hs + WIN]))
        tails.append(to_float(fr[-WIN:]))
        sub = to_float(fr[:: max(1, fr.shape[0] // 20)])
        stats.append((sub.mean((0, 2, 3)), sub.std((0, 2, 3)), float((sub - _blur(sub)).std())))
        del v, fr

    Hh, Ww = ref_hw

    # ---- geometry: cumulative full affine to clip-0 space ----
    cum_H = [np.eye(3)]
    for i in range(1, N):
        cum_H.append(cum_H[-1] @ seam_affine(last_f[i - 1], align_f[i]))
    sxy = [axis_scales(H) for H in cum_H]
    zoom_x = max(1.0 / min(s[0] for s in sxy), 1.0)
    zoom_y = max(1.0 / min(s[1] for s in sxy), 1.0)
    cz = np.array([[zoom_x, 0, (1 - zoom_x) * Ww / 2],
                   [0, zoom_y, (1 - zoom_y) * Hh / 2], [0, 0, 1]], dtype=np.float64)
    Ms = [(cz @ H)[:2].astype(np.float32) for H in cum_H]

    # ---- colour / exposure / sharpness ----
    if mode == "tight":
        cum_g, cum_b, cum_s = [np.ones(3)], [np.zeros(3)], [1.0]
        for i in range(1, N):
            g, b, sh = color_sharp_fit(tails[i - 1], heads[i])
            cum_g.append(cum_g[-1] * g)
            cum_b.append(cum_g[-2] * b + cum_b[-1])
            cum_s.append(cum_s[-1] * sh)
    else:  # balanced: map every clip to the average look
        tgt_m = torch.stack([s[0] for s in stats]).mean(0)
        tgt_s = torch.stack([s[1] for s in stats]).mean(0)
        tgt_h = float(np.mean([s[2] for s in stats]))
        cum_g, cum_b, cum_s = [], [], []
        for i in range(N):
            g = (tgt_s / (stats[i][1] + 1e-5)).clamp(0.85, 1.18)
            cum_g.append(g.numpy())
            cum_b.append((tgt_m - g * stats[i][0]).numpy())
            cum_s.append(float(np.clip(tgt_h / (stats[i][2] + 1e-5), 0.8, 1.25)))

    # ---- lighting: spatially-varying low-freq gain, chained in reference space ----
    cum_L = [torch.ones(3, *GRID)]
    for i in range(1, N):
        pf = apply_frame(last_f[i - 1], Ms[i - 1], cum_g[i - 1], cum_b[i - 1], cum_s[i - 1], Ww, Hh)
        nf = apply_frame(align_f[i], Ms[i], cum_g[i], cum_b[i], cum_s[i], Ww, Hh)
        cum_L.append((cum_L[-1] * lighting_gain(pf, nf)).clamp(0.80, 1.25))

    # ---- PHASE 2: render (stream one clip at a time), drop duplicate seam frames,
    #      optionally insert interpolated in-between frames at each seam (v2) ----
    full_path = os.path.join(out_dir, "result_full.mp4")
    itp = None
    if interpolate > 0:
        from vbf.interp import get_interpolator
        itp = get_interpolator(interp_backend, device=("cuda" if torch.cuda.is_available() else "cpu"))
        prog("render", 0.30, f"Interpolation on ({getattr(itp, 'backend', interp_backend)}, K={interpolate})")

    def _apply(fr_u8, k, i):
        f = to_float(fr_u8[k:k + 1])[0]
        return apply_frame(f, Ms[i], cum_g[i], cum_b[i], cum_s[i], Ww, Hh, cum_L[i])

    def gen():
        prev_corr_last = None
        for i, p in enumerate(clip_paths):
            prog("render", 0.30 + 0.55 * i / N, f"Rendering clip {i + 1}/{N}")
            v = load_video(p)
            fr = v.frames
            if (fr.shape[2], fr.shape[3]) != ref_hw:
                fr = to_uint8(F.interpolate(to_float(fr), size=ref_hw, mode="bilinear", align_corners=False))
            start = min(ov, fr.shape[0] - 1) if i > 0 else 0    # drop the overlap[0..ov-1] duplicates
            first_corr = _apply(fr, start, i)
            if itp is not None and i > 0 and prev_corr_last is not None:   # synth midpoints at the seam
                for j in range(1, interpolate + 1):
                    t = j / (interpolate + 1)
                    yield to_uint8(itp.interpolate(prev_corr_last, first_corr, t))
            yield to_uint8(first_corr)
            for k in range(start + 1, fr.shape[0]):
                corr = _apply(fr, k, i)
                yield to_uint8(corr)
            prev_corr_last = corr if fr.shape[0] > start + 1 else first_corr
            del v, fr

    n = save_video_stream(full_path, gen(), fps=fps)

    # ---- seam positions (raw vs de-duped, incl. inserted frames) + per-seam metric ----
    raw_seam, fix_seam, emitted = [], [], 0
    for i in range(N):
        if i > 0:
            emitted += interpolate           # midpoints inserted before this clip's frames
            fix_seam.append(emitted)         # window centres on the interpolated seam region
        emitted += lengths[i] - (min(ov, lengths[i] - 1) if i > 0 else 0)
    acc = 0
    for i in range(N):
        if i > 0:
            raw_seam.append(acc)
        acc += lengths[i]

    seams = []
    for i in range(1, N):
        raw_gap = float((align_f[i] - last_f[i - 1]).abs().mean())   # drift on the duplicate pair
        seams.append({"index": i, "pair": f"{i}->{i + 1}", "raw_gap": round(raw_gap, 4),
                      "scale_x_pct": round((sxy[i][0] - 1) * 100, 2),
                      "scale_y_pct": round((sxy[i][1] - 1) * 100, 2)})

    # ---- PHASE 3: raw concat + side-by-side slow comparison (ffmpeg) ----
    slow_path = None
    if make_slow:
        prog("compare", 0.88, "Building boundary comparison")
        slow_path = _build_slow(clip_paths, full_path, out_dir, raw_seam, fix_seam, fps)

    prog("done", 1.0, "Done")
    return ChainResult(
        full_path=full_path, slow_path=slow_path, num_frames=n, fps=fps, mode=mode,
        seams=seams, seconds=round(time.time() - t0, 1),
        overlap=ov,
        interpolate=interpolate,
        interp_backend=(getattr(itp, "backend", interp_backend) if itp is not None else ""),
        transforms={
            "scale_xy_pct": [[round((x - 1) * 100, 2), round((y - 1) * 100, 2)] for x, y in sxy],
            "colour_gain": [round(float(np.mean(g)), 3) for g in cum_g],
            "sharpness": [round(float(s), 3) for s in cum_s],
            "lighting_gain": [round(float(L.mean()), 3) for L in cum_L],
        },
    )


def _win_expr(seams: list[int], r: int) -> str:
    return "+".join(f"between(n\\,{max(0, s - r)}\\,{s + r})" for s in seams)


def _build_slow(clip_paths, full_path, out_dir, raw_seam, fix_seam, fps) -> str | None:
    """RAW (straight concat) | FIXED (normalized) side-by-side, boundary regions, slowed.
    Each side is windowed around ITS OWN seam positions so the same moment lines up."""
    raw_concat = os.path.join(out_dir, "raw_concat.mp4")
    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]
    n = len(clip_paths)
    concat = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[o]"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs,
                        "-filter_complex", concat, "-map", "[o]", "-r", str(int(round(fps))),
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", raw_concat],
                       check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    slow_path = os.path.join(out_dir, "result_boundaries_slow.mp4")
    raw_sel = f"select='{_win_expr(raw_seam, SLOW_RADIUS)}',setpts=N/{SLOW_FACTOR}/TB"
    fix_sel = f"select='{_win_expr(fix_seam, SLOW_RADIUS)}',setpts=N/{SLOW_FACTOR}/TB"
    dt = "drawtext=text='{t}':x=24:y=24:fontsize=54:fontcolor=yellow:box=1:boxcolor=black@0.5"
    fc = (f"[0:v]{raw_sel},{dt.format(t='RAW')}[l];"
          f"[1:v]{fix_sel},{dt.format(t='FIXED')}[r];[l][r]hstack=inputs=2[o]")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_concat, "-i", full_path,
                        "-filter_complex", fc, "-map", "[o]", "-r", str(int(round(fps))),
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", slow_path],
                       check=True)
    except subprocess.CalledProcessError:
        # drawtext (fontconfig) can be unavailable; retry without labels
        fc2 = (f"[0:v]{raw_sel}[l];[1:v]{fix_sel}[r];[l][r]hstack=inputs=2[o]")
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_concat, "-i", full_path,
                            "-filter_complex", fc2, "-map", "[o]", "-r", str(int(round(fps))),
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", slow_path],
                           check=True)
        except subprocess.CalledProcessError:
            return None
    return slow_path
