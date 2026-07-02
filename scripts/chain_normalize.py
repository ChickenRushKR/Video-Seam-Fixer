"""CLI wrapper for the v1 chain boundary-normalizer (vbf.normalize.chain).

    python scripts/chain_normalize.py A.mp4 B.mp4 C.mp4 D.mp4 --out experiments/FINAL/run --mode tight

Back-compat: `python scripts/chain_normalize.py 3` runs the video3 sample chain.
"""

import argparse

from vbf.normalize import normalize_chain


def main():
    ap = argparse.ArgumentParser(description="Chain boundary normalization (v1)")
    ap.add_argument("clips", nargs="+", help="clip paths in order (or a single '2'/'3' sample id)")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--mode", default="tight", choices=["tight", "balanced", "local"],
                    help="tight/balanced: whole-chain consistent (accumulates); local: per-seam ramp, bodies untouched (best for long chains)")
    ap.add_argument("--ramp-frames", type=int, default=12, help="local mode: fade the seam correction to 0 over this many frames")
    ap.add_argument("--no-drop-dup", action="store_true", help="keep the duplicate seam frame (= --overlap 0)")
    ap.add_argument("--overlap", default=None, help="duplicate frames per seam to drop: int (0/1/K, default 1) or 'auto' to detect per seam")
    ap.add_argument("--no-slow", action="store_true", help="skip the boundary slow-mo comparison")
    ap.add_argument("--interpolate", type=int, default=0, help="v2: insert K synthesized frames per seam (0=off; 1 keeps length)")
    ap.add_argument("--interp-backend", default="rife", choices=["rife", "flow"], help="interpolation backend (rife falls back to flow)")
    args = ap.parse_args()

    clips = args.clips
    if len(clips) == 1 and clips[0] in ("2", "3"):        # sample shortcut
        v = clips[0]
        clips = [f"samples/video{v}-{c}.mp4" for c in "ABCD"]
        out = args.out or f"experiments/FINAL/video{v}_v1_{args.mode}"
    else:
        out = args.out or "experiments/FINAL/chain_run"

    def prog(stage, frac, msg):
        print(f"[{frac*100:5.1f}%] {stage:8} {msg}")

    ov = args.overlap
    if ov is not None and ov.lstrip("-").isdigit():
        ov = int(ov)
    r = normalize_chain(clips, out, mode=args.mode, drop_dup=not args.no_drop_dup,
                        overlap=ov, make_slow=not args.no_slow, ramp_frames=args.ramp_frames,
                        interpolate=args.interpolate, interp_backend=args.interp_backend, progress=prog)
    print(f"\n{r.num_frames}f @ {r.fps}fps in {r.seconds}s (overlap={r.overlap}) -> {r.full_path}")
    if r.interpolate:
        print(f"interpolation: {r.interpolate} frame(s)/seam via {r.interp_backend}")
    if r.slow_path:
        print(f"boundary slow-mo -> {r.slow_path}")
    print("scale x/y %:", r.transforms["scale_xy_pct"])
    print("colour gain:", r.transforms["colour_gain"], " sharpness:", r.transforms["sharpness"],
          " lighting:", r.transforms["lighting_gain"])
    for s in r.seams:
        print(f"  seam {s['pair']}: overlap={s['overlap']}  raw_gap {s['raw_gap']}  scale {s['scale_x_pct']}/{s['scale_y_pct']}%")


if __name__ == "__main__":
    main()
