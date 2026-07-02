<!-- Language: [🇰🇷 한국어](README.md) · **🇺🇸 English** -->

# seam-fixer

**Erase the seam between consecutively-generated video clips** — deterministic alignment + learned interpolation, no retraining.

```
A ──┐  seam  ┌── B ──┐  seam  ┌── C ──┐  seam  ┌── D
    ▼        ▼       ▼        ▼       ▼        ▼
 [ scale·colour·exposure·sharpness·lighting drift + a duplicate freeze frame ]
                    │  seam-fixer
                    ▼
A ─────────────────────────────────────── D   (one continuous take)
```

## Problem
Extend a video clip by clip (A→B→C→D, each **conditioned on the previous clip's last frame**) and
the joins **pop (zoom), flicker (colour/exposure), and hitch**. Cause: per-clip **anisotropic
scale (x≠y) + colour/exposure/sharpness/lighting drift** (accumulating down the chain) + a
**duplicate freeze frame** (`B[0]≈A[-1]`).

![Seam example](docs/seam_example.png)

> Top: `A_last`·`B_first` (nearly identical). Bottom: their difference ×5 — RAW (left) glows all
> over incl. background edges = the seam (0.032); after alignment (right) the background is gone
> and **only the person** remains (0.013) = what Stage 2 interpolation handles.

## How it works (2 stages)

**Stage 1 · Alignment (deterministic).** Map **every clip into the first clip's space** (cumulative)
so both sides of each cut match and no downstream cut breaks (a local fix would change a clip's
tail = the next clip's anchor). Per clip:
- **geometry** — full affine (independent x/y scale; anisotropic, so not a similarity transform), matched on the **static background only**
- **colour/exposure** (per-channel mean+std) · **sharpness** (high-freq) · **lighting** (spatially-varying low-freq gain, subject-masked)
- **drop the duplicate frame** → freeze becomes normal motion. If the generator repeated **K
  frames** at each boundary, use **`--overlap K`** (drops K, aligns on the `prev[-1]↔next[K-1]`
  duplicate pair — actually more accurate). Default 1; 0 drops nothing; **`--overlap auto`
  detects K per seam** (when you don't know how many frames repeat).
- modes: `tight` (max seam match, default) / `balanced` (keep clips natural) / **`local`** (ramp
  the correction only on each clip's head, bodies stay original → no accumulation, **preserves
  resolution & colour**; recommended for long chains)

**Stage 2 · Interpolation (learned).** What remains is subject motion (`prev[-1]`≠`next[0]` in time).
Synthesize K midpoints per seam (`K=1` + duplicate-drop = length-preserving).
- `rife` (recommended) — learned (RIFE v4.26, `ccvfi`); reconstructs occluded/fast parts (a hand) cleanly.
- `flow` — RAFT warp, no deps; fine for small motion, ghosts on fast occlusion. (auto-fallback if `ccvfi` absent)

## Install
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your GPU
pip install opencv-python-headless imageio[ffmpeg] numpy flask
pip install ccvfi        # (optional) RIFE backend; falls back to flow. ffmpeg must be on PATH.
```

## Usage
```bash
# web app (recommended): add clips in order -> Run -> progress -> result + boundary slow-mo
python webapp/server.py            # http://127.0.0.1:5000

# CLI
python scripts/chain_normalize.py A.mp4 B.mp4 C.mp4 D.mp4 --out out/ --mode tight
python scripts/chain_normalize.py 3 --interpolate 1 --interp-backend rife   # sample + Stage 2
```
```python
from vbf.normalize import normalize_chain
r = normalize_chain(["A.mp4","B.mp4","C.mp4","D.mp4"], "out/", mode="tight", interpolate=1, interp_backend="rife")
```
Outputs: `result_full.mp4` (normalized chain), `result_boundaries_slow.mp4` (each cut 4× slow, RAW∣FIXED).
Clips are resized to the first clip's resolution/fps and streamed one at a time. 4×~290f 1080×1920 ≈ 80s (RTX 5090).

## Layout
```
vbf/normalize/chain.py   # core: alignment (Stage 1) + interpolation (Stage 2)
vbf/interp/              # backends: flow (RAFT) + rife (ccvfi)
scripts/                 # chain_normalize.py (CLI) · eval_boundaries.py · interp_experiment.py
webapp/                  # Flask server + UI
```

## Limitations
- Assumes one continuous scene per chain (shared background); hard scene cuts are out of scope.
- Corrects global per-clip drift (within-clip drift is approximated); very large / heavily-occluded motion isn't perfect even with `rife`.
- Every decision is metric-verified (SSIM · Lab ΔE · motion baseline · downstream) — `scripts/eval_boundaries.py`.
