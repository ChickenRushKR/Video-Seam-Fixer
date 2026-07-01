# Video Boundary Fixer — local web app (v1)

Add generated clips in order, hit **Run**, and get the seam-normalized full video plus a
boundary slow-motion comparison. Wraps `vbf.normalize.normalize_chain` (the v1 core).

## Run

```bash
# from the repo root
pip install flask                      # only extra dep (torch/opencv/ffmpeg already used)
python webapp/server.py                # -> http://127.0.0.1:5000
```

Open http://127.0.0.1:5000, add ≥2 clips (drag to reorder), choose a mode, press **Run**.
Progress (stage + %, elapsed) polls live; when done, the full result and the
raw|fixed boundary slow-motion play inline with download links.

## What it does per boundary
geometry (anisotropic full-affine, background-locked) → colour/exposure → sharpness →
low-freq lighting → drop the duplicate seam frame. All chained so every clip lands in the
first clip's space (no downstream seam is broken).

**Modes:** `tight` = match each cut as closely as possible (recommended); `balanced` = keep
each clip's own look more natural, seams matched a bit less tightly.

## Notes
- Clips are resized to the first clip's resolution; fps is taken from the first clip.
- Outputs land in `experiments/webapp_runs/<job>/out/` (`result_full.mp4`,
  `result_boundaries_slow.mp4`).
- Processing streams one clip at a time (bounded memory); a 4×~290-frame 1080×1920 chain
  takes ~80s on this machine.
- Residual at each seam is genuine subject motion — going further needs frame
  interpolation or regeneration, not post-hoc alignment.
