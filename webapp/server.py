"""Local web app for the v1 chain boundary-normalizer.

Run:
    python webapp/server.py            # then open http://127.0.0.1:5000

Add clips in order, hit Run, watch the per-stage progress, and play/download the
full result + the boundary slow-motion comparison. Processing runs in a background
thread; the browser polls /api/status.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid

# make the repo importable no matter where this is launched from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402

from vbf.normalize import normalize_chain  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
WORK = os.path.join(ROOT, "experiments", "webapp_runs")
os.makedirs(WORK, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload cap

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _set(job_id, **kw):
    with LOCK:
        JOBS[job_id].update(kw)


def _run_job(job_id, clip_paths, out_dir, mode, drop_dup):
    t0 = time.time()

    def prog(stage, frac, message):
        _set(job_id, stage=stage, frac=round(float(frac), 3), message=message,
             elapsed=round(time.time() - t0, 1))

    try:
        _set(job_id, status="running")
        res = normalize_chain(clip_paths, out_dir, mode=mode, drop_dup=drop_dup,
                              make_slow=True, progress=prog)
        _set(job_id, status="done", frac=1.0, stage="done", message="Done",
             elapsed=res.seconds,
             result={
                 "num_frames": res.num_frames, "fps": res.fps, "seconds": res.seconds,
                 "mode": res.mode, "seams": res.seams, "transforms": res.transforms,
                 "has_full": bool(res.full_path and os.path.exists(res.full_path)),
                 "has_slow": bool(res.slow_path and os.path.exists(res.slow_path)),
             })
    except Exception as e:  # surface the error to the UI
        import traceback
        traceback.print_exc()
        _set(job_id, status="error", message=str(e))


@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.route("/api/run", methods=["POST"])
def api_run():
    files = request.files.getlist("clips")          # ordered by the client
    if len(files) < 2:
        return jsonify(error="add at least 2 clips"), 400
    mode = request.form.get("mode", "tight")
    drop_dup = request.form.get("drop_dup", "true") == "true"

    job_id = uuid.uuid4().hex[:12]
    in_dir = os.path.join(WORK, job_id, "in")
    out_dir = os.path.join(WORK, job_id, "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    clip_paths = []
    for i, f in enumerate(files):
        safe = f"{i:02d}_{os.path.basename(f.filename or f'clip{i}.mp4')}"
        p = os.path.join(in_dir, safe)
        f.save(p)
        clip_paths.append(p)

    with LOCK:
        JOBS[job_id] = {"status": "queued", "stage": "queued", "frac": 0.0,
                        "message": "Queued", "elapsed": 0.0, "out_dir": out_dir,
                        "clips": [os.path.basename(p) for p in clip_paths], "mode": mode}
    threading.Thread(target=_run_job, args=(job_id, clip_paths, out_dir, mode, drop_dup),
                     daemon=True).start()
    return jsonify(job_id=job_id)


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error="unknown job"), 404
        return jsonify({k: v for k, v in job.items() if k != "out_dir"})


@app.route("/api/video/<job_id>/<which>")
def api_video(job_id, which):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        return "unknown job", 404
    name = {"full": "result_full.mp4", "slow": "result_boundaries_slow.mp4"}.get(which)
    if not name:
        return "bad video", 400
    path = os.path.join(job["out_dir"], name)
    if not os.path.exists(path):
        return "not ready", 404
    return send_file(path, mimetype="video/mp4", conditional=True)  # conditional=True => Range/seek


if __name__ == "__main__":
    print("Video Boundary Fixer - http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
