// Video Boundary Fixer — v1 web UI
const $ = (s) => document.querySelector(s);
const slotsEl = $("#slots");
let slots = [];            // { file, url, id }
let poll = null;

// ---------- slot management ----------
function addFiles(fileList) {
  for (const f of fileList) {
    if (!f.type.startsWith("video/")) continue;
    slots.push({ file: f, url: URL.createObjectURL(f), id: Math.random().toString(36).slice(2) });
  }
  render();
}
function removeSlot(id) {
  const s = slots.find((x) => x.id === id);
  if (s) URL.revokeObjectURL(s.url);
  slots = slots.filter((x) => x.id !== id);
  render();
}
function move(id, d) {
  const i = slots.findIndex((x) => x.id === id);
  const j = i + d;
  if (j < 0 || j >= slots.length) return;
  [slots[i], slots[j]] = [slots[j], slots[i]];
  render();
}
function fmtSize(b) { return b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : (b / 1e3).toFixed(0) + " KB"; }

function render() {
  slotsEl.innerHTML = "";
  slots.forEach((s, i) => {
    const li = document.createElement("li");
    li.className = "slot";
    li.draggable = true;
    li.dataset.id = s.id;
    li.innerHTML = `
      <span class="idx">${i + 1}</span>
      <video class="thumb" src="${s.url}#t=0.1" muted preload="metadata"></video>
      <div class="meta"><div class="name">${s.file.name}</div><div class="size">${fmtSize(s.file.size)}</div></div>
      <div class="slot-actions">
        <button class="mini" data-act="up" ${i === 0 ? "disabled" : ""}>▲</button>
        <button class="mini" data-act="down" ${i === slots.length - 1 ? "disabled" : ""}>▼</button>
        <button class="mini danger" data-act="rm">✕</button>
      </div>`;
    li.querySelector('[data-act="up"]').onclick = () => move(s.id, -1);
    li.querySelector('[data-act="down"]').onclick = () => move(s.id, 1);
    li.querySelector('[data-act="rm"]').onclick = () => removeSlot(s.id);
    // drag reorder
    li.addEventListener("dragstart", (e) => { e.dataTransfer.setData("id", s.id); li.classList.add("dragging"); });
    li.addEventListener("dragend", () => li.classList.remove("dragging"));
    li.addEventListener("dragover", (e) => e.preventDefault());
    li.addEventListener("drop", (e) => {
      e.preventDefault();
      const from = e.dataTransfer.getData("id");
      const to = s.id;
      if (from === to) return;
      const fi = slots.findIndex((x) => x.id === from);
      const el = slots.splice(fi, 1)[0];
      const ti = slots.findIndex((x) => x.id === to);
      slots.splice(ti, 0, el);
      render();
    });
    slotsEl.appendChild(li);
  });
  $("#emptyHint").classList.toggle("hidden", slots.length > 0);
  $("#runBtn").disabled = slots.length < 2;
}

// ---------- inputs ----------
$("#interpOn").onchange = (e) => { $("#interpOpts").style.display = e.target.checked ? "flex" : "none"; };
$("#addBtn").onclick = () => $("#fileInput").click();
$("#fileInput").onchange = (e) => { addFiles(e.target.files); e.target.value = ""; };
$("#clearBtn").onclick = () => { slots.forEach((s) => URL.revokeObjectURL(s.url)); slots = []; render(); };

// page-level drag & drop
["dragover", "drop"].forEach((ev) =>
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "drop" && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  })
);

// ---------- run + poll ----------
const STEP_ORDER = ["analyze", "render", "compare", "done"];
$("#runBtn").onclick = async () => {
  if (slots.length < 2) return;
  hide("#resultPanel"); hide("#err"); show("#progressPanel");
  $("#runBtn").disabled = true;
  setProgress(0, "queued", "업로드 중…", 0);

  const fd = new FormData();
  slots.forEach((s) => fd.append("clips", s.file, s.file.name));
  fd.append("mode", $("#mode").value);
  fd.append("overlap", $("#overlap").value || "1");
  fd.append("interpolate", $("#interpOn").checked ? $("#interpK").value : "0");
  fd.append("interp_backend", $("#interpBackend").value);

  let job;
  try {
    const r = await fetch("/api/run", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "run failed");
    job = j.job_id;
  } catch (e) { return fail(e.message); }

  poll = setInterval(async () => {
    try {
      const r = await fetch(`/api/status/${job}`);
      const s = await r.json();
      if (!r.ok) throw new Error(s.error || "status failed");
      setProgress((s.frac || 0) * 100, s.stage, s.message, s.elapsed || 0);
      if (s.status === "done") { clearInterval(poll); showResult(job, s.result); }
      else if (s.status === "error") { clearInterval(poll); fail(s.message); }
    } catch (e) { clearInterval(poll); fail(e.message); }
  }, 700);
};

function setProgress(pct, stage, msg, elapsed) {
  $("#barFill").style.width = Math.max(2, pct) + "%";
  $("#pct").textContent = Math.round(pct) + "%";
  $("#stage").textContent = stage || "";
  $("#msg").textContent = msg || "";
  $("#timer").textContent = (elapsed || 0).toFixed(1) + "s";
  const active = STEP_ORDER.indexOf(stage);
  $$("#steps li").forEach((li, i) => {
    li.classList.toggle("active", STEP_ORDER.indexOf(li.dataset.k) === active);
    li.classList.toggle("done", STEP_ORDER.indexOf(li.dataset.k) < active || stage === "done");
  });
}

function showResult(job, res) {
  $("#runBtn").disabled = false;
  const t = res.transforms || {};
  const seamRows = (res.seams || []).map((s) =>
    `<tr><td>${s.pair}</td><td>${s.raw_gap}</td><td>${s.scale_x_pct}% / ${s.scale_y_pct}%</td></tr>`).join("");
  $("#stats").innerHTML = `
    <div class="stat"><b>${res.num_frames}</b><span>frames @ ${res.fps}fps</span></div>
    <div class="stat"><b>${res.seconds}s</b><span>처리 시간</span></div>
    <div class="stat"><b>${res.mode}</b><span>모드</span></div>
    <div class="stat"><b>${res.overlap}</b><span>중복 프레임/경계</span></div>
    ${res.interpolate ? `<div class="stat"><b>K=${res.interpolate}</b><span>보간 (${res.interp_backend})</span></div>` : ""}
    <table class="seams"><thead><tr><th>경계</th><th>raw 갭</th><th>누적 스케일 x/y</th></tr></thead><tbody>${seamRows}</tbody></table>`;
  const bust = "?t=" + Date.now();
  if (res.has_slow) { $("#slowVid").src = `/api/video/${job}/slow${bust}`; $("#slowDl").href = `/api/video/${job}/slow`; }
  if (res.has_full) { $("#fullVid").src = `/api/video/${job}/full${bust}`; $("#fullDl").href = `/api/video/${job}/full`; }
  show("#resultPanel");
  $("#resultPanel").scrollIntoView({ behavior: "smooth" });
}

function fail(m) { $("#runBtn").disabled = false; const e = $("#err"); e.textContent = "오류: " + m; show("#err"); }
function show(s) { $(s).classList.remove("hidden"); }
function hide(s) { $(s).classList.add("hidden"); }
function $$(s) { return document.querySelectorAll(s); }

render();
