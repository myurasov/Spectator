// SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
// SPDX-License-Identifier: Apache-2.0
//
// Spectator Web UI — single-page client. No framework, no build step.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const STATUS_POLL_MS = 4000;
const JOBS_POLL_MS = 2500;

let selectedFile = null;
const liveSockets = new Map(); // job_id -> WebSocket
const expandedJobs = new Set();

// ---- helpers ---------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function fmtDuration(s) {
  if (s == null || s < 0) return "—";
  const sec = Math.round(s);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const r = sec % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m ${String(r).padStart(2, "0")}s`;
  if (m) return `${m}m ${String(r).padStart(2, "0")}s`;
  return `${r}s`;
}

function fmtRTF(rt) {
  if (rt == null || rt === 0) return "—";
  if (rt >= 1) return `${rt.toFixed(2)}× faster`;
  return `${(1 / rt).toFixed(2)}× slower`;
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText} — ${text || "(no detail)"}`);
  }
  if (r.status === 204) return null;
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.text();
}

// ---- status bar ------------------------------------------------------------

function setPill(id, label, kind) {
  const e = document.getElementById(id);
  if (!e) return;
  e.textContent = label;
  e.classList.remove("ok", "warn", "err");
  if (kind) e.classList.add(kind);
}

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    setPill("status-target", `target: ${s.target}`, s.target === "local" ? "ok" : "warn");
    if (s.target === "local") {
      setPill("status-audio",
        `audio-venv: ${s.audio_venv_installed ? "installed" : "missing"}`,
        s.audio_venv_installed ? "ok" : "err");
    } else {
      setPill("status-audio", "audio-venv: (remote)", "warn");
    }
    setPill("status-vss",
      `VSS: ${s.vss_api_reachable ? "reachable" : "unreachable"}`,
      s.vss_api_reachable ? "ok" : "warn");
    setPill("status-jobs", `jobs in flight: ${s.jobs_in_flight}`, s.jobs_in_flight > 0 ? "warn" : "ok");
    $("#status-version").textContent = `v${s.spectator_version}`;
    $("#footer-version").textContent = `Spectator v${s.spectator_version} · workdir ${s.workdir} · target ${s.target}`;
  } catch (e) {
    setPill("status-target", "server unreachable", "err");
  }
}

// ---- VSS controls ----------------------------------------------------------

function bindVssControls() {
  const out = $("#vss-output");
  const setBusy = (b) => $$("#panel-vss button").forEach((btn) => btn.disabled = b);
  const writeOut = (txt) => { out.textContent = txt || "(no output)"; };

  $("#btn-vss-up").onclick = async () => {
    setBusy(true);
    writeOut("(VSS up: kicking off… first run is 30-45 min)");
    try {
      const r = await api("/api/vss/up", { method: "POST" });
      writeOut(`rc=${r.rc} ok=${r.ok}\n\n${r.stdout || ""}\n\n${r.stderr || ""}`);
    } catch (e) { writeOut(`error: ${e.message}`); }
    setBusy(false); refreshStatus();
  };
  $("#btn-vss-down").onclick = async () => {
    setBusy(true);
    writeOut("(VSS down: stopping…)");
    try {
      const r = await api("/api/vss/down", { method: "POST" });
      writeOut(`rc=${r.rc} ok=${r.ok}\n\n${r.stdout || ""}\n\n${r.stderr || ""}`);
    } catch (e) { writeOut(`error: ${e.message}`); }
    setBusy(false); refreshStatus();
  };
  $("#btn-vss-status").onclick = async () => {
    setBusy(true);
    writeOut("(VSS status: querying…)");
    try {
      const r = await api("/api/vss/status");
      writeOut(r.stdout || "(empty)");
    } catch (e) { writeOut(`error: ${e.message}`); }
    setBusy(false); refreshStatus();
  };
}

// ---- submit form -----------------------------------------------------------

function bindSubmitForm() {
  const dz = $("#dropzone");
  const fileInput = $("#file-input");
  const setFile = (f) => {
    selectedFile = f;
    $("#selected-file").textContent = f ? `${f.name} (${(f.size / 1e6).toFixed(1)} MB)` : "no file selected";
  };
  fileInput.addEventListener("change", () => setFile(fileInput.files[0] || null));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.remove("over");
  }));
  dz.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
      setFile(e.dataTransfer.files[0]);
      fileInput.files = e.dataTransfer.files;
    }
  });

  const kindSelect = $("#kind-select");
  const sync = () => {
    const isAudio = kindSelect.value === "audio";
    $$(".audio-only").forEach((e) => e.hidden = !isAudio);
    $$(".video-only").forEach((e) => e.hidden = isAudio);
  };
  kindSelect.addEventListener("change", sync); sync();

  $("#form-submit").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) { alert("Pick a file first."); return; }
    const fd = new FormData(e.target);
    if (!fd.get("file") || (fd.get("file").size || 0) === 0) {
      fd.set("file", selectedFile);
    }
    try {
      const job = await api("/api/jobs", { method: "POST", body: fd });
      console.log("submitted", job);
      e.target.reset(); setFile(null);
      kindSelect.dispatchEvent(new Event("change"));
      refreshJobs();
    } catch (err) {
      alert(`submit failed: ${err.message}`);
    }
  });
}

// ---- jobs table + WS -------------------------------------------------------

function jobRow(job) {
  const metrics = job.metrics || {};
  const progressPct = metrics.percent ?? 0;
  const tr = el("tr", { "data-job-id": job.id, class: expandedJobs.has(job.id) ? "expanded" : "" },
    el("td", {},
      el("code", {}, job.id.slice(0, 8)),
      el("br"),
      el("span", { class: "muted small" },
        job.input_path ? job.input_path.split("/").pop() : "—")
    ),
    el("td", {}, job.kind),
    el("td", {}, el("span", { class: `status-tag ${job.status}` }, job.status)),
    el("td", {},
      el("span", { class: "progress-bar" },
        el("span", { class: "progress-fill", style: `width: ${progressPct}%` })),
      `${(metrics.percent ?? 0).toFixed(0)}%`),
    el("td", {}, fmtRTF(metrics.rt_factor)),
    el("td", {}, fmtDuration(metrics.wall_clock_s)),
    el("td", {}, fmtDuration(metrics.eta_s)),
    el("td", { class: "outputs" }, ...(jobOutputLinks(job))),
    el("td", {},
      job.status === "running" || job.status === "queued"
        ? el("button", { class: "danger small", onclick: () => cancelJob(job.id) }, "kill")
        : el("button", { class: "small", onclick: () => toggleExpand(job.id) },
            expandedJobs.has(job.id) ? "hide" : "log"))
  );
  return tr;
}

function jobOutputLinks(job) {
  if (job.status !== "completed") return [el("span", { class: "muted small" }, "—")];
  if (job.kind !== "audio") return [el("span", { class: "muted small" }, "(video)")];
  // For audio: provide download links for the four sidecar formats.
  const stem = (job.input_path || "").split("/").pop().replace(/\.[^.]+$/, "");
  const files = [`${stem}.txt`, `${stem}.srt`, `${stem}.vtt`, `${stem}.json`, `${stem}.tsv`];
  return files.map((f) => el("a",
    { href: `/api/jobs/${job.id}/output/${encodeURIComponent(f)}`, download: f, class: "small" },
    f.split(".").pop()));
}

async function cancelJob(jobId) {
  try { await api(`/api/jobs/${jobId}`, { method: "DELETE" }); refreshJobs(); }
  catch (e) { alert(`cancel failed: ${e.message}`); }
}

async function toggleExpand(jobId) {
  if (expandedJobs.has(jobId)) expandedJobs.delete(jobId);
  else expandedJobs.add(jobId);
  await refreshJobs();
}

async function loadLog(jobId) {
  try { return await api(`/api/jobs/${jobId}/log`); }
  catch (e) { return `(error: ${e.message})`; }
}

async function refreshJobs() {
  let data;
  try { data = await api("/api/jobs"); }
  catch (e) { return; }
  const tbody = $("#jobs-tbody");
  tbody.innerHTML = "";
  if (!data.jobs || data.jobs.length === 0) {
    tbody.appendChild(el("tr", {},
      el("td", { colspan: 9, class: "muted" }, "(no jobs yet — submit one above)")));
    return;
  }
  for (const job of data.jobs) {
    tbody.appendChild(jobRow(job));
    if (expandedJobs.has(job.id)) {
      const logTd = el("td", { colspan: 9 },
        el("pre", { class: "log", id: `log-${job.id}` }, "(loading…)"));
      tbody.appendChild(el("tr", { class: "detail-row" }, logTd));
      loadLog(job.id).then((txt) => {
        const e = document.getElementById(`log-${job.id}`);
        if (e) e.textContent = txt || "(empty)";
      });
    }
    if (job.status === "running" && !liveSockets.has(job.id)) {
      attachWebSocket(job);
    }
  }
  // Refresh the audio-job dropdown for query.
  const sel = $("#query-job-id");
  const completedAudio = data.jobs.filter((j) => j.kind === "audio" && j.status === "completed");
  sel.innerHTML = "";
  for (const j of completedAudio) {
    sel.appendChild(el("option", { value: j.id },
      `${(j.input_path || "").split("/").pop()} (${j.id.slice(0, 8)})`));
  }
  if (completedAudio.length === 0) {
    sel.appendChild(el("option", { value: "", disabled: true }, "(no completed audio jobs)"));
  }
}

function attachWebSocket(job) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/api/jobs/${job.id}/progress`);
  liveSockets.set(job.id, ws);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.snapshot) {
        // Apply snapshot directly to the job row's metric cells without
        // a full re-render.
        const tr = document.querySelector(`tr[data-job-id="${job.id}"]`);
        if (tr) {
          const tds = tr.querySelectorAll("td");
          const snap = msg.snapshot;
          // [3]: progress, [4]: rt-factor, [5]: wall-clock, [6]: eta
          const fill = tds[3].querySelector(".progress-fill");
          if (fill) fill.style.width = `${snap.percent || 0}%`;
          tds[3].lastChild.textContent = ` ${(snap.percent || 0).toFixed(0)}%`;
          tds[4].textContent = fmtRTF(snap.rt_factor);
          tds[5].textContent = fmtDuration(snap.wall_clock_s);
          tds[6].textContent = fmtDuration(snap.eta_s);
        }
      }
      if (msg.final) {
        liveSockets.delete(job.id);
        ws.close();
        refreshJobs();
      }
    } catch (e) { console.error("ws parse", e); }
  };
  ws.onclose = () => liveSockets.delete(job.id);
  ws.onerror = () => { liveSockets.delete(job.id); };
}

// ---- query -----------------------------------------------------------------

function bindQuery() {
  const kindSel = $("#query-kind");
  const sync = () => {
    const isAudio = kindSel.value === "audio";
    $$(".audio-query-only").forEach((e) => e.hidden = !isAudio);
  };
  kindSel.addEventListener("change", sync); sync();

  $("#form-query").addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = $("#query-question").value.trim();
    if (!question) return;
    const ans = $("#query-answer");
    ans.textContent = "(thinking…)";
    try {
      let r;
      if (kindSel.value === "video") {
        r = await api("/api/query/video", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });
      } else {
        const jobId = $("#query-job-id").value;
        if (!jobId) { ans.textContent = "(pick a completed audio job first)"; return; }
        r = await api("/api/query/audio", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job_id: jobId, question }),
        });
      }
      ans.textContent = r.answer || "(empty answer)";
    } catch (err) {
      ans.textContent = `error: ${err.message}`;
    }
  });
}

// ---- boot ------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", () => {
  bindVssControls();
  bindSubmitForm();
  bindQuery();
  refreshStatus();
  refreshJobs();
  setInterval(refreshStatus, STATUS_POLL_MS);
  setInterval(refreshJobs, JOBS_POLL_MS);
});
