// ── Feeling Engine Inspector ──
// Loads an exported arc bundle from examples/arcs/ and renders:
//   • A brain-activity timeline (7 HCP categories over duration)
//   • A mechanism-firing timeline (labels × time)
//   • A scrubber + detail panel for the selected moment
//
// Zero build step, zero dependencies. Run via any static server from
// the repo's `examples/` directory:
//     cd feeling-engine/examples && python3 -m http.server 8080
// then open http://localhost:8080/inspector/

const AXES = ["interoception", "core_affect", "regulation", "reward", "memory", "social", "language"];
const AXIS_COLORS = {
  interoception: "#f97316",
  core_affect:   "#eab308",
  regulation:    "#22d3ee",
  reward:        "#a78bfa",
  memory:        "#6ee7b7",
  social:        "#fb7185",
  language:      "#60a5fa",
};

const state = {
  bundle: null,
  slug: null,
  mode: "sigma",       // "sigma" or "absolute"
  cursorT: 0,
  duration: 0,
  playing: false,
  speed: 1,
  lastFrameTime: 0,
  rafId: null,
  hemiAvailable: false,
  hemiUserPref: true,  // user wants hemispheres shown if available
};

async function loadBundle(slug) {
  const url = `../arcs/${slug}.json`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url}: ${r.status}`);
  return r.json();
}

function setMeta(bundle) {
  const src = bundle.source;
  const el = document.getElementById("meta");
  const secs = Math.round(src.duration_sec);
  const dur = `${Math.floor(secs / 60)}m${String(secs % 60).padStart(2, "0")}s`;
  const ytLink = src.url
    ? `<a class="yt-link" href="${escapeHtml(src.url)}" target="_blank" rel="noopener">Watch on YouTube ↗</a>`
    : "";
  el.innerHTML = `
    <div class="meta-title">${escapeHtml(src.title || "(untitled)")}</div>
    <div>
      <strong>${escapeHtml(src.channel_handle || src.channel || "")}</strong>
      · ${dur}
      · niche: ${escapeHtml(src.niche || "—")}
      · modality: ${escapeHtml(src.modality || "—")}
      ${src.outlier_ratio ? `· outlier ${src.outlier_ratio.toFixed(1)}×` : ""}
      ${ytLink ? " · " + ytLink : ""}
    </div>
    <div style="margin-top:6px; color:#7c8591">
      ${bundle.tribe_profiles.length} timesteps
      · abs: ${bundle.counts.n_labels_absolute} mechanisms, ${bundle.counts.n_sequences_absolute} sequences
      · σ: ${bundle.counts.n_labels_sigma} mechanisms, ${bundle.counts.n_sequences_sigma} sequences
    </div>
  `;
}

async function probeHemispheres(slug) {
  // HEAD the first expected PNG. If it 200s, hemispheres are available.
  const url = `../arcs/${slug}/brain/t00_left.png`;
  try {
    const r = await fetch(url, { method: "HEAD" });
    return r.ok;
  } catch {
    return false;
  }
}

function applyHemisphereVisibility() {
  const card = document.getElementById("hemi-card");
  const wrap = document.getElementById("hemi-wrap");
  const placeholder = document.getElementById("hemi-placeholder");

  if (!state.hemiUserPref) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");
  if (state.hemiAvailable) {
    wrap.style.display = "";
    placeholder.hidden = true;
  } else {
    wrap.style.display = "none";
    placeholder.hidden = false;
  }
}

function updateHemispheres(t) {
  if (!state.hemiAvailable || !state.hemiUserPref) return;
  const idx = Math.min(
    state.bundle.tribe_profiles.length - 1,
    Math.max(0, Math.round(t))
  );
  const pad = String(idx).padStart(2, "0");
  const base = `../arcs/${state.slug}/brain`;
  document.getElementById("hemi-left").src  = `${base}/t${pad}_left.png`;
  document.getElementById("hemi-right").src = `${base}/t${pad}_right.png`;
}

function renderBrainTimeline(bundle) {
  const svg = document.getElementById("brain-timeline");
  svg.innerHTML = "";
  const profiles = bundle.tribe_profiles;
  const N = profiles.length;
  if (!N) return;

  const W = svg.clientWidth || 900;
  const H = 220;
  const PADL = 10, PADR = 10, PADT = 10, PADB = 20;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  // Find global min/max across all axes for y-axis scale
  let gmin = Infinity, gmax = -Infinity;
  for (const p of profiles) {
    for (const a of AXES) {
      const v = p.categories[a];
      if (v < gmin) gmin = v;
      if (v > gmax) gmax = v;
    }
  }
  const ySpan = Math.max(0.01, gmax - gmin);
  const x = i => PADL + (i / (N - 1)) * (W - PADL - PADR);
  const y = v => PADT + (H - PADT - PADB) * (1 - (v - gmin) / ySpan);

  // zero line
  const zeroY = y(0);
  appendSvg(svg, "line", {
    x1: PADL, x2: W - PADR, y1: zeroY, y2: zeroY,
    stroke: "#2a3038", "stroke-dasharray": "2 3", "stroke-width": 1,
  });

  // One polyline per axis
  for (const a of AXES) {
    const pts = profiles.map((p, i) => `${x(i).toFixed(1)},${y(p.categories[a]).toFixed(1)}`).join(" ");
    appendSvg(svg, "polyline", {
      points: pts,
      fill: "none",
      stroke: AXIS_COLORS[a],
      "stroke-width": 1.2,
      opacity: 0.85,
    });
  }

  // Cursor line
  appendSvg(svg, "line", {
    id: "brain-cursor",
    x1: x(0), x2: x(0), y1: PADT, y2: H - PADB,
    stroke: "#fff", "stroke-width": 1.5, opacity: 0.7,
  });

  // Legend
  const legend = document.getElementById("brain-legend");
  legend.innerHTML = AXES.map(a => `
    <span>
      <span class="swatch" style="background:${AXIS_COLORS[a]}"></span>
      ${a.replace("_", " ")}
    </span>
  `).join("");
}

function renderMechanismTimeline(bundle) {
  const svg = document.getElementById("mech-timeline");
  svg.innerHTML = "";
  const arc = state.mode === "sigma" ? bundle.arc_sigma : bundle.arc_absolute;
  const N = bundle.tribe_profiles.length;
  if (!N) return;

  // Group by label, preserve order of first appearance
  const labelOrder = [];
  const byLabel = new Map();
  for (const app of arc) {
    if (!byLabel.has(app.label)) {
      labelOrder.push(app.label);
      byLabel.set(app.label, []);
    }
    byLabel.get(app.label).push(app);
  }

  const W = svg.clientWidth || 900;
  const rowH = 14;
  const H = Math.max(60, labelOrder.length * rowH + 20);
  const PADL = 120, PADR = 10, PADT = 6;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const x = sec => PADL + (sec / state.duration) * (W - PADL - PADR);

  const mechInfo = document.getElementById("mech-info");
  mechInfo.textContent = `${arc.length} firings · ${labelOrder.length} distinct labels · mode: ${state.mode}`;

  labelOrder.forEach((label, row) => {
    const cy = PADT + row * rowH + rowH / 2;
    // label column
    appendSvg(svg, "text", {
      x: PADL - 8, y: cy + 3,
      "text-anchor": "end",
      "font-size": 10.5,
      fill: "#a8b2be",
      "font-family": "monospace",
    }).textContent = label;

    // row divider
    appendSvg(svg, "line", {
      x1: PADL, x2: W - PADR, y1: cy + rowH / 2, y2: cy + rowH / 2,
      stroke: "#1a1f25", "stroke-width": 1,
    });

    for (const app of byLabel.get(label)) {
      const cx = x(app.start_sec);
      const size = 3 + Math.min(app.intensity * 3, 3);
      appendSvg(svg, "circle", {
        cx, cy, r: size,
        fill: "#4a9eff",
        opacity: 0.5 + Math.min(app.intensity * 0.5, 0.5),
      });
    }
  });

  // Cursor
  appendSvg(svg, "line", {
    id: "mech-cursor",
    x1: x(0), x2: x(0), y1: PADT, y2: H - 10,
    stroke: "#fff", "stroke-width": 1.5, opacity: 0.7,
  });
}

function updateCursor(t) {
  state.cursorT = t;
  // Brain cursor
  const brainSvg = document.getElementById("brain-timeline");
  const brainCursor = document.getElementById("brain-cursor");
  if (brainCursor) {
    const vb = brainSvg.viewBox.baseVal;
    const PADL = 10, PADR = 10;
    const N = state.bundle.tribe_profiles.length;
    const idx = Math.round((t / state.duration) * (N - 1));
    const cx = PADL + (idx / (N - 1)) * (vb.width - PADL - PADR);
    brainCursor.setAttribute("x1", cx);
    brainCursor.setAttribute("x2", cx);
  }
  // Mech cursor
  const mechSvg = document.getElementById("mech-timeline");
  const mechCursor = document.getElementById("mech-cursor");
  if (mechCursor) {
    const vb = mechSvg.viewBox.baseVal;
    const PADL = 120, PADR = 10;
    const cx = PADL + (t / state.duration) * (vb.width - PADL - PADR);
    mechCursor.setAttribute("x1", cx);
    mechCursor.setAttribute("x2", cx);
  }

  document.getElementById("scrubber-readout").textContent = `t=${t.toFixed(0)}s / ${Math.round(state.duration)}s`;
  updateDetail(t);
  updateHemispheres(t);
}

function updateDetail(t) {
  const bundle = state.bundle;
  // Find closest timestep
  const idx = Math.min(
    bundle.tribe_profiles.length - 1,
    Math.max(0, Math.round((t / state.duration) * (bundle.tribe_profiles.length - 1)))
  );
  const step = bundle.tribe_profiles[idx];

  // Brain state
  const brainUl = document.getElementById("brain-state");
  brainUl.innerHTML = AXES.map(a => {
    const v = step.categories[a];
    const cls = v > 0.01 ? "positive" : v < -0.01 ? "negative" : "";
    return `<li>
      <span class="axis" style="color:${AXIS_COLORS[a]}">${a.replace("_", " ")}</span>
      <span class="val ${cls}">${v >= 0 ? "+" : ""}${v.toFixed(3)}</span>
    </li>`;
  }).join("");

  // Mechanisms firing AT this second (start_sec == idx)
  const arc = state.mode === "sigma" ? bundle.arc_sigma : bundle.arc_absolute;
  const firing = arc.filter(app => Math.floor(app.start_sec) === idx);
  const firingUl = document.getElementById("firing-mechs");
  if (!firing.length) {
    firingUl.innerHTML = `<li style="color:#7c8591">(no mechanisms fire at this second)</li>`;
  } else {
    firing.sort((a, b) => b.intensity - a.intensity);
    firingUl.innerHTML = firing.map(app => `
      <li>
        <span class="label">${escapeHtml(app.label)}</span>
        <span class="tier">T${app.tier}</span>
        <span class="intensity">· intensity ${app.intensity.toFixed(2)}</span>
      </li>
    `).join("");
  }

  // Transcript word (if any)
  const tw = document.getElementById("transcript-word");
  const words = bundle.transcript?.words || [];
  const word = words.find(w => w.start <= t && t < (w.end || w.start + 0.5));
  if (word) {
    tw.textContent = `"${word.word}"`;
    tw.classList.remove("empty");
  } else {
    tw.textContent = "— no dialogue at this moment —";
    tw.classList.add("empty");
  }
}

async function selectArc(slug) {
  try {
    const bundle = await loadBundle(slug);
    state.bundle = bundle;
    state.slug = slug;
    state.duration = bundle.source.duration_sec;
    state.hemiAvailable = await probeHemispheres(slug);
    setMeta(bundle);
    applyHemisphereVisibility();
    renderBrainTimeline(bundle);
    renderMechanismTimeline(bundle);
    const scrubber = document.getElementById("scrubber");
    scrubber.max = Math.round(state.duration);
    scrubber.value = 0;
    updateCursor(0);
  } catch (err) {
    document.getElementById("meta").innerHTML =
      `<div style="color:#fca5a5">Error loading arc: ${escapeHtml(err.message)}<br>
      If you opened this file via file://, run a local server first:<br>
      <code>cd feeling-engine/examples && python3 -m http.server 8080</code><br>
      then open <code>http://localhost:8080/inspector/</code>.</div>`;
  }
}

// ── playback ──
function playTick(now) {
  if (!state.playing) return;
  const dt = (now - state.lastFrameTime) / 1000;  // seconds since last frame
  state.lastFrameTime = now;
  let nextT = state.cursorT + dt * state.speed;
  if (nextT >= state.duration) {
    nextT = state.duration;
    setPlaying(false);
  }
  const scrubber = document.getElementById("scrubber");
  scrubber.value = Math.round(nextT);
  updateCursor(nextT);
  if (state.playing) {
    state.rafId = requestAnimationFrame(playTick);
  }
}

function setPlaying(on) {
  state.playing = on;
  const btn = document.getElementById("play-btn");
  btn.textContent = on ? "❚❚" : "▶";
  btn.setAttribute("aria-label", on ? "Pause" : "Play");
  btn.classList.toggle("playing", on);
  if (on) {
    // If we're at the end, restart from 0
    if (state.cursorT >= state.duration - 0.1) {
      state.cursorT = 0;
      document.getElementById("scrubber").value = 0;
      updateCursor(0);
    }
    state.lastFrameTime = performance.now();
    state.rafId = requestAnimationFrame(playTick);
  } else if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
}

// ── wiring ──
document.getElementById("arc-select").addEventListener("change", e => {
  setPlaying(false);
  selectArc(e.target.value);
});
document.getElementById("mode-sigma").addEventListener("change", e => {
  state.mode = e.target.checked ? "sigma" : "absolute";
  if (state.bundle) {
    renderMechanismTimeline(state.bundle);
    updateCursor(state.cursorT);
  }
});
document.getElementById("scrubber").addEventListener("input", e => {
  // Manual scrub pauses playback
  if (state.playing) setPlaying(false);
  updateCursor(Number(e.target.value));
});
document.getElementById("play-btn").addEventListener("click", () => setPlaying(!state.playing));
document.getElementById("speed-select").addEventListener("change", e => {
  state.speed = Number(e.target.value);
});
document.getElementById("show-hemispheres").addEventListener("change", e => {
  state.hemiUserPref = e.target.checked;
  applyHemisphereVisibility();
  if (state.hemiUserPref) updateHemispheres(state.cursorT);
});
document.addEventListener("keydown", e => {
  // Spacebar toggles play/pause (ignore if user is typing in a form field)
  if (e.code === "Space" && e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") {
    e.preventDefault();
    setPlaying(!state.playing);
  }
});
window.addEventListener("resize", () => {
  if (state.bundle) {
    renderBrainTimeline(state.bundle);
    renderMechanismTimeline(state.bundle);
    updateCursor(state.cursorT);
  }
});

// ── helpers ──
function appendSvg(parent, tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  parent.appendChild(el);
  return el;
}
function escapeHtml(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

// ── init ──
selectArc(document.getElementById("arc-select").value);
