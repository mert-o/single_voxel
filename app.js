const META_PATHS = ["./data/metadata.json", "./metadata.json"];

const panelGrid = document.getElementById("panelGrid");
const panelTemplate = document.getElementById("panelTemplate");
const statusText = document.getElementById("statusText");
const pickedText = document.getElementById("pickedText");
const resetBtn = document.getElementById("resetBtn");
const alphaSlider = document.getElementById("alphaSlider");

let state = {
  panels: [],
  alpha: 1.0,
  selected: null,
  width: 0,
  height: 0,
  featureDim: 0,
  dataBaseDir: "./data",
};

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function colorMapViridis(t) {
  const x = clamp01(t);
  // Matplotlib-like viridis anchor points.
  const p = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const n = p.length - 1;
  const pos = x * n;
  const i = Math.floor(pos);
  const j = Math.min(n, i + 1);
  const f = pos - i;
  return [
    Math.round(p[i][0] * (1 - f) + p[j][0] * f),
    Math.round(p[i][1] * (1 - f) + p[j][1] * f),
    Math.round(p[i][2] * (1 - f) + p[j][2] * f),
  ];
}

function drawBaseT1(canvas, subj, marker = null) {
  const ctx = canvas.getContext("2d");
  const w = subj.width;
  const h = subj.height;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const g = subj.t1u8[i];
    img.data[i * 4 + 0] = g;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = g;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

  if (marker) {
    ctx.strokeStyle = "#111";
    ctx.fillStyle = "#ff3b1f";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(marker.x + 0.5, marker.y + 0.5, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function drawOverlay(canvas, subj, simValues, alpha) {
  const ctx = canvas.getContext("2d");
  const w = subj.width;
  const h = subj.height;
  const base = ctx.getImageData(0, 0, w, h);

  // Match cos_sim.py display clip=(0.0, 1.0) using viridis.
  const clipLo = 0.0;
  const clipHi = 1.0;
  const clipDen = clipHi - clipLo + 1e-8;

  for (let i = 0; i < subj.maskPixels.length; i++) {
    const flat = subj.maskPixels[i];
    const sim = simValues[i];
    const t = clamp01((sim - clipLo) / clipDen);
    const [r, g, b] = colorMapViridis(t);

    const a = alpha;
    const bi = flat * 4;
    base.data[bi + 0] = Math.round(base.data[bi + 0] * (1 - a) + r * a);
    base.data[bi + 1] = Math.round(base.data[bi + 1] * (1 - a) + g * a);
    base.data[bi + 2] = Math.round(base.data[bi + 2] * (1 - a) + b * a);
    base.data[bi + 3] = 255;
  }

  ctx.putImageData(base, 0, 0);
}

function cosineAll(subj, refVec) {
  const sims = new Float32Array(subj.maskPixels.length);

  for (let i = 0; i < subj.maskPixels.length; i++) {
    let dot = 0;
    const pix = subj.maskPixels[i];
    const base = pix * subj.k;
    for (let j = 0; j < subj.k; j++) {
      dot += subj.features[base + j] * refVec[j];
    }
    const den = subj.norms[pix] + 1e-8;
    sims[i] = dot / den;
  }

  return sims;
}

function buildPanels(subjects) {
  panelGrid.innerHTML = "";
  state.panels = [];

  subjects.forEach((subj, idx) => {
    const node = panelTemplate.content.firstElementChild.cloneNode(true);
    const canvas = node.querySelector("canvas");
    const caption = node.querySelector("figcaption");
    canvas.width = subj.width;
    canvas.height = subj.height;

    caption.textContent = subj.name;

    drawBaseT1(canvas, subj, null);

    canvas.style.cursor = "crosshair";
    canvas.addEventListener("click", (ev) => handlePick(ev, canvas, idx));

    panelGrid.appendChild(node);
    state.panels.push({ canvas, caption, subject: subj, idx });
  });
}

function showLoadingPanels(meta) {
  panelGrid.innerHTML = "";
  state.panels = [];

  for (const [idx, s] of meta.subjects.entries()) {
    const node = panelTemplate.content.firstElementChild.cloneNode(true);
    const canvas = node.querySelector("canvas");
    const caption = node.querySelector("figcaption");
    node.classList.add("is-loading");

    canvas.width = meta.width;
    canvas.height = meta.height;
    caption.textContent = `${s.name} (loading...)`;

    panelGrid.appendChild(node);
    state.panels.push({ canvas, caption, subject: null, idx });
  }
}

function handlePick(ev, canvas, subjectIndex) {
  const subj = state.panels[subjectIndex].subject;
  const rect = canvas.getBoundingClientRect();
  const px = Math.floor(((ev.clientX - rect.left) / rect.width) * subj.width);
  const py = Math.floor(((ev.clientY - rect.top) / rect.height) * subj.height);
  const flat = py * subj.width + px;

  if (subj.mask[flat] === 0) {
    statusText.textContent = "Pick inside brain mask";
    return;
  }

  renderFromSelection(subjectIndex, px, py);
}

function renderFromSelection(subjectIndex, px, py) {
  const subjRef = state.panels[subjectIndex].subject;
  const flat = py * subjRef.width + px;
  if (subjRef.mask[flat] === 0) {
    statusText.textContent = "Pick inside brain mask";
    return;
  }

  const base = flat * subjRef.k;
  const refVec = new Float32Array(subjRef.k);
  for (let j = 0; j < subjRef.k; j++) {
    refVec[j] = subjRef.features[base + j];
  }
  const refNorm = Math.sqrt(refVec.reduce((acc, v) => acc + v * v, 0));
  if (refNorm < 1e-8) {
    statusText.textContent = "Reference vector is empty at this voxel";
    return;
  }
  for (let j = 0; j < subjRef.k; j++) {
    refVec[j] /= refNorm;
  }

  state.selected = { subjectIndex, x: px, y: py };
  pickedText.textContent = `Selected voxel: subj=${subjectIndex + 1}, x=${px}, y=${py}`;
  statusText.textContent = "Computing cosine maps...";

  for (const panel of state.panels) {
    const isPicked = panel.idx === subjectIndex;
    drawBaseT1(panel.canvas, panel.subject, isPicked ? state.selected : null);
    const sims = cosineAll(panel.subject, refVec);
    drawOverlay(panel.canvas, panel.subject, sims, state.alpha);
  }

  statusText.textContent = "Done";
}

function resetView() {
  state.selected = null;
  pickedText.textContent = "No point selected.";
  statusText.textContent = "Reset";

  for (const panel of state.panels) {
    drawBaseT1(panel.canvas, panel.subject, null);
  }
}

function normalizeT1ToU8(t1f, mask) {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < t1f.length; i++) {
    if (!mask[i]) {
      continue;
    }
    const v = t1f[i];
    if (v < lo) {
      lo = v;
    }
    if (v > hi) {
      hi = v;
    }
  }
  const out = new Uint8Array(t1f.length);
  const den = hi - lo + 1e-8;
  for (let i = 0; i < t1f.length; i++) {
    const x = (t1f[i] - lo) / den;
    out[i] = Math.max(0, Math.min(255, Math.round(x * 255)));
  }
  return out;
}

async function loadArray(url, ctor, expectedLength) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Could not fetch ${url}`);
  }
  const buf = await res.arrayBuffer();
  const arr = new ctor(buf);
  if (arr.length !== expectedLength) {
    throw new Error(`Unexpected length for ${url}: got ${arr.length}, expected ${expectedLength}`);
  }
  return arr;
}

async function loadMetadata() {
  for (const path of META_PATHS) {
    try {
      const res = await fetch(path);
      if (!res.ok) {
        continue;
      }
      const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : ".";
      statusText.textContent = `Loaded ${path}`;
      return { meta: await res.json(), baseDir: dir || "." };
    } catch (err) {
      // Try next candidate.
    }
  }
  throw new Error("Could not load metadata.json");
}

async function loadSubjects(meta, baseDir) {
  const width = meta.width;
  const height = meta.height;
  const k = meta.feature_dim;
  const pixelCount = width * height;

  const subjects = [];
  for (const s of meta.subjects) {
    const featUrl = `${baseDir}/${s.feature_file}`;
    const t1Url = `${baseDir}/${s.t1_file}`;
    const maskUrl = `${baseDir}/${s.mask_file}`;

    const features = await loadArray(featUrl, Float32Array, pixelCount * k);
    const t1f = await loadArray(t1Url, Float32Array, pixelCount);
    const mask = await loadArray(maskUrl, Uint8Array, pixelCount);

    const norms = new Float32Array(pixelCount);
    const maskPixels = [];
    for (let pix = 0; pix < pixelCount; pix++) {
      if (mask[pix] > 0) {
        maskPixels.push(pix);
      }
      let n2 = 0;
      const base = pix * k;
      for (let j = 0; j < k; j++) {
        const v = features[base + j];
        n2 += v * v;
      }
      norms[pix] = Math.sqrt(n2);
    }

    subjects.push({
      name: s.name,
      width,
      height,
      k,
      features,
      t1f,
      t1u8: normalizeT1ToU8(t1f, mask),
      mask,
      maskPixels: Uint32Array.from(maskPixels),
      norms,
    });
  }

  return subjects;
}

async function init() {
  try {
    const { meta, baseDir } = await loadMetadata();
    state.width = meta.width;
    state.height = meta.height;
    state.featureDim = meta.feature_dim;
    state.dataBaseDir = baseDir;

    showLoadingPanels(meta);
    statusText.textContent = "Loading slices...";

    const subjects = await loadSubjects(meta, baseDir);
    buildPanels(subjects);
    statusText.textContent = "Click any subject panel";
  } catch (err) {
    statusText.textContent = "Failed to load data files";
    pickedText.textContent = String(err.message || err);
  }
}

resetBtn.addEventListener("click", resetView);
alphaSlider.addEventListener("input", (ev) => {
  state.alpha = Number(ev.target.value) / 100;
  if (state.selected) {
    renderFromSelection(state.selected.subjectIndex, state.selected.x, state.selected.y);
  }
});

alphaSlider.value = "100";

init();
