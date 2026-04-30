const META_PATHS = ["./data/metadata.json", "./metadata.json"];
const FETCH_OPTIONS = { cache: "no-store" };

const datasetTabs = document.getElementById("datasetTabs");
const datasetDescription = document.getElementById("datasetDescription");
const panelGrid = document.getElementById("panelGrid");
const panelTemplate = document.getElementById("panelTemplate");
const statusText = document.getElementById("statusText");
const pickedText = document.getElementById("pickedText");
const resetBtn = document.getElementById("resetBtn");
const alphaSlider = document.getElementById("alphaSlider");
const zoomSlider = document.getElementById("zoomSlider");
const zoomValue = document.getElementById("zoomValue");
const regionControls = document.getElementById("regionControls");
const similarityControls = document.getElementById("similarityControls");
const contrastSlider = document.getElementById("contrastSlider");
const contrastLabel = document.getElementById("contrastLabel");
const contrastPrevBtn = document.getElementById("contrastPrevBtn");
const contrastNextBtn = document.getElementById("contrastNextBtn");
const featureCount = document.getElementById("featureCount");
const activeFamiliesText = document.getElementById("activeFamiliesText");
const selectionSummary = document.getElementById("selectionSummary");
const healthyPrepText = document.getElementById("healthyPrepText");
const tumorPrepText = document.getElementById("tumorPrepText");
const deliveryText = document.getElementById("deliveryText");

const state = {
  meta: null,
  baseDir: "./data",
  datasetCache: new Map(),
  panels: [],
  datasetId: null,
  alpha: 0.86,
  zoom: 1,
  activeRegionId: null,
  activeSimilarityGroups: new Set(),
  displayIndex: 0,
  selected: null,
  similarityCache: null,
};

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function clamp01(x) {
  return clamp(x, 0, 1);
}

function subjectLabel(index) {
  return `Subject-${index + 1}`;
}

function colorMapViridis(t) {
  const x = clamp01(t);
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

function percentile(sortedValues, q) {
  if (!sortedValues.length) {
    return NaN;
  }
  const pos = clamp01(q) * (sortedValues.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const frac = pos - lo;
  return sortedValues[lo] * (1 - frac) + sortedValues[hi] * frac;
}

function normalizeImageToU8(values, mask, options = {}) {
  const backgroundGray = options.backgroundGray ?? 16;
  const lowerQ = options.lowerQ ?? 0.01;
  const upperQ = options.upperQ ?? 0.99;

  const selected = [];
  for (let i = 0; i < values.length; i++) {
    if (mask[i]) {
      selected.push(values[i]);
    }
  }

  const out = new Uint8Array(values.length);
  if (!selected.length) {
    out.fill(backgroundGray);
    return out;
  }

  selected.sort((a, b) => a - b);
  let lo = percentile(selected, lowerQ);
  let hi = percentile(selected, upperQ);

  if (!(hi > lo)) {
    lo = selected[0];
    hi = selected[selected.length - 1];
  }

  if (!(hi > lo)) {
    out.fill(backgroundGray);
    for (let i = 0; i < values.length; i++) {
      if (mask[i]) {
        out[i] = 128;
      }
    }
    return out;
  }

  const den = hi - lo + 1e-8;
  for (let i = 0; i < values.length; i++) {
    if (!mask[i]) {
      out[i] = backgroundGray;
      continue;
    }
    out[i] = Math.max(0, Math.min(255, Math.round(clamp01((values[i] - lo) / den) * 255)));
  }
  return out;
}

function normalizeGroupList(groups, fallbackFeatureDim) {
  if (!groups || !groups.length) {
    return [
      {
        id: "all",
        label: "All features",
        ranges: [[0, fallbackFeatureDim]],
      },
    ];
  }

  if (!Array.isArray(groups)) {
    return Object.entries(groups).map(([label, [start, end]]) => ({
      id: String(label).toLowerCase().replace(/[^a-z0-9]+/g, "_"),
      label: String(label),
      ranges: [[Number(start), Number(end)]],
    }));
  }

  return groups.map((group) => ({
    id: String(group.id ?? group.label),
    label: String(group.label ?? group.id),
    ranges: (group.ranges ?? [[group.start, group.end]]).map(([a, b]) => [Number(a), Number(b)]),
  }));
}

function normalizeDtypes(dtypeConfig = {}) {
  return {
    feature: String(dtypeConfig.feature ?? "float32").toLowerCase(),
    t1: String(dtypeConfig.t1 ?? "float32").toLowerCase(),
    mask: String(dtypeConfig.mask ?? "uint8").toLowerCase(),
    region: String(dtypeConfig.region ?? "uint8").toLowerCase(),
  };
}

function float16ToFloat32(value) {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return sign * 0;
    }
    return sign * Math.pow(2, -14) * (fraction / 1024);
  }

  if (exponent === 0x1f) {
    return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  }

  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

const FLOAT16_LUT = (() => {
  const table = new Float32Array(65536);
  for (let i = 0; i < table.length; i++) {
    table[i] = float16ToFloat32(i);
  }
  return table;
})();

function decodeFloat16Buffer(buffer, expectedLength, url) {
  const arr = new Uint16Array(buffer);
  if (arr.length !== expectedLength) {
    throw new Error(`Unexpected length for ${url}: got ${arr.length}, expected ${expectedLength}`);
  }
  const out = new Float32Array(expectedLength);
  for (let i = 0; i < arr.length; i++) {
    out[i] = FLOAT16_LUT[arr[i]];
  }
  return out;
}

function wrapLegacyMetadata(meta) {
  return {
    version: 1,
    default_dataset_id: "legacy",
    datasets: [
      {
        id: "legacy",
        name: "Voxel Similarity Demo",
        description: "Legacy single-dataset export",
        width: meta.width,
        height: meta.height,
        feature_dim: meta.feature_dim,
        t1_label: "T1 structural",
        default_display: { kind: "t1" },
        display_t1_at_feature: null,
        feature_groups: [{ id: "all", label: "All features", start: 0, end: meta.feature_dim }],
        similarity_groups: [{ id: "all", label: "All features", start: 0, end: meta.feature_dim }],
        regions: [],
        subjects: meta.subjects.map((subject) => ({ ...subject, region_files: {} })),
      },
    ],
  };
}

function formatFeatureLabel(featureGroups, index) {
  for (const group of featureGroups) {
    for (const [a, b] of group.ranges) {
      if (index >= a && index < b) {
        return `${group.label} · feature dim ${index}`;
      }
    }
  }
  return `Feature dim ${index}`;
}

function buildDisplayOptions(datasetMeta) {
  const options = [];
  const anchorIndex =
    Number.isInteger(datasetMeta.displayT1AtFeature) &&
    datasetMeta.displayT1AtFeature >= 0 &&
    datasetMeta.displayT1AtFeature < datasetMeta.featureDim
      ? datasetMeta.displayT1AtFeature
      : null;

  if (anchorIndex === null) {
    options.push({
      id: "t1",
      kind: "t1",
      label: datasetMeta.t1Label || "T1 structural",
    });
  }

  for (let i = 0; i < datasetMeta.featureDim; i++) {
    if (anchorIndex !== null && i === anchorIndex) {
      options.push({
        id: "t1",
        kind: "t1",
        label: datasetMeta.t1Label || "T1 structural",
      });
      continue;
    }
    options.push({
      id: `f${i}`,
      kind: "feature",
      index: i,
      label: formatFeatureLabel(datasetMeta.featureGroups, i),
    });
  }

  return options;
}

function normalizeMetadata(rawMeta) {
  const meta = Array.isArray(rawMeta.datasets) ? rawMeta : wrapLegacyMetadata(rawMeta);
  const datasets = meta.datasets.map((dataset) => {
    const featureDim = Number(dataset.feature_dim ?? dataset.featureDim);
    const featureGroups = normalizeGroupList(dataset.feature_groups ?? dataset.featureGroups, featureDim);
    const similarityGroups = normalizeGroupList(
      dataset.similarity_groups ?? dataset.similarityGroups ?? featureGroups,
      featureDim
    );
    const hydrated = {
      id: String(dataset.id),
      name: String(dataset.name),
      description: String(dataset.description ?? ""),
      width: Number(dataset.width),
      height: Number(dataset.height),
      featureDim,
      dtype: normalizeDtypes(dataset.dtype),
      t1Label: String(dataset.t1_label ?? dataset.t1Label ?? "T1 structural"),
      preprocessingSummary: String(dataset.preprocessing_summary ?? dataset.preprocessingSummary ?? ""),
      deliverySummary: String(dataset.delivery_summary ?? dataset.deliverySummary ?? ""),
      displayT1AtFeature: Number.isFinite(dataset.display_t1_at_feature ?? dataset.displayT1AtFeature)
        ? Number(dataset.display_t1_at_feature ?? dataset.displayT1AtFeature)
        : null,
      defaultDisplay: dataset.default_display ?? dataset.defaultDisplay ?? { kind: "t1" },
      featureGroups,
      similarityGroups,
      regions: (dataset.regions ?? []).map((region) => ({
        id: String(region.id),
        label: String(region.label ?? region.id),
      })),
      subjects: dataset.subjects ?? [],
    };
    hydrated.displayOptions = buildDisplayOptions(hydrated);
    return hydrated;
  });

  return {
    version: meta.version ?? 1,
    defaultDatasetId: meta.default_dataset_id ?? datasets[0]?.id,
    datasets,
  };
}

function findDisplayIndex(datasetMeta, displaySpec) {
  if (!displaySpec || displaySpec.kind === "t1") {
    const idx = datasetMeta.displayOptions.findIndex((option) => option.kind === "t1");
    return idx >= 0 ? idx : 0;
  }
  if (displaySpec.kind === "feature") {
    const idx = datasetMeta.displayOptions.findIndex(
      (option) => option.kind === "feature" && option.index === Number(displaySpec.index)
    );
    return idx >= 0 ? idx : 0;
  }
  return 0;
}

function buildMaskInfo(mask, width, height) {
  const pixels = [];
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let flat = 0; flat < mask.length; flat++) {
    if (!mask[flat]) {
      continue;
    }
    pixels.push(flat);
    const x = flat % width;
    const y = Math.floor(flat / width);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  return {
    mask,
    pixels: Uint32Array.from(pixels),
    bbox:
      pixels.length > 0
        ? {
            minX,
            minY,
            maxX,
            maxY,
            width: maxX - minX + 1,
            height: maxY - minY + 1,
          }
        : null,
  };
}

async function loadTypedArray(url, dtype, expectedLength) {
  const res = await fetch(url, FETCH_OPTIONS);
  if (!res.ok) {
    throw new Error(`Could not fetch ${url}`);
  }
  const buf = await res.arrayBuffer();
  if (dtype === "float32") {
    const arr = new Float32Array(buf);
    if (arr.length !== expectedLength) {
      throw new Error(`Unexpected length for ${url}: got ${arr.length}, expected ${expectedLength}`);
    }
    return arr;
  }
  if (dtype === "float16") {
    return decodeFloat16Buffer(buf, expectedLength, url);
  }
  if (dtype === "uint8") {
    const arr = new Uint8Array(buf);
    if (arr.length !== expectedLength) {
      throw new Error(`Unexpected length for ${url}: got ${arr.length}, expected ${expectedLength}`);
    }
    return arr;
  }
  throw new Error(`Unsupported dtype '${dtype}' for ${url}`);
}

async function loadMetadata() {
  for (const path of META_PATHS) {
    try {
      const res = await fetch(path, FETCH_OPTIONS);
      if (!res.ok) {
        continue;
      }
      const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : ".";
      return { meta: normalizeMetadata(await res.json()), baseDir: dir || "." };
    } catch (err) {
      // Try the next candidate.
    }
  }
  throw new Error("Could not load metadata.json");
}

async function loadSubjectPreview(datasetMeta, subjectMeta, baseDir) {
  const pixelCount = datasetMeta.width * datasetMeta.height;
  const regionEntries = Object.entries(subjectMeta.region_files ?? {});
  const regionPromises = regionEntries.map(async ([regionId, filename]) => [
    regionId,
    await loadTypedArray(`${baseDir}/${filename}`, datasetMeta.dtype.region, pixelCount),
  ]);

  const [t1f, mask, ...regionPairs] = await Promise.all([
    loadTypedArray(`${baseDir}/${subjectMeta.t1_file}`, datasetMeta.dtype.t1, pixelCount),
    loadTypedArray(`${baseDir}/${subjectMeta.mask_file}`, datasetMeta.dtype.mask, pixelCount),
    ...regionPromises,
  ]);

  const subject = {
    name: subjectMeta.name,
    slice: subjectMeta.slice,
    width: datasetMeta.width,
    height: datasetMeta.height,
    k: datasetMeta.featureDim,
    pixelCount,
    t1f,
    mask,
    maskInfo: buildMaskInfo(mask, datasetMeta.width, datasetMeta.height),
    emptyMaskInfo: buildMaskInfo(new Uint8Array(pixelCount), datasetMeta.width, datasetMeta.height),
    displayCache: new Map(),
    regions: {},
    features: null,
    norms: null,
    featuresReady: false,
  };

  subject.displayCache.set("t1", normalizeImageToU8(t1f, mask, { backgroundGray: 14, lowerQ: 0.01, upperQ: 0.995 }));
  for (const [regionId, regionMask] of regionPairs) {
    subject.regions[regionId] = buildMaskInfo(regionMask, datasetMeta.width, datasetMeta.height);
  }

  return subject;
}

async function createDataset(datasetMeta, baseDir) {
  const subjects = [];
  for (let i = 0; i < datasetMeta.subjects.length; i++) {
    statusText.textContent = `Loading preview slices (${i + 1}/${datasetMeta.subjects.length})`;
    subjects.push(await loadSubjectPreview(datasetMeta, datasetMeta.subjects[i], baseDir));
  }

  return {
    meta: datasetMeta,
    subjects,
    featuresReadyCount: 0,
    featuresTotal: datasetMeta.subjects.length,
    featureLoadPromise: null,
    featureLoadError: null,
  };
}

function getActiveDataset() {
  return state.datasetCache.get(state.datasetId);
}

function getDisplayOption(datasetMeta) {
  return datasetMeta.displayOptions[state.displayIndex] ?? datasetMeta.displayOptions[0];
}

function getSubjectDisplayImage(subject, displayOption) {
  if (displayOption.kind === "t1" || !subject.featuresReady) {
    return subject.displayCache.get("t1");
  }

  const key = `f${displayOption.index}`;
  if (subject.displayCache.has(key)) {
    return subject.displayCache.get(key);
  }

  const values = new Float32Array(subject.pixelCount);
  for (let flat = 0; flat < subject.pixelCount; flat++) {
    values[flat] = subject.features[flat * subject.k + displayOption.index];
  }
  const u8 = normalizeImageToU8(values, subject.mask, { backgroundGray: 12, lowerQ: 0.02, upperQ: 0.98 });
  subject.displayCache.set(key, u8);
  return u8;
}

function getRegionLabel(dataset, regionId) {
  if (!regionId) {
    return "Whole Brain";
  }
  const region = dataset.meta.regions.find((item) => item.id === regionId);
  return region?.label ?? regionId;
}

function getActiveMaskInfo(subject) {
  if (!state.activeRegionId) {
    return subject.maskInfo;
  }
  return subject.regions[state.activeRegionId] ?? subject.emptyMaskInfo;
}

function getViewportCenter(subject, regionInfo, viewport) {
  if (state.selected && state.selected.datasetId === state.datasetId) {
    return {
      x: clamp(state.selected.x + 0.5, viewport.x, viewport.x + viewport.width),
      y: clamp(state.selected.y + 0.5, viewport.y, viewport.y + viewport.height),
    };
  }

  if (regionInfo?.bbox) {
    return {
      x: regionInfo.bbox.minX + regionInfo.bbox.width / 2,
      y: regionInfo.bbox.minY + regionInfo.bbox.height / 2,
    };
  }

  return {
    x: subject.width / 2,
    y: subject.height / 2,
  };
}

function computeViewport(subject) {
  const regionInfo = getActiveMaskInfo(subject);
  let viewport;

  if (!state.activeRegionId || !regionInfo.bbox) {
    viewport = { x: 0, y: 0, width: subject.width, height: subject.height };
  } else {
    const pad = Math.max(12, Math.round(Math.max(regionInfo.bbox.width, regionInfo.bbox.height) * 0.42));
    const x0 = clamp(regionInfo.bbox.minX - pad, 0, subject.width - 1);
    const y0 = clamp(regionInfo.bbox.minY - pad, 0, subject.height - 1);
    const x1 = clamp(regionInfo.bbox.maxX + pad, 0, subject.width - 1);
    const y1 = clamp(regionInfo.bbox.maxY + pad, 0, subject.height - 1);
    viewport = {
      x: x0,
      y: y0,
      width: x1 - x0 + 1,
      height: y1 - y0 + 1,
    };
  }

  if (state.zoom <= 1.01) {
    return viewport;
  }

  const center = getViewportCenter(subject, regionInfo, viewport);
  const zoomWidth = clamp(Math.round(viewport.width / state.zoom), 24, viewport.width);
  const zoomHeight = clamp(Math.round(viewport.height / state.zoom), 24, viewport.height);
  return {
    x: clamp(Math.round(center.x - zoomWidth / 2), viewport.x, viewport.x + viewport.width - zoomWidth),
    y: clamp(Math.round(center.y - zoomHeight / 2), viewport.y, viewport.y + viewport.height - zoomHeight),
    width: zoomWidth,
    height: zoomHeight,
  };
}

function containRect(srcWidth, srcHeight, dstWidth, dstHeight) {
  const scale = Math.min(dstWidth / srcWidth, dstHeight / srcHeight);
  const width = Math.max(1, Math.floor(srcWidth * scale));
  const height = Math.max(1, Math.floor(srcHeight * scale));
  return {
    x: Math.floor((dstWidth - width) / 2),
    y: Math.floor((dstHeight - height) / 2),
    width,
    height,
  };
}

function drawMarker(ctx, x, y) {
  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.96)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(x, y, 3.1, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = "#ff5b4d";
  ctx.lineWidth = 1.0;
  ctx.beginPath();
  ctx.arc(x, y, 1.8, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "#ff3b30";
  ctx.beginPath();
  ctx.arc(x, y, 0.75, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawRegionOutline(ctx, regionInfo, width) {
  if (!regionInfo?.pixels?.length) {
    return;
  }

  ctx.save();
  ctx.fillStyle = "rgba(255, 195, 94, 0.95)";
  for (let i = 0; i < regionInfo.pixels.length; i++) {
    const flat = regionInfo.pixels[i];
    const x = flat % width;
    const y = Math.floor(flat / width);
    const left = x === 0 ? 0 : regionInfo.mask[flat - 1];
    const right = x === width - 1 ? 0 : regionInfo.mask[flat + 1];
    const top = y === 0 ? 0 : regionInfo.mask[flat - width];
    const bottom = y === Math.floor(regionInfo.mask.length / width) - 1 ? 0 : regionInfo.mask[flat + width];
    if (!left || !right || !top || !bottom) {
      ctx.fillRect(x, y, 1, 1);
    }
  }
  ctx.restore();
}

function drawOverlay(ctx, pixels, sims, alpha) {
  if (!pixels.length || !sims.length) {
    return;
  }

  const image = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  for (let i = 0; i < pixels.length; i++) {
    const flat = pixels[i];
    const t = clamp01(sims[i]);
    const [r, g, b] = colorMapViridis(t);
    const base = flat * 4;
    image.data[base + 0] = Math.round(image.data[base + 0] * (1 - alpha) + r * alpha);
    image.data[base + 1] = Math.round(image.data[base + 1] * (1 - alpha) + g * alpha);
    image.data[base + 2] = Math.round(image.data[base + 2] * (1 - alpha) + b * alpha);
    image.data[base + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function getActiveGroupIds() {
  return Array.from(state.activeSimilarityGroups).sort();
}

function activeGroupKey() {
  return getActiveGroupIds().join("|");
}

function getActiveChannelIndices(dataset) {
  const indices = [];
  for (const group of dataset.meta.similarityGroups) {
    if (!state.activeSimilarityGroups.has(group.id)) {
      continue;
    }
    for (const [a, b] of group.ranges) {
      for (let i = a; i < b; i++) {
        indices.push(i);
      }
    }
  }
  return Uint16Array.from(indices);
}

function buildReferenceVector(subject, flat, channelIndices) {
  const ref = new Float32Array(channelIndices.length);
  let norm2 = 0;
  const base = flat * subject.k;
  for (let i = 0; i < channelIndices.length; i++) {
    const value = subject.features[base + channelIndices[i]];
    ref[i] = value;
    norm2 += value * value;
  }
  const norm = Math.sqrt(norm2);
  if (norm < 1e-8) {
    return null;
  }
  for (let i = 0; i < ref.length; i++) {
    ref[i] /= norm;
  }
  return ref;
}

function cosineAll(subject, refVec, channelIndices, pixelInfo) {
  const sims = new Float32Array(pixelInfo.pixels.length);
  for (let i = 0; i < pixelInfo.pixels.length; i++) {
    const flat = pixelInfo.pixels[i];
    const base = flat * subject.k;
    let dot = 0;
    let norm2 = 0;
    for (let j = 0; j < channelIndices.length; j++) {
      const value = subject.features[base + channelIndices[j]];
      dot += value * refVec[j];
      norm2 += value * value;
    }
    sims[i] = dot / (Math.sqrt(norm2) + 1e-8);
  }
  return sims;
}

function summarizeSimilarity(sims) {
  if (!sims.length) {
    return null;
  }
  let sum = 0;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < sims.length; i++) {
    sum += sims[i];
    if (sims[i] > max) {
      max = sims[i];
    }
  }
  return {
    mean: sum / sims.length,
    max,
  };
}

function invalidateSimilarityCache() {
  state.similarityCache = null;
}

function featureProgressText(dataset) {
  return `${dataset.featuresReadyCount}/${dataset.featuresTotal}`;
}

function loadingStatusText(dataset, prefix = "Loading feature data") {
  return `${prefix} ${featureProgressText(dataset)}. Pick after load.`;
}

function previewReadyStatusText(dataset) {
  return `Preview ready. Features ${featureProgressText(dataset)}.`;
}

function ensureSimilarityCache() {
  const dataset = getActiveDataset();
  if (!dataset || !state.selected || state.selected.datasetId !== state.datasetId) {
    return null;
  }

  if (dataset.featuresReadyCount < dataset.featuresTotal) {
    statusText.textContent = loadingStatusText(dataset);
    return null;
  }

  const cacheKey = {
    datasetId: state.datasetId,
    subjectIndex: state.selected.subjectIndex,
    x: state.selected.x,
    y: state.selected.y,
    regionId: state.activeRegionId,
    groupKey: activeGroupKey(),
  };

  if (
    state.similarityCache &&
    state.similarityCache.datasetId === cacheKey.datasetId &&
    state.similarityCache.subjectIndex === cacheKey.subjectIndex &&
    state.similarityCache.x === cacheKey.x &&
    state.similarityCache.y === cacheKey.y &&
    state.similarityCache.regionId === cacheKey.regionId &&
    state.similarityCache.groupKey === cacheKey.groupKey
  ) {
    return state.similarityCache;
  }

  const channelIndices = getActiveChannelIndices(dataset);
  if (!channelIndices.length) {
    statusText.textContent = "Enable at least one contrast family.";
    return null;
  }

  const refSubject = dataset.subjects[state.selected.subjectIndex];
  const refFlat = state.selected.y * refSubject.width + state.selected.x;
  const activeRefMask = getActiveMaskInfo(refSubject);
  if (!activeRefMask.mask[refFlat]) {
    statusText.textContent = `Pick inside ${getRegionLabel(dataset, state.activeRegionId)}.`;
    return null;
  }

  const refVec = buildReferenceVector(refSubject, refFlat, channelIndices);
  if (!refVec) {
    statusText.textContent = "Reference vector is empty at this voxel.";
    return null;
  }

  const perPanel = dataset.subjects.map((subject) => {
    const pixelInfo = getActiveMaskInfo(subject);
    if (!pixelInfo.pixels.length) {
      return { pixelInfo, sims: new Float32Array(0), stats: null };
    }
    const sims = cosineAll(subject, refVec, channelIndices, pixelInfo);
    return {
      pixelInfo,
      sims,
      stats: summarizeSimilarity(sims),
    };
  });

  state.similarityCache = { ...cacheKey, perPanel };
  statusText.textContent = "Homology maps ready.";
  return state.similarityCache;
}

function renderPanel(panel, similarityEntry) {
  const dataset = getActiveDataset();
  const subject = panel.subject;
  const displayOption = getDisplayOption(dataset.meta);
  const visible = getSubjectDisplayImage(subject, displayOption);
  const focusInfo = getActiveMaskInfo(subject);
  const isRegionMode = Boolean(state.activeRegionId);
  const isReference =
    state.selected &&
    state.selected.datasetId === state.datasetId &&
    state.selected.subjectIndex === panel.idx;

  const bufferCtx = panel.bufferCanvas.getContext("2d");
  const image = bufferCtx.createImageData(subject.width, subject.height);
  for (let i = 0; i < subject.pixelCount; i++) {
    let gray = visible[i];
    if (isRegionMode && !focusInfo.mask[i]) {
      gray = Math.round(10 + gray * 0.16);
    }
    const base = i * 4;
    image.data[base + 0] = gray;
    image.data[base + 1] = gray;
    image.data[base + 2] = gray;
    image.data[base + 3] = 255;
  }
  bufferCtx.putImageData(image, 0, 0);

  if (similarityEntry?.sims?.length) {
    drawOverlay(bufferCtx, similarityEntry.pixelInfo.pixels, similarityEntry.sims, state.alpha);
  }
  if (isRegionMode && focusInfo.pixels.length) {
    drawRegionOutline(bufferCtx, focusInfo, subject.width);
  }

  const viewport = computeViewport(subject);
  const fit = containRect(viewport.width, viewport.height, panel.canvas.width, panel.canvas.height);
  const ctx = panel.canvas.getContext("2d");
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#10151f";
  ctx.fillRect(0, 0, panel.canvas.width, panel.canvas.height);
  ctx.drawImage(
    panel.bufferCanvas,
    viewport.x,
    viewport.y,
    viewport.width,
    viewport.height,
    fit.x,
    fit.y,
    fit.width,
    fit.height
  );

  if (isReference) {
    const markerX = fit.x + ((state.selected.x + 0.5 - viewport.x) / viewport.width) * fit.width;
    const markerY = fit.y + ((state.selected.y + 0.5 - viewport.y) / viewport.height) * fit.height;
    drawMarker(ctx, markerX, markerY);
  }
  ctx.restore();

  panel.viewState = { viewport, drawRect: fit };
  panel.title.textContent = subjectLabel(panel.idx);
  panel.meta.textContent = `Axial slice z=${subject.slice}`;

  if (!subject.featuresReady) {
    panel.badge.textContent = `Loading ${featureProgressText(dataset)}`;
  } else if (displayOption.kind === "feature" && !subject.displayCache.has(`f${displayOption.index}`)) {
    panel.badge.textContent = "Preparing";
  } else if (isReference && similarityEntry?.stats) {
    panel.badge.textContent = `Ref · ${similarityEntry.stats.mean.toFixed(2)}`;
  } else if (isReference) {
    panel.badge.textContent = "Ref";
  } else if (isRegionMode && !focusInfo.pixels.length) {
    panel.badge.textContent = "No target";
  } else if (similarityEntry?.stats) {
    panel.badge.textContent = `Mean ${similarityEntry.stats.mean.toFixed(2)}`;
  } else if (displayOption.kind === "feature" && subject.featuresReady) {
    panel.badge.textContent = "Contrast view";
  } else {
    panel.badge.textContent = "Preview";
  }

  panel.node.classList.remove("is-loading");
  panel.node.classList.toggle("is-reference", Boolean(isReference));
}

function renderPanels() {
  const dataset = getActiveDataset();
  if (!dataset) {
    return;
  }

  const similarityCache = ensureSimilarityCache();
  for (const panel of state.panels) {
    const similarityEntry = similarityCache?.perPanel?.[panel.idx] ?? null;
    renderPanel(panel, similarityEntry);
  }
  updateSelectionSummary();
}

function showLoadingPanels(datasetMeta) {
  panelGrid.innerHTML = "";
  state.panels = [];

  datasetMeta.subjects.forEach((subject, idx) => {
    const node = panelTemplate.content.firstElementChild.cloneNode(true);
    const canvas = node.querySelector("canvas");
    const title = node.querySelector(".panel-title");
    const meta = node.querySelector(".panel-meta");
    const badge = node.querySelector(".panel-badge");
    node.classList.add("is-loading");
    canvas.width = datasetMeta.width;
    canvas.height = datasetMeta.height;
    title.textContent = subjectLabel(idx);
    meta.textContent = "Loading preview slice...";
    badge.textContent = "Preview loading";
    panelGrid.appendChild(node);
    state.panels.push({
      node,
      canvas,
      title,
      meta,
      badge,
      bufferCanvas: null,
      subject: null,
      idx,
      viewState: null,
    });
  });
}

function buildPanels(dataset) {
  panelGrid.innerHTML = "";
  state.panels = [];

  dataset.subjects.forEach((subject, idx) => {
    const node = panelTemplate.content.firstElementChild.cloneNode(true);
    const canvas = node.querySelector("canvas");
    const title = node.querySelector(".panel-title");
    const meta = node.querySelector(".panel-meta");
    const badge = node.querySelector(".panel-badge");
    const bufferCanvas = document.createElement("canvas");
    bufferCanvas.width = subject.width;
    bufferCanvas.height = subject.height;
    canvas.width = subject.width;
    canvas.height = subject.height;
    title.textContent = subjectLabel(idx);
    meta.textContent = `Axial slice z=${subject.slice}`;
    badge.textContent = "Preview";
    canvas.style.cursor = "crosshair";

    const panel = {
      node,
      canvas,
      title,
      meta,
      badge,
      bufferCanvas,
      subject,
      idx,
      viewState: null,
    };

    canvas.addEventListener("click", (event) => handlePick(event, panel));
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        setZoom(state.zoom + (event.deltaY < 0 ? 0.15 : -0.15));
      },
      { passive: false }
    );

    panelGrid.appendChild(node);
    state.panels.push(panel);
  });

  renderPanels();
}

function getCanvasPixel(event, panel) {
  const rect = panel.canvas.getBoundingClientRect();
  const cx = ((event.clientX - rect.left) / rect.width) * panel.canvas.width;
  const cy = ((event.clientY - rect.top) / rect.height) * panel.canvas.height;
  const draw = panel.viewState?.drawRect;
  const viewport = panel.viewState?.viewport;
  if (!draw || !viewport) {
    return null;
  }
  if (cx < draw.x || cy < draw.y || cx > draw.x + draw.width || cy > draw.y + draw.height) {
    return null;
  }
  const rx = clamp01((cx - draw.x) / draw.width);
  const ry = clamp01((cy - draw.y) / draw.height);
  return {
    x: clamp(Math.floor(viewport.x + rx * viewport.width), 0, panel.subject.width - 1),
    y: clamp(Math.floor(viewport.y + ry * viewport.height), 0, panel.subject.height - 1),
  };
}

function clearSelection(statusOverride = null) {
  state.selected = null;
  invalidateSimilarityCache();
  pickedText.textContent = "No reference voxel selected. Click a slice to project voxel homology across subjects.";
  if (statusOverride) {
    statusText.textContent = statusOverride;
  }
  renderPanels();
}

function handlePick(event, panel) {
  const dataset = getActiveDataset();
  if (!dataset) {
    return;
  }
  if (dataset.featuresReadyCount < dataset.featuresTotal) {
    statusText.textContent = loadingStatusText(dataset, "Feature data still loading");
    return;
  }

  const coords = getCanvasPixel(event, panel);
  if (!coords) {
    return;
  }

  const flat = coords.y * panel.subject.width + coords.x;
  const focusInfo = getActiveMaskInfo(panel.subject);
  if (!focusInfo.mask[flat]) {
    statusText.textContent = state.activeRegionId
      ? `Pick inside ${getRegionLabel(dataset, state.activeRegionId)}.`
      : "Pick inside brain mask.";
    return;
  }

  state.selected = {
    datasetId: state.datasetId,
    subjectIndex: panel.idx,
    x: coords.x,
    y: coords.y,
  };
  invalidateSimilarityCache();
  pickedText.textContent = `${subjectLabel(panel.idx)} · x=${coords.x}, y=${coords.y} · ${getRegionLabel(
    dataset,
    state.activeRegionId
  )}`;
  statusText.textContent = "Computing homology maps...";
  renderPanels();
}

function updateSelectionSummary() {
  const dataset = getActiveDataset();
  if (!dataset) {
    selectionSummary.textContent = "";
    return;
  }

  const parts = [
    dataset.meta.name,
    getRegionLabel(dataset, state.activeRegionId),
    `${state.activeSimilarityGroups.size}/${dataset.meta.similarityGroups.length} families`,
  ];
  if (dataset.featuresReadyCount < dataset.featuresTotal) {
    parts.push(`Loading ${featureProgressText(dataset)}`);
  }
  if (state.selected && state.selected.datasetId === state.datasetId) {
    parts.push(`Ref: ${subjectLabel(state.selected.subjectIndex)}`);
  }
  selectionSummary.textContent = parts.join(" · ");
}

function updateStats() {
  const dataset = getActiveDataset();
  if (!dataset || !state.meta) {
    return;
  }

  const healthyMeta = state.meta.datasets.find((item) => item.id === "healthy") ?? null;
  const tumorMeta = state.meta.datasets.find((item) => item.id === "tumor") ?? null;
  const deliverySummary =
    dataset.meta.deliverySummary ||
    healthyMeta?.deliverySummary ||
    tumorMeta?.deliverySummary ||
    "";

  featureCount.textContent = String(dataset.meta.featureDim);
  datasetDescription.textContent =
    dataset.featuresReadyCount < dataset.featuresTotal
      ? `${dataset.meta.description} Web-ready feature tensors loading ${featureProgressText(dataset)}.`
      : dataset.meta.description;
  activeFamiliesText.textContent = `${state.activeSimilarityGroups.size} of ${
    dataset.meta.similarityGroups.length
  } contrast families active.`;
  contrastLabel.textContent = getDisplayOption(dataset.meta).label;
  healthyPrepText.textContent = healthyMeta?.preprocessingSummary ?? "";
  healthyPrepText.hidden = !healthyMeta?.preprocessingSummary;
  tumorPrepText.textContent = tumorMeta?.preprocessingSummary ?? "";
  tumorPrepText.hidden = !tumorMeta?.preprocessingSummary;
  deliveryText.textContent = deliverySummary;
  deliveryText.hidden = !deliverySummary;
}

function renderRegionControls() {
  const dataset = getActiveDataset();
  regionControls.innerHTML = "";
  if (!dataset) {
    return;
  }

  const allButton = document.createElement("button");
  allButton.type = "button";
  allButton.className = "chip";
  allButton.textContent = "Whole brain";
  allButton.classList.toggle("is-active", !state.activeRegionId);
  allButton.addEventListener("click", () => {
    state.activeRegionId = null;
    invalidateSimilarityCache();
    renderRegionControls();
    renderPanels();
    updateStats();
  });
  regionControls.appendChild(allButton);

  for (const region of dataset.meta.regions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.textContent = region.label;
    button.classList.toggle("is-active", state.activeRegionId === region.id);
    button.addEventListener("click", () => {
      state.activeRegionId = region.id;
      invalidateSimilarityCache();
      renderRegionControls();
      if (state.selected) {
        const subject = dataset.subjects[state.selected.subjectIndex];
        const flat = state.selected.y * subject.width + state.selected.x;
        if (!getActiveMaskInfo(subject).mask[flat]) {
          clearSelection(`Reference cleared. Pick inside ${region.label}.`);
          updateStats();
          return;
        }
      }
      renderPanels();
      updateStats();
    });
    regionControls.appendChild(button);
  }
}

function renderSimilarityControls() {
  const dataset = getActiveDataset();
  similarityControls.innerHTML = "";
  if (!dataset) {
    return;
  }

  for (const group of dataset.meta.similarityGroups) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip chip-wide";
    button.textContent = group.label;
    button.classList.toggle("is-active", state.activeSimilarityGroups.has(group.id));
    button.addEventListener("click", () => {
      if (state.activeSimilarityGroups.has(group.id)) {
        state.activeSimilarityGroups.delete(group.id);
      } else {
        state.activeSimilarityGroups.add(group.id);
      }
      invalidateSimilarityCache();
      renderSimilarityControls();
      renderPanels();
      updateStats();
    });
    similarityControls.appendChild(button);
  }
}

function renderDatasetTabs() {
  datasetTabs.innerHTML = "";
  for (const dataset of state.meta.datasets) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "dataset-tab";
    button.textContent = dataset.name;
    button.classList.toggle("is-active", dataset.id === state.datasetId);
    button.addEventListener("click", () => {
      activateDataset(dataset.id).catch((err) => {
        statusText.textContent = "Failed to switch dataset.";
        pickedText.textContent = String(err.message || err);
      });
    });
    datasetTabs.appendChild(button);
  }
}

function syncContrastControls() {
  const dataset = getActiveDataset();
  if (!dataset) {
    return;
  }
  contrastSlider.min = "0";
  contrastSlider.max = String(dataset.meta.displayOptions.length - 1);
  contrastSlider.value = String(state.displayIndex);
  contrastLabel.textContent = getDisplayOption(dataset.meta).label;
}

function syncZoomControls() {
  zoomSlider.value = String(Math.round(state.zoom * 100));
  zoomValue.textContent = `${state.zoom.toFixed(1)}x`;
}

function setZoom(nextZoom) {
  const clamped = clamp(nextZoom, 1, 2.8);
  if (Math.abs(clamped - state.zoom) < 1e-6) {
    return;
  }
  state.zoom = clamped;
  syncZoomControls();
  renderPanels();
}

function stepContrast(direction) {
  const dataset = getActiveDataset();
  if (!dataset) {
    return;
  }
  state.displayIndex = clamp(state.displayIndex + direction, 0, dataset.meta.displayOptions.length - 1);
  syncContrastControls();
  renderPanels();
  updateStats();
}

async function loadFeaturesIntoSubject(dataset, idx, baseDir) {
  const subject = dataset.subjects[idx];
  const subjectMeta = dataset.meta.subjects[idx];
  const featureLength = subject.pixelCount * dataset.meta.featureDim;
  const features = await loadTypedArray(`${baseDir}/${subjectMeta.feature_file}`, dataset.meta.dtype.feature, featureLength);
  const norms = new Float32Array(subject.pixelCount);

  for (let flat = 0; flat < subject.pixelCount; flat++) {
    let norm2 = 0;
    const base = flat * subject.k;
    for (let j = 0; j < subject.k; j++) {
      const value = features[base + j];
      norm2 += value * value;
    }
    norms[flat] = Math.sqrt(norm2);
  }

  subject.features = features;
  subject.norms = norms;
  subject.featuresReady = true;
}

function startFeatureLoading(dataset) {
  if (dataset.featureLoadPromise) {
    return dataset.featureLoadPromise;
  }

  dataset.featureLoadPromise = (async () => {
    const concurrency = Math.min(2, dataset.subjects.length);
    let nextIndex = 0;

    const worker = async () => {
      while (nextIndex < dataset.subjects.length) {
        const i = nextIndex;
        nextIndex += 1;

        if (dataset.subjects[i].featuresReady) {
          continue;
        }

        if (state.datasetId === dataset.meta.id) {
          statusText.textContent = loadingStatusText(dataset);
        }

        await loadFeaturesIntoSubject(dataset, i, state.baseDir);
        dataset.featuresReadyCount += 1;

        if (state.datasetId === dataset.meta.id) {
          updateStats();
          renderPanels();
          if (dataset.featuresReadyCount < dataset.featuresTotal) {
            statusText.textContent = loadingStatusText(dataset);
          } else {
            statusText.textContent = "Click any subject panel to choose a reference voxel.";
          }
        }
      }
    };

    await Promise.all(Array.from({ length: concurrency }, () => worker()));
  })().catch((err) => {
    dataset.featureLoadError = err;
    if (state.datasetId === dataset.meta.id) {
      statusText.textContent = "Failed to load feature data.";
      pickedText.textContent = String(err.message || err);
    }
  });

  return dataset.featureLoadPromise;
}

function resetForDataset(dataset) {
  state.activeRegionId = null;
  state.selected = null;
  state.similarityCache = null;
  state.activeSimilarityGroups = new Set(dataset.meta.similarityGroups.map((group) => group.id));
  state.displayIndex = findDisplayIndex(dataset.meta, dataset.meta.defaultDisplay);
  state.zoom = 1;
  pickedText.textContent = "No reference voxel selected. Click a slice to project voxel homology across subjects.";
}

async function activateDataset(datasetId) {
  const datasetMeta = state.meta.datasets.find((dataset) => dataset.id === datasetId);
  if (!datasetMeta) {
    throw new Error(`Unknown dataset: ${datasetId}`);
  }

  state.datasetId = datasetId;
  renderDatasetTabs();

  let dataset = state.datasetCache.get(datasetId);
  if (!dataset) {
    showLoadingPanels(datasetMeta);
    resetForDataset({ meta: datasetMeta });
    syncZoomControls();
    statusText.textContent = `Preparing ${datasetMeta.name}...`;
    dataset = await createDataset(datasetMeta, state.baseDir);
    state.datasetCache.set(datasetId, dataset);
  } else {
    resetForDataset(dataset);
  }

  buildPanels(dataset);
  renderRegionControls();
  renderSimilarityControls();
  syncContrastControls();
  syncZoomControls();
  updateStats();

  if (dataset.featureLoadError) {
    statusText.textContent = "Failed to load feature data.";
    pickedText.textContent = String(dataset.featureLoadError.message || dataset.featureLoadError);
    return;
  }

  if (dataset.featuresReadyCount < dataset.featuresTotal) {
    statusText.textContent = previewReadyStatusText(dataset);
    startFeatureLoading(dataset);
  } else {
    statusText.textContent = "Click any subject panel to choose a reference voxel.";
  }
}

async function init() {
  try {
    statusText.textContent = "Loading demo metadata...";
    const { meta, baseDir } = await loadMetadata();
    state.meta = meta;
    state.baseDir = baseDir;
    renderDatasetTabs();
    await activateDataset(meta.defaultDatasetId ?? meta.datasets[0].id);
  } catch (err) {
    statusText.textContent = "Failed to load data files";
    pickedText.textContent = String(err.message || err);
  }
}

resetBtn.addEventListener("click", () => clearSelection("Reference cleared."));
alphaSlider.addEventListener("input", (event) => {
  state.alpha = Number(event.target.value) / 100;
  renderPanels();
});
zoomSlider.addEventListener("input", (event) => {
  setZoom(Number(event.target.value) / 100);
});
contrastSlider.addEventListener("input", (event) => {
  state.displayIndex = Number(event.target.value);
  syncContrastControls();
  renderPanels();
  updateStats();
});
contrastPrevBtn.addEventListener("click", () => stepContrast(-1));
contrastNextBtn.addEventListener("click", () => stepContrast(1));

alphaSlider.value = "86";
syncZoomControls();

init();
