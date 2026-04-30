const META_PATHS = ["./data/metadata.json", "./metadata.json"];
const FETCH_OPTIONS = { cache: "no-store" };
const LOCAL_THREE_MODULE = "./vendor/three.module.js";
const LOCAL_ORBIT_CONTROLS = "./vendor/OrbitControls.js";

const datasetTabs = document.getElementById("datasetTabs");
const datasetDescription = document.getElementById("datasetDescription");
const panelGrid = document.getElementById("panelGrid");
const viewerTitle = document.getElementById("viewerTitle");
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
const contrastControlStack = document.getElementById("contrastControlStack");
const zoomControlStack = document.getElementById("zoomControlStack");
const cortexViewToggle = document.getElementById("cortexViewToggle");
const cortexView2dBtn = document.getElementById("cortexView2dBtn");
const cortexView3dBtn = document.getElementById("cortexView3dBtn");
const interactionHint = document.getElementById("interactionHint");
const regionStripCard = document.getElementById("regionStripCard");
const similarityStripCard = document.getElementById("similarityStripCard");
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
  dragPicking: null,
  meshLib: null,
  meshClickCandidate: null,
  meshPickedLabel: "",
  cortexViewMode: "slice",
};

let layoutSyncFrame = 0;
let globalRenderTick = 0;

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function clamp01(x) {
  return clamp(x, 0, 1);
}

function dirnameUrl(url) {
  const trimmed = String(url ?? "").replace(/\/+$/, "");
  const idx = trimmed.lastIndexOf("/");
  return idx >= 0 ? trimmed.slice(0, idx) || "." : ".";
}

function joinUrl(base, path) {
  const rel = String(path ?? "");
  if (!rel) {
    return String(base ?? ".");
  }
  if (/^(?:[a-z]+:)?\/\//i.test(rel) || rel.startsWith("/")) {
    return rel;
  }
  const root = String(base ?? ".").replace(/\/+$/, "");
  return `${root || "."}/${rel.replace(/^\.?\//, "")}`;
}

function buildInflatedDisplayCoords(coords, hemi) {
  const out = new Float32Array(coords.length);
  for (let i = 0; i < coords.length; i += 3) {
    const x = coords[i + 0];
    const y = coords[i + 1];
    const z = coords[i + 2];
    out[i + 0] = x;
    out[i + 1] = z;
    out[i + 2] = -y;
  }
  return out;
}

function subjectLabel(index) {
  return `Subject ${index + 1}`;
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
    coords: String(dtypeConfig.coords ?? "float32").toLowerCase(),
    faces: String(dtypeConfig.faces ?? "uint32").toLowerCase(),
    color: String(dtypeConfig.color ?? "uint8").toLowerCase(),
    label: String(dtypeConfig.label ?? "uint8").toLowerCase(),
    index: String(dtypeConfig.index ?? "uint16").toLowerCase(),
    valid: String(dtypeConfig.valid ?? "uint8").toLowerCase(),
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
      viewerKind: String(dataset.viewer_kind ?? dataset.viewerKind ?? "slice"),
      dtype: normalizeDtypes(dataset.dtype),
      t1Label: String(dataset.t1_label ?? dataset.t1Label ?? "T1 structural"),
      preprocessingSummary: String(dataset.preprocessing_summary ?? dataset.preprocessingSummary ?? ""),
      deliverySummary: String(dataset.delivery_summary ?? dataset.deliverySummary ?? ""),
      interactionHint: String(dataset.interaction_hint ?? dataset.interactionHint ?? ""),
      parcellation: String(dataset.parcellation ?? ""),
      annotName: String(dataset.annot_name ?? dataset.annotName ?? ""),
      parcelNames: dataset.parcel_names ?? dataset.parcelNames ?? null,
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
    if (hydrated.id === "cortex") {
      hydrated.displayOptions = hydrated.displayOptions.filter((option) => option.kind === "feature");
    }
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

function firstFeatureDisplayIndex(datasetMeta) {
  const idx = datasetMeta.displayOptions.findIndex((option) => option.kind === "feature");
  return idx >= 0 ? idx : 0;
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
  if (dtype === "uint16") {
    const arr = new Uint16Array(buf);
    if (arr.length !== expectedLength) {
      throw new Error(`Unexpected length for ${url}: got ${arr.length}, expected ${expectedLength}`);
    }
    return arr;
  }
  if (dtype === "uint32") {
    const arr = new Uint32Array(buf);
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
    await loadTypedArray(joinUrl(baseDir, filename), datasetMeta.dtype.region, pixelCount),
  ]);

  const [t1f, mask, ...regionPairs] = await Promise.all([
    loadTypedArray(joinUrl(baseDir, subjectMeta.t1_file), datasetMeta.dtype.t1, pixelCount),
    loadTypedArray(joinUrl(baseDir, subjectMeta.mask_file), datasetMeta.dtype.mask, pixelCount),
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

async function loadCortexSliceSubjectPreview(healthyMeta, healthySubjectMeta, cortexSubjectMeta, baseDir) {
  const ribbonFile = healthySubjectMeta.region_files?.cortical_ribbon;
  if (!ribbonFile) {
    throw new Error(`Missing cortical ribbon mask for ${healthySubjectMeta.name ?? cortexSubjectMeta.name}`);
  }
  const ribbonSubjectMeta = {
    ...healthySubjectMeta,
    region_files: {
      cortical_ribbon: ribbonFile,
    },
  };
  const subject = await loadSubjectPreview(healthyMeta, ribbonSubjectMeta, baseDir);
  subject.name = cortexSubjectMeta.name ?? subject.name;
  subject.featureFile = healthySubjectMeta.feature_file;
  return subject;
}

async function loadJson(url) {
  const res = await fetch(url, FETCH_OPTIONS);
  if (!res.ok) {
    throw new Error(`Could not fetch ${url}`);
  }
  return res.json();
}

async function loadCortexSubjectPreview(datasetMeta, subjectMeta, baseDir) {
  const manifestUrl = joinUrl(baseDir, subjectMeta.mesh_manifest_file);
  const manifestDir = dirnameUrl(manifestUrl);
  const manifest = await loadJson(manifestUrl);
  const hemis = {};
  await Promise.all(
    ["lh", "rh"].map(async (hemi) => {
      const hemiMeta = manifest.hemis[hemi];
      const files = hemiMeta.files;
      const validPromise = files.valid
        ? loadTypedArray(joinUrl(manifestDir, files.valid), datasetMeta.dtype.valid, hemiMeta.vertex_count)
        : Promise.resolve(new Uint8Array(hemiMeta.vertex_count).fill(1));
      const [coords, faces, colors, labels, fullToSample, validMask] = await Promise.all([
        loadTypedArray(joinUrl(manifestDir, files.coords), datasetMeta.dtype.coords, hemiMeta.vertex_count * 3),
        loadTypedArray(joinUrl(manifestDir, files.faces), datasetMeta.dtype.faces, hemiMeta.face_count * 3),
        loadTypedArray(joinUrl(manifestDir, files.colors), datasetMeta.dtype.color, hemiMeta.vertex_count * 3),
        loadTypedArray(joinUrl(manifestDir, files.labels), datasetMeta.dtype.label, hemiMeta.vertex_count),
        loadTypedArray(joinUrl(manifestDir, files.full_to_sample), datasetMeta.dtype.index, hemiMeta.vertex_count),
        validPromise,
      ]);

      hemis[hemi] = {
        vertexCount: hemiMeta.vertex_count,
        faceCount: hemiMeta.face_count,
        sampleCount: hemiMeta.sample_count,
        files,
        coords,
        displayCoords: buildInflatedDisplayCoords(coords, hemi),
        faces,
        colors,
        labels,
        fullToSample,
        validMask,
        sampleFeatures: null,
        baseColors: new Uint8Array(colors),
      };
    })
  );

  return {
    name: subjectMeta.name,
    parcellation: manifest.parcellation ?? "destrieux",
    annotName: manifest.annot_name ?? "aparc.a2009s",
    normalizationMode: manifest.normalization_mode ?? "",
    blockWeightMode: manifest.block_weight_mode ?? "",
    hemis,
    meshReady: true,
    featuresReady: false,
    displayCache: new Map(),
  };
}

async function ensureMeshLib() {
  if (state.meshLib) {
    return state.meshLib;
  }

  const [threeModule, controlsModule] = await Promise.all([
    import(LOCAL_THREE_MODULE),
    import(LOCAL_ORBIT_CONTROLS),
  ]);
  state.meshLib = {
    THREE: threeModule,
    OrbitControls: controlsModule.OrbitControls,
    raycaster: new threeModule.Raycaster(),
    pointer: new threeModule.Vector2(),
  };
  return state.meshLib;
}

async function createDataset(datasetMeta, baseDir) {
  if (datasetMeta.viewerKind === "mesh") {
    const healthyMeta = state.meta.datasets.find((item) => item.id === "healthy");
    if (!healthyMeta) {
      throw new Error("Healthy dataset is required for cortex 2D mode.");
    }
    if (healthyMeta.subjects.length < datasetMeta.subjects.length) {
      throw new Error("Healthy dataset has fewer subjects than cortex dataset.");
    }

    let loadedCount = 0;
    const total = datasetMeta.subjects.length;
    statusText.textContent = `Loading cortical meshes and 2D slices (0/${total})`;
    const subjects = await Promise.all(
      datasetMeta.subjects.map(async (subjectMeta, idx) => {
        const subject = await loadCortexSubjectPreview(datasetMeta, subjectMeta, baseDir);
        subject.sliceSubject = await loadCortexSliceSubjectPreview(
          healthyMeta,
          healthyMeta.subjects[idx],
          subjectMeta,
          baseDir
        );
        loadedCount += 1;
        statusText.textContent = `Loading cortical meshes and 2D slices (${loadedCount}/${total})`;
        return subject;
      })
    );

    return {
      meta: datasetMeta,
      subjects,
      featuresReadyCount: 0,
      featuresTotal: datasetMeta.subjects.length,
      featureLoadPromise: null,
      featureLoadError: null,
    };
  }

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

function isCortexDataset(dataset = getActiveDataset()) {
  return dataset?.meta?.id === "cortex";
}

function isCortexMeshMode(dataset = getActiveDataset()) {
  return isCortexDataset(dataset) && state.cortexViewMode === "mesh";
}

function isCortexSliceMode(dataset = getActiveDataset()) {
  return isCortexDataset(dataset) && state.cortexViewMode === "slice";
}

function formatParcelLabel(rawLabel) {
  if (!rawLabel) {
    return "";
  }

  const raw = String(rawLabel).trim();
  const lowerRaw = raw.toLowerCase();
  const desikanMap = {
    unknown: "Unknown",
    bankssts: "Banks of Superior Temporal Sulcus",
    caudalanteriorcingulate: "Caudal Anterior Cingulate",
    caudalmiddlefrontal: "Caudal Middle Frontal",
    corpuscallosum: "Corpus Callosum",
    cuneus: "Cuneus",
    entorhinal: "Entorhinal Cortex",
    fusiform: "Fusiform Gyrus",
    inferiorparietal: "Inferior Parietal",
    inferiortemporal: "Inferior Temporal",
    isthmuscingulate: "Isthmus Cingulate",
    lateraloccipital: "Lateral Occipital",
    lateralorbitofrontal: "Lateral Orbitofrontal",
    lingual: "Lingual Gyrus",
    medialorbitofrontal: "Medial Orbitofrontal",
    middletemporal: "Middle Temporal",
    parahippocampal: "Parahippocampal Gyrus",
    paracentral: "Paracentral",
    parsopercularis: "Pars Opercularis",
    parsorbitalis: "Pars Orbitalis",
    parstriangularis: "Pars Triangularis",
    pericalcarine: "Pericalcarine Cortex",
    postcentral: "Postcentral Gyrus",
    posteriorcingulate: "Posterior Cingulate",
    precentral: "Precentral Gyrus",
    precuneus: "Precuneus",
    rostralanteriorcingulate: "Rostral Anterior Cingulate",
    rostralmiddlefrontal: "Rostral Middle Frontal",
    superiorfrontal: "Superior Frontal",
    superiorparietal: "Superior Parietal",
    superiortemporal: "Superior Temporal",
    supramarginal: "Supramarginal Gyrus",
    frontalpole: "Frontal Pole",
    temporalpole: "Temporal Pole",
    transversetemporal: "Transverse Temporal",
    insula: "Insula",
  };
  if (desikanMap[lowerRaw]) {
    return desikanMap[lowerRaw];
  }

  let text = raw.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
  if (!text) {
    return "";
  }

  const tokenMap = {
    G: "Gyrus",
    S: "Sulcus",
    inf: "Inferior",
    sup: "Superior",
    ant: "Anterior",
    post: "Posterior",
    lat: "Lateral",
    med: "Medial",
    temp: "Temporal",
    front: "Frontal",
    pariet: "Parietal",
    transv: "Transverse",
    opercular: "Opercular",
    triangul: "Triangular",
    fusifor: "Fusiform",
    parahip: "Parahippocampal",
  };

  text = text
    .split(" ")
    .map((token) => {
      const lower = token.toLowerCase();
      const mapped = tokenMap[token] ?? tokenMap[lower] ?? token;
      return mapped.replace(/\b\w/g, (ch) => ch.toUpperCase());
    })
    .join(" ");

  return text;
}

function getParcelLabel(dataset, hemi, label) {
  if (!dataset?.meta?.parcelNames) {
    return label >= 0 ? `parcel ${label}` : "unlabeled";
  }
  const raw = dataset.meta.parcelNames[`${hemi}:${label}`] ?? `parcel ${label}`;
  return formatParcelLabel(raw);
}

function formatParcellationName(datasetMeta) {
  const parcellation = datasetMeta?.parcellation || "destrieux";
  if (parcellation.toLowerCase() === "destrieux") {
    return "Destrieux";
  }
  if (parcellation.toLowerCase() === "desikan") {
    return "Desikan-Killiany";
  }
  if (parcellation.toLowerCase() === "brodmann") {
    return "Brodmann";
  }
  return parcellation.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function getDisplayOption(datasetMeta) {
  return datasetMeta.displayOptions[state.displayIndex] ?? datasetMeta.displayOptions[0];
}

function getSubjectDisplayImage(subject, displayOption, options = {}) {
  if (displayOption.kind === "t1" || !subject.featuresReady) {
    return subject.displayCache.get("t1");
  }

  const key = `f${displayOption.index}${options.cacheSuffix ?? ""}`;
  if (subject.displayCache.has(key)) {
    return subject.displayCache.get(key);
  }

  const mask = options.mask ?? subject.mask;
  const values = new Float32Array(subject.pixelCount);
  for (let flat = 0; flat < subject.pixelCount; flat++) {
    values[flat] = subject.features[flat * subject.k + displayOption.index];
  }
  const u8 = normalizeImageToU8(values, mask, {
    backgroundGray: options.backgroundGray ?? 12,
    lowerQ: 0.02,
    upperQ: 0.98,
  });
  subject.displayCache.set(key, u8);
  return u8;
}

function getRegionLabel(dataset, regionId) {
  if (isCortexSliceMode(dataset)) {
    return "Cortical ribbon";
  }
  if (!regionId) {
    return "Whole Brain";
  }
  const region = dataset.meta.regions.find((item) => item.id === regionId);
  return region?.label ?? regionId;
}

function getActiveMaskInfo(subject) {
  if (isCortexSliceMode()) {
    return subject.regions.cortical_ribbon ?? subject.emptyMaskInfo;
  }
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

function drawRegionFill(ctx, regionInfo, color = [255, 80, 70], alpha = 0.18) {
  if (!regionInfo?.pixels?.length) {
    return;
  }

  // Blend one flat color into the selected mask pixels.
  const image = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  for (let i = 0; i < regionInfo.pixels.length; i++) {
    const base = regionInfo.pixels[i] * 4;
    image.data[base + 0] = Math.round(image.data[base + 0] * (1 - alpha) + color[0] * alpha);
    image.data[base + 1] = Math.round(image.data[base + 1] * (1 - alpha) + color[1] * alpha);
    image.data[base + 2] = Math.round(image.data[base + 2] * (1 - alpha) + color[2] * alpha);
    image.data[base + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
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

function drawCortexRibbonImage(ctx, subject, visible, ribbonInfo) {
  const image = ctx.createImageData(subject.width, subject.height);
  for (let i = 0; i < subject.pixelCount; i++) {
    const base = i * 4;
    image.data[base + 3] = 255;
    if (!ribbonInfo.mask[i]) {
      continue;
    }
    const gray = visible[i];
    image.data[base + 0] = gray;
    image.data[base + 1] = gray;
    image.data[base + 2] = gray;
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
  return `${prefix} ${featureProgressText(dataset)}.`;
}

function previewReadyStatusText(dataset) {
  return isCortexMeshMode(dataset)
    ? `Loading cortical feature anchors ${featureProgressText(dataset)}.`
    : `Loading feature data ${featureProgressText(dataset)}.`;
}

function hasActiveReference() {
  return Boolean(state.selected && state.selected.datasetId === state.datasetId);
}

function syncLegendSizing() {
  cancelAnimationFrame(layoutSyncFrame);
  layoutSyncFrame = requestAnimationFrame(() => {
    const firstPanel = panelGrid.querySelector(".panel");
    const firstShell = firstPanel?.querySelector(".slice-shell");
    if (!firstPanel || !firstShell) {
      return;
    }

    const shellRect = firstShell.getBoundingClientRect();
    const panelRect = firstPanel.getBoundingClientRect();
    document.documentElement.style.setProperty("--slice-view-height", `${Math.round(shellRect.height)}px`);
    document.documentElement.style.setProperty(
      "--slice-top-offset",
      `${Math.max(0, Math.round(shellRect.top - panelRect.top))}px`
    );
  });
}

function getMeshSelectionText(dataset, hemi, label, suffix = "") {
  return `Picked parcel: ${hemi.toUpperCase()} ${getParcelLabel(dataset, hemi, label)}${suffix}`;
}

function meshPickedText() {
  if (state.meshPickedLabel) {
    return state.meshPickedLabel;
  }
  const dataset = getActiveDataset();
  if (!dataset || !state.selected || state.selected.datasetId !== state.datasetId || state.selected.kind !== "mesh") {
    return "";
  }
  return getMeshSelectionText(dataset, state.selected.hemi, state.selected.label);
}

function buildMeshReferenceVector(subject, hemi, sampleIndex, channelIndices) {
  const hemiData = subject.hemis[hemi];
  const ref = new Float32Array(channelIndices.length);
  let norm2 = 0;
  const base = sampleIndex * state.meta.datasets.find((item) => item.id === state.datasetId).featureDim;
  for (let i = 0; i < channelIndices.length; i++) {
    const value = hemiData.sampleFeatures[base + channelIndices[i]];
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

function cosineMeshSamples(hemiData, refVec, channelIndices, featureDim) {
  const sims = new Float32Array(hemiData.sampleCount);
  const features = hemiData.sampleFeatures;
  for (let sampleIdx = 0; sampleIdx < hemiData.sampleCount; sampleIdx++) {
    const base = sampleIdx * featureDim;
    let dot = 0;
    let norm2 = 0;
    for (let j = 0; j < channelIndices.length; j++) {
      const value = features[base + channelIndices[j]];
      dot += value * refVec[j];
      norm2 += value * value;
    }
    sims[sampleIdx] = dot / (Math.sqrt(norm2) + 1e-8);
  }
  return sims;
}

function summarizeMeshSimilarity(perHemi) {
  let sum = 0;
  let n = 0;
  let max = Number.NEGATIVE_INFINITY;
  for (const hemi of ["lh", "rh"]) {
    const sims = perHemi[hemi]?.sims ?? null;
    if (!sims?.length) {
      continue;
    }
    for (let i = 0; i < sims.length; i++) {
      sum += sims[i];
      n += 1;
      if (sims[i] > max) {
        max = sims[i];
      }
    }
  }
  if (!n) {
    return null;
  }
  return { mean: sum / n, max };
}

function updateMeshColors(panel, similarityEntry) {
  for (const hemi of ["lh", "rh"]) {
    const hemiRuntime = panel.meshRuntime?.hemis?.[hemi];
    const hemiData = panel.subject.hemis[hemi];
    if (!hemiRuntime || !hemiData) {
      continue;
    }
    const colorArray = hemiRuntime.colorAttr.array;
    const baseColors = hemiData.baseColors;
    const sims = similarityEntry?.hemis?.[hemi]?.sims ?? null;
    if (!sims) {
      colorArray.set(baseColors);
      hemiRuntime.colorAttr.needsUpdate = true;
      continue;
    }
    for (let vertexIdx = 0; vertexIdx < hemiData.vertexCount; vertexIdx++) {
      const sampleIdx = hemiData.fullToSample[vertexIdx];
      if (!hemiData.validMask[vertexIdx] || sampleIdx >= sims.length) {
        const baseOffset = vertexIdx * 3;
        colorArray[baseOffset + 0] = baseColors[baseOffset + 0];
        colorArray[baseOffset + 1] = baseColors[baseOffset + 1];
        colorArray[baseOffset + 2] = baseColors[baseOffset + 2];
        continue;
      }
      const sim = sims[sampleIdx];
      const [r, g, b] = colorMapViridis(clamp01(sim));
      const baseOffset = vertexIdx * 3;
      colorArray[baseOffset + 0] = Math.round(baseColors[baseOffset + 0] * (1 - state.alpha) + r * state.alpha);
      colorArray[baseOffset + 1] = Math.round(baseColors[baseOffset + 1] * (1 - state.alpha) + g * state.alpha);
      colorArray[baseOffset + 2] = Math.round(baseColors[baseOffset + 2] * (1 - state.alpha) + b * state.alpha);
    }
    hemiRuntime.colorAttr.needsUpdate = true;
  }
}

function syncMeshMarker(panel) {
  const marker = panel.meshRuntime?.selectionMarker;
  if (!marker) {
    return;
  }
  const isReference =
    state.selected &&
    state.selected.datasetId === state.datasetId &&
    state.selected.subjectIndex === panel.idx &&
    state.selected.kind === "mesh";
  if (!isReference) {
    marker.visible = false;
    return;
  }
  const hemiRuntime = panel.meshRuntime.hemis[state.selected.hemi];
  const positionAttr = hemiRuntime.geometry.getAttribute("position");
  const i3 = state.selected.fullVertexIndex * 3;
  marker.position.set(
    positionAttr.array[i3 + 0],
    positionAttr.array[i3 + 1],
    positionAttr.array[i3 + 2]
  );
  marker.position.add(hemiRuntime.mesh.position);
  marker.visible = true;
}

function renderMeshPanel(panel, similarityEntry) {
  if (!panel.meshRuntime) {
    return;
  }

  const dataset = getActiveDataset();
  updateMeshColors(panel, similarityEntry);
  syncMeshMarker(panel);

  panel.title.textContent = "Cortical Surface";
  panel.meta.textContent = `${panel.subject.name ?? subjectLabel(panel.idx)} · inflated 3D mesh`;

  const isReference =
    state.selected &&
    state.selected.datasetId === state.datasetId &&
    state.selected.subjectIndex === panel.idx &&
    state.selected.kind === "mesh";

  if (!panel.subject.featuresReady) {
    panel.badge.textContent = "Loading";
  } else if (isReference && similarityEntry?.stats) {
    panel.badge.textContent = `Ref · ${similarityEntry.stats.mean.toFixed(2)}`;
  } else if (isReference) {
    panel.badge.textContent = "Ref";
  } else if (similarityEntry?.stats) {
    panel.badge.textContent = `Mean ${similarityEntry.stats.mean.toFixed(2)}`;
  } else {
    panel.badge.textContent = "Parcellation";
  }

  panel.node.classList.remove("is-loading");
  panel.node.classList.add("is-mesh");
  panel.node.classList.toggle("is-reference", Boolean(isReference));
  panel.node.classList.toggle(
    "is-awaiting-pick",
    !hasActiveReference() && dataset.featuresReadyCount >= dataset.featuresTotal
  );
  const pickedLabel = meshPickedText();
  panel.node.classList.toggle("has-picked-parcel", Boolean(pickedLabel));
  panel.note.innerHTML = panel.subject.featuresReady
    ? `<span class="mesh-picked-note">${pickedLabel}</span><span class="mesh-help-note">Drag to rotate · wheel to zoom · click cortex to pick</span>`
    : `<span class="mesh-help-note">Loading features ${featureProgressText(dataset)}</span>`;

  const rect = panel.canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const widthChanged = Math.abs((panel.meshRuntime.canvasCssWidth ?? 0) - width) > 0.5;
  const heightChanged = Math.abs((panel.meshRuntime.canvasCssHeight ?? 0) - height) > 0.5;
  if (widthChanged || heightChanged) {
    panel.meshRuntime.renderer.setSize(width, height, false);
    panel.meshRuntime.canvasCssWidth = width;
    panel.meshRuntime.canvasCssHeight = height;
    fitMeshCamera(panel);
  }
  panel.meshRuntime.renderer.render(panel.meshRuntime.scene, panel.meshRuntime.camera);
}

function requestPanelRender(panel) {
  if (!panel?.meshRuntime) {
    return;
  }
  cancelAnimationFrame(panel.meshRuntime.frameHandle ?? 0);
  panel.meshRuntime.frameHandle = requestAnimationFrame(() =>
    renderMeshPanel(panel, state.similarityCache?.perPanel?.[panel.idx] ?? null)
  );
}

function rotateMeshPanel(panel, dx, dy) {
  const runtime = panel.meshRuntime;
  if (!runtime) {
    return;
  }

  // Rotate the centered brain group directly. This keeps the behavior simple.
  runtime.group.rotation.y += dx * 0.008;
  runtime.group.rotation.x += dy * 0.008;
  runtime.group.rotation.x = clamp(runtime.group.rotation.x, -Math.PI * 0.48, Math.PI * 0.48);
  requestPanelRender(panel);
}

function zoomMeshPanel(panel, deltaY) {
  const runtime = panel.meshRuntime;
  if (!runtime) {
    return;
  }

  // Move the camera toward or away from the fixed orbit target.
  const target = runtime.orbitTarget ?? runtime.controls.target;
  const offset = runtime.camera.position.clone().sub(target);
  const oldDistance = offset.length();
  const scale = deltaY < 0 ? 0.88 : 1.12;
  const newDistance = clamp(
    oldDistance * scale,
    runtime.minCameraDistance ?? oldDistance * 0.35,
    runtime.maxCameraDistance ?? oldDistance * 3.0
  );
  offset.setLength(newDistance);
  runtime.camera.position.copy(target).add(offset);
  runtime.camera.lookAt(target);
  requestPanelRender(panel);
}

function addRaisedBoundaryPoint(values, coords, vertexIndex, center, lift) {
  const i3 = vertexIndex * 3;
  const x = coords[i3 + 0];
  const y = coords[i3 + 1];
  const z = coords[i3 + 2];
  const dx = x - center.x;
  const dy = y - center.y;
  const dz = z - center.z;
  const length = Math.hypot(dx, dy, dz) || 1;

  // Lift the line just above the surface to avoid flickering against triangles.
  values.push(x + (dx / length) * lift, y + (dy / length) * lift, z + (dz / length) * lift);
}

function buildParcelBoundaryPositions(hemiData, center) {
  const coords = hemiData.displayCoords ?? hemiData.coords;
  const faces = hemiData.faces;
  const labels = hemiData.labels;
  const values = [];
  const seenEdges = new Set();
  const lift = 0.35;

  const addEdge = (a, b) => {
    if (labels[a] === labels[b]) {
      return;
    }

    // Store each mesh edge once, even though it belongs to two triangles.
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const key = lo * hemiData.vertexCount + hi;
    if (seenEdges.has(key)) {
      return;
    }
    seenEdges.add(key);

    addRaisedBoundaryPoint(values, coords, a, center, lift);
    addRaisedBoundaryPoint(values, coords, b, center, lift);
  };

  for (let i = 0; i < faces.length; i += 3) {
    const a = faces[i + 0];
    const b = faces[i + 1];
    const c = faces[i + 2];
    addEdge(a, b);
    addEdge(b, c);
    addEdge(c, a);
  }

  return new Float32Array(values);
}

function resolveMeshVertexFromHit(intersection, mesh) {
  const attr = mesh.geometry.getAttribute("position");
  const pointLocal = mesh.worldToLocal(intersection.point.clone());
  const indices = [intersection.face.a, intersection.face.b, intersection.face.c];
  let bestVertex = indices[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const vertexIndex of indices) {
    const i3 = vertexIndex * 3;
    const dx = attr.array[i3 + 0] - pointLocal.x;
    const dy = attr.array[i3 + 1] - pointLocal.y;
    const dz = attr.array[i3 + 2] - pointLocal.z;
    const dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 < bestDistance) {
      bestDistance = dist2;
      bestVertex = vertexIndex;
    }
  }
  return bestVertex;
}

function pickMeshReference(event, panel) {
  const dataset = getActiveDataset();
  if (!dataset || dataset.featuresReadyCount < dataset.featuresTotal) {
    return false;
  }

  const rect = panel.canvas.getBoundingClientRect();
  const { raycaster, pointer } = state.meshLib;
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, panel.meshRuntime.camera);
  const objects = [panel.meshRuntime.hemis.lh.mesh, panel.meshRuntime.hemis.rh.mesh];
  const intersections = raycaster.intersectObjects(objects, false);
  if (!intersections.length) {
    return false;
  }

  const hit = intersections[0];
  const mesh = hit.object;
  const hemi = mesh.userData.hemi;
  const fullVertexIndex = resolveMeshVertexFromHit(hit, mesh);
  const hemiData = panel.subject.hemis[hemi];
  const sampleIndex = hemiData.fullToSample[fullVertexIndex];
  const label = hemiData.labels[fullVertexIndex];

  if (!hemiData.validMask[fullVertexIndex] || sampleIndex >= hemiData.sampleCount) {
    state.selected = null;
    state.meshPickedLabel = getMeshSelectionText(dataset, hemi, label, " (not sampled)");
    invalidateSimilarityCache();
    pickedText.textContent = "";
    statusText.textContent = "That parcel is outside the sampled ribbon slab.";
    renderPanels();
    return false;
  }

  state.selected = {
    datasetId: state.datasetId,
    subjectIndex: panel.idx,
    kind: "mesh",
    hemi,
    fullVertexIndex,
    sampleIndex,
    label,
  };
  state.meshPickedLabel = getMeshSelectionText(dataset, hemi, label);
  invalidateSimilarityCache();
  pickedText.textContent = "";
  statusText.textContent = "Computing cortical similarity maps...";
  renderPanels();
  return true;
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
    selectionKey:
      state.selected.kind === "mesh"
        ? `${state.selected.hemi}:${state.selected.sampleIndex}`
        : `${state.selected.x}:${state.selected.y}`,
    regionId: state.activeRegionId,
    groupKey: activeGroupKey(),
  };

  if (
    state.similarityCache &&
    state.similarityCache.datasetId === cacheKey.datasetId &&
    state.similarityCache.subjectIndex === cacheKey.subjectIndex &&
    state.similarityCache.selectionKey === cacheKey.selectionKey &&
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

  if (isCortexMeshMode(dataset)) {
    const refSubject = dataset.subjects[state.selected.subjectIndex];
    const refVec = buildMeshReferenceVector(refSubject, state.selected.hemi, state.selected.sampleIndex, channelIndices);
    if (!refVec) {
      statusText.textContent = "Reference vector is empty at this vertex.";
      return null;
    }

    const perPanel = dataset.subjects.map((subject) => {
      const perHemi = {};
      for (const hemi of ["lh", "rh"]) {
        perHemi[hemi] = {
          sims: cosineMeshSamples(subject.hemis[hemi], refVec, channelIndices, dataset.meta.featureDim),
        };
      }
      return {
        hemis: perHemi,
        stats: summarizeMeshSimilarity(perHemi),
      };
    });

    state.similarityCache = { ...cacheKey, perPanel };
    statusText.textContent = "Cortical similarity maps ready.";
    return state.similarityCache;
  }

  const sliceSubjects = isCortexSliceMode(dataset)
    ? dataset.subjects.map((subject) => subject.sliceSubject)
    : dataset.subjects;
  const refSubject = sliceSubjects[state.selected.subjectIndex];
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

  const perPanel = sliceSubjects.map((subject) => {
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
  if (isCortexMeshMode(dataset)) {
    renderMeshPanel(panel, similarityEntry);
    return;
  }

  const subject = panel.subject;
  const focusInfo = getActiveMaskInfo(subject);
  const cortexRibbonView = isCortexSliceMode(dataset);
  const displayOption = getDisplayOption(dataset.meta);
  const visible = cortexRibbonView
    ? subject.displayCache.get("t1")
    : getSubjectDisplayImage(subject, displayOption);
  const isRegionMode = !cortexRibbonView && Boolean(state.activeRegionId);
  const isCorticalRibbonMode = state.activeRegionId === "cortical_ribbon";
  const isReference =
    state.selected &&
    state.selected.datasetId === state.datasetId &&
    state.selected.subjectIndex === panel.idx;

  const bufferCtx = panel.bufferCanvas.getContext("2d");
  if (cortexRibbonView) {
    drawCortexRibbonImage(bufferCtx, subject, visible, focusInfo);
  } else {
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
  }

  if (similarityEntry?.sims?.length) {
    drawOverlay(bufferCtx, similarityEntry.pixelInfo.pixels, similarityEntry.sims, state.alpha);
  }
  if (isRegionMode && isCorticalRibbonMode && focusInfo.pixels.length) {
    drawRegionFill(bufferCtx, focusInfo);
  } else if (isRegionMode && focusInfo.pixels.length) {
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
  panel.meta.textContent = cortexRibbonView ? `Cortical ribbon · axial z=${subject.slice}` : `Axial slice z=${subject.slice}`;

  if (!subject.featuresReady) {
    panel.badge.textContent = "Loading";
  } else if (!cortexRibbonView && displayOption.kind === "feature" && !subject.displayCache.has(`f${displayOption.index}`)) {
    panel.badge.textContent = "Preparing";
  } else if (isReference && similarityEntry?.stats) {
    panel.badge.textContent = `Ref · ${similarityEntry.stats.mean.toFixed(2)}`;
  } else if (isReference) {
    panel.badge.textContent = "Ref";
  } else if (isRegionMode && !focusInfo.pixels.length) {
    panel.badge.textContent = "No target";
  } else if (similarityEntry?.stats) {
    panel.badge.textContent = `Mean ${similarityEntry.stats.mean.toFixed(2)}`;
  } else if (cortexRibbonView) {
    panel.badge.textContent = "T1 ribbon";
  } else if (displayOption.kind === "feature" && subject.featuresReady) {
    panel.badge.textContent = "Contrast view";
  } else {
    panel.badge.textContent = "Preview";
  }

  panel.node.classList.remove("is-loading");
  panel.node.classList.toggle("is-reference", Boolean(isReference));
  panel.node.classList.toggle(
    "is-awaiting-pick",
    !hasActiveReference() && dataset.featuresReadyCount >= dataset.featuresTotal
  );
  panel.note.textContent = subject.featuresReady ? "" : `Loading features ${featureProgressText(dataset)}`;
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
  syncLegendSizing();
}

function showLoadingPanels(datasetMeta) {
  panelGrid.innerHTML = "";
  panelGrid.classList.remove("is-single-mesh");
  panelGrid.classList.toggle("is-cortex-grid", datasetMeta.id === "cortex");
  state.panels = [];
  const showMeshPlaceholders = datasetMeta.id === "cortex" && state.cortexViewMode === "mesh";

  const loadingSubjects = datasetMeta.subjects;

  loadingSubjects.forEach((subject, idx) => {
    const node = panelTemplate.content.firstElementChild.cloneNode(true);
    const canvas = node.querySelector("canvas");
    const title = node.querySelector(".panel-title");
    const meta = node.querySelector(".panel-meta");
    const badge = node.querySelector(".panel-badge");
    const note = node.querySelector(".slice-note");
    node.classList.add("is-loading");
    canvas.width = showMeshPlaceholders ? 1200 : datasetMeta.width;
    canvas.height = showMeshPlaceholders ? 640 : datasetMeta.height;
    title.textContent = subjectLabel(idx);
    meta.textContent = showMeshPlaceholders ? "Loading inflated 3D mesh..." : "Loading preview slice...";
    badge.textContent = showMeshPlaceholders ? "Loading" : "Preview loading";
    panelGrid.appendChild(node);
    state.panels.push({
      node,
      canvas,
      title,
      meta,
      badge,
      note,
      bufferCanvas: null,
      subject: null,
      idx,
      viewState: null,
    });
  });

  syncLegendSizing();
}

function disposePanelRuntime(panel) {
  if (!panel?.meshRuntime) {
    return;
  }
  cancelAnimationFrame(panel.meshRuntime.frameHandle ?? 0);
  panel.meshRuntime.controls?.dispose();
  panel.meshRuntime.renderer?.dispose();
  for (const hemi of ["lh", "rh"]) {
    panel.meshRuntime.hemis?.[hemi]?.geometry?.dispose();
    panel.meshRuntime.hemis?.[hemi]?.material?.dispose();
    panel.meshRuntime.hemis?.[hemi]?.boundaryGeometry?.dispose();
    panel.meshRuntime.hemis?.[hemi]?.boundaryMaterial?.dispose();
  }
  panel.meshRuntime.selectionMarker?.geometry?.dispose();
  panel.meshRuntime.selectionMarker?.material?.dispose();
  panel.meshRuntime = null;
}

function fitMeshCamera(panel) {
  const runtime = panel.meshRuntime;
  if (!runtime) {
    return;
  }

  if (!runtime.hasInitialPose) {
    // Start from a lateral view, with a small top tilt so the surface reads as 3D.
    runtime.group.rotation.x = -0.22;
    runtime.group.rotation.y = Math.PI / 2;
    runtime.hasInitialPose = true;
  }

  const rect = panel.canvas.getBoundingClientRect();
  const aspect = Math.max(1e-3, rect.width / Math.max(rect.height, 1));
  const box = new runtime.THREE.Box3().setFromObject(runtime.group);
  const size = box.getSize(new runtime.THREE.Vector3());
  const center = box.getCenter(new runtime.THREE.Vector3());
  const halfY = Math.max(size.y * 0.5, 1);
  const halfX = Math.max(size.x * 0.5, 1);
  const fov = (runtime.camera.fov * Math.PI) / 180;
  const distY = halfY / Math.tan(fov / 2);
  const distX = halfX / (aspect * Math.tan(fov / 2));
  const distance = Math.max(distX, distY, size.z * 0.75) * 1.38;

  runtime.camera.aspect = aspect;
  runtime.camera.near = Math.max(0.1, distance * 0.1);
  runtime.camera.far = distance * 6 + size.z * 3;
  runtime.camera.position.set(center.x + distance * 0.18, center.y + distance * 0.1, center.z + distance);
  runtime.camera.lookAt(center);
  runtime.camera.updateProjectionMatrix();

  runtime.orbitTarget = center.clone();
  runtime.minCameraDistance = distance * 0.45;
  runtime.maxCameraDistance = distance * 3.0;
  runtime.controls.target.copy(center);
  runtime.controls.minDistance = runtime.minCameraDistance;
  runtime.controls.maxDistance = runtime.maxCameraDistance;
  runtime.controls.update();
}

async function initializeMeshPanel(panel) {
  const { THREE, OrbitControls } = await ensureMeshLib();
  const renderer = new THREE.WebGLRenderer({ canvas: panel.canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 2000);
  const controls = new OrbitControls(camera, panel.canvas);
  controls.enablePan = false;
  controls.enableDamping = false;
  controls.enabled = false;
  controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
  controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
  controls.touches.ONE = THREE.TOUCH.ROTATE;

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
  keyLight.position.set(0.2, 0.3, 1.0);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xd7f7ff, 0.45);
  fillLight.position.set(-0.6, 0.1, 0.6);
  scene.add(fillLight);

  const hemis = {};
  const group = new THREE.Group();
  scene.add(group);

  const layout = {};
  for (const hemi of ["lh", "rh"]) {
    const hemiData = panel.subject.hemis[hemi];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(hemiData.displayCoords ?? hemiData.coords, 3));
    const colorAttr = new THREE.Uint8BufferAttribute(new Uint8Array(hemiData.baseColors), 3, true);
    geometry.setAttribute("color", colorAttr);
    geometry.setIndex(new THREE.BufferAttribute(hemiData.faces, 1));
    geometry.computeVertexNormals();
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.FrontSide,
      shininess: 18,
      specular: 0x26313a,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.hemi = hemi;

    const box = new THREE.Box3().setFromBufferAttribute(geometry.getAttribute("position"));
    const center = box.getCenter(new THREE.Vector3());
    const boundaryGeometry = new THREE.BufferGeometry();
    boundaryGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(buildParcelBoundaryPositions(hemiData, center), 3)
    );
    const boundaryMaterial = new THREE.LineBasicMaterial({
      color: 0x061019,
      transparent: true,
      opacity: 0.74,
      depthTest: true,
      depthWrite: false,
    });
    const boundaryLine = new THREE.LineSegments(boundaryGeometry, boundaryMaterial);
    boundaryLine.renderOrder = 3;

    group.add(mesh);
    group.add(boundaryLine);
    layout[hemi] = {
      center,
      size: box.getSize(new THREE.Vector3()),
    };
    hemis[hemi] = { mesh, geometry, material, colorAttr, boundaryLine, boundaryGeometry, boundaryMaterial };
  }

  const gap = Math.max(layout.lh.size.x, layout.rh.size.x) * 0.16;
  const xOffsets = {
    lh: -(layout.lh.size.x * 0.5 + gap * 0.5),
    rh: layout.rh.size.x * 0.5 + gap * 0.5,
  };
  for (const hemi of ["lh", "rh"]) {
    const { center } = layout[hemi];
    hemis[hemi].mesh.position.set(xOffsets[hemi] - center.x, -center.y, -center.z);
    hemis[hemi].boundaryLine.position.copy(hemis[hemi].mesh.position);
  }

  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(2.3, 20, 20),
    new THREE.MeshBasicMaterial({ color: 0xff7446 })
  );
  marker.visible = false;
  group.add(marker);

  const groupBox = new THREE.Box3().setFromObject(group);
  const center = groupBox.getCenter(new THREE.Vector3());
  group.position.set(-center.x, -center.y, -center.z);
  controls.addEventListener("change", () => requestPanelRender(panel));

  panel.meshRuntime = {
    THREE,
    renderer,
    scene,
    camera,
    controls,
    group,
    hemis,
    selectionMarker: marker,
    orbitTarget: new THREE.Vector3(),
    minCameraDistance: 1,
    maxCameraDistance: 2000,
    hasInitialPose: false,
    canvasCssWidth: 0,
    canvasCssHeight: 0,
    frameHandle: 0,
  };

  fitMeshCamera(panel);
}

async function buildPanels(dataset) {
  for (const panel of state.panels) {
    disposePanelRuntime(panel);
  }
  panelGrid.innerHTML = "";
  panelGrid.classList.remove("is-single-mesh");
  panelGrid.classList.toggle("is-cortex-grid", dataset.meta.id === "cortex");
  state.panels = [];

  const visibleSubjects = dataset.subjects.map((subject, idx) => ({
    subject: isCortexSliceMode(dataset) ? subject.sliceSubject : subject,
    idx,
  }));

  visibleSubjects.forEach(({ subject, idx }) => {
    const node = panelTemplate.content.firstElementChild.cloneNode(true);
    const canvas = node.querySelector("canvas");
    const shell = node.querySelector(".slice-shell");
    const title = node.querySelector(".panel-title");
    const meta = node.querySelector(".panel-meta");
    const badge = node.querySelector(".panel-badge");
    const note = node.querySelector(".slice-note");
    const bufferCanvas = isCortexMeshMode(dataset) ? null : document.createElement("canvas");
    if (bufferCanvas) {
      bufferCanvas.width = subject.width;
      bufferCanvas.height = subject.height;
    }
    if (isCortexMeshMode(dataset)) {
      canvas.width = 1200;
      canvas.height = 640;
    } else {
      canvas.width = subject.width;
      canvas.height = subject.height;
    }
    title.textContent = subjectLabel(idx);
    meta.textContent = isCortexMeshMode(dataset) ? "Inflated 3D mesh" : `Axial slice z=${subject.slice}`;
    badge.textContent = isCortexMeshMode(dataset) ? "Parcellation" : "Preview";
    canvas.style.cursor = isCortexMeshMode(dataset) ? "grab" : "crosshair";
    canvas.title =
      isCortexMeshMode(dataset)
        ? "Drag to rotate. Wheel to zoom. Click cortex to pick."
        : "Drag to scrub a reference voxel";

    const panel = {
      node,
      canvas,
      shell,
      title,
      meta,
      badge,
      note,
      bufferCanvas,
      subject,
      idx,
      viewState: null,
      meshRuntime: null,
    };

    if (isCortexMeshMode(dataset)) {
      const handleMeshWheel = (event) => {
        event.preventDefault();
        event.stopPropagation();
        const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
        zoomMeshPanel(panel, delta);
      };
      canvas.addEventListener("pointerdown", (event) => handlePanelPointerDown(event, panel));
      canvas.addEventListener("pointermove", (event) => handlePanelPointerMove(event, panel));
      canvas.addEventListener("pointerup", (event) => handlePanelPointerUp(event, panel));
      canvas.addEventListener("pointercancel", (event) => handlePanelPointerUp(event, panel));
      node.addEventListener("wheel", handleMeshWheel, { passive: false, capture: true });
      canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    } else {
      canvas.addEventListener("pointerdown", (event) => handlePanelPointerDown(event, panel));
      canvas.addEventListener("pointermove", (event) => handlePanelPointerMove(event, panel));
      canvas.addEventListener(
        "wheel",
        (event) => {
          event.preventDefault();
          setZoom(state.zoom + (event.deltaY < 0 ? 0.15 : -0.15));
        },
        { passive: false }
      );
    }

    panelGrid.appendChild(node);
    state.panels.push(panel);
  });

  if (isCortexMeshMode(dataset)) {
    await Promise.all(state.panels.map((panel) => initializeMeshPanel(panel)));
  }

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

function pickSliceReference(panel, coords) {
  const dataset = getActiveDataset();
  const flat = coords.y * panel.subject.width + coords.x;
  const focusInfo = getActiveMaskInfo(panel.subject);
  if (!focusInfo.mask[flat]) {
    statusText.textContent = state.activeRegionId
      ? `Pick inside ${getRegionLabel(dataset, state.activeRegionId)}.`
      : "Pick inside brain mask.";
    return false;
  }

  state.selected = {
    datasetId: state.datasetId,
    subjectIndex: panel.idx,
    kind: "slice",
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
  return true;
}

function clearSelection(statusOverride = null) {
  state.selected = null;
  state.meshPickedLabel = "";
  invalidateSimilarityCache();
  pickedText.textContent = isCortexMeshMode()
    ? ""
    : "No reference voxel selected. Drag or click a slice to project voxel homology across subjects.";
  if (statusOverride) {
    statusText.textContent = statusOverride;
  }
  renderPanels();
}

function handlePanelPointerDown(event, panel) {
  const dataset = getActiveDataset();
  if (!dataset) {
    return;
  }
  if (event.button !== 0) {
    return;
  }

  if (isCortexMeshMode(dataset)) {
    event.preventDefault();
    state.meshClickCandidate = {
      panelIdx: panel.idx,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      moved: false,
    };
    panel.canvas.setPointerCapture?.(event.pointerId);
    panel.canvas.style.cursor = "grabbing";
    return;
  }

  if (dataset.featuresReadyCount < dataset.featuresTotal) {
    statusText.textContent = loadingStatusText(dataset, "Feature data still loading");
    return;
  }

  state.dragPicking = { panelIdx: panel.idx, pointerId: event.pointerId };
  panel.canvas.setPointerCapture?.(event.pointerId);
  const coords = getCanvasPixel(event, panel);
  if (coords) {
    pickSliceReference(panel, coords);
  }
}

function handlePanelPointerMove(event, panel) {
  const dataset = getActiveDataset();
  if (isCortexMeshMode(dataset)) {
    const candidate = state.meshClickCandidate;
    if (candidate && candidate.panelIdx === panel.idx && candidate.pointerId === event.pointerId) {
      event.preventDefault();
      const totalDx = event.clientX - candidate.startX;
      const totalDy = event.clientY - candidate.startY;
      const stepDx = event.clientX - candidate.lastX;
      const stepDy = event.clientY - candidate.lastY;
      candidate.lastX = event.clientX;
      candidate.lastY = event.clientY;
      if (Math.hypot(totalDx, totalDy) > 4) {
        candidate.moved = true;
      }
      if (event.buttons & 1) {
        rotateMeshPanel(panel, stepDx, stepDy);
      }
    }
    return;
  }

  if (
    !state.dragPicking ||
    state.dragPicking.panelIdx !== panel.idx ||
    state.dragPicking.pointerId !== event.pointerId ||
    !(event.buttons & 1)
  ) {
    return;
  }

  if (!dataset) {
    return;
  }

  const coords = getCanvasPixel(event, panel);
  if (coords) {
    pickSliceReference(panel, coords);
  }
}

function handlePanelPointerUp(event, panel) {
  const dataset = getActiveDataset();
  if (!isCortexMeshMode(dataset)) {
    state.dragPicking = null;
    return;
  }

  const candidate = state.meshClickCandidate;
  if (!candidate || candidate.panelIdx !== panel.idx || candidate.pointerId !== event.pointerId) {
    return;
  }

  panel.canvas.releasePointerCapture?.(event.pointerId);
  panel.canvas.style.cursor = "grab";
  state.meshClickCandidate = null;
  if (!candidate.moved) {
    pickMeshReference(event, panel);
  }
}

function updateSelectionSummary() {
  const dataset = getActiveDataset();
  if (!dataset) {
    selectionSummary.textContent = "";
    return;
  }
  if (isCortexMeshMode(dataset)) {
    selectionSummary.textContent = "";
    return;
  }

  const parts = [
    dataset.meta.name,
    `${state.activeSimilarityGroups.size}/${dataset.meta.similarityGroups.length} families`,
  ];
  if (!isCortexMeshMode(dataset)) {
    parts.splice(1, 0, getRegionLabel(dataset, state.activeRegionId));
  }
  if (state.selected && state.selected.datasetId === state.datasetId) {
    if (state.selected.kind === "mesh") {
      parts.push(`Ref: ${subjectLabel(state.selected.subjectIndex)} ${state.selected.hemi.toUpperCase()}`);
    } else {
      parts.push(`Ref: ${subjectLabel(state.selected.subjectIndex)}`);
    }
  }
  selectionSummary.textContent = parts.join(" · ");
}

function updateStats() {
  const dataset = getActiveDataset();
  if (!dataset || !state.meta) {
    return;
  }
  const cortexMode = isCortexDataset(dataset);
  const meshMode = isCortexMeshMode(dataset);

  const healthyMeta = state.meta.datasets.find((item) => item.id === "healthy") ?? null;
  const tumorMeta = state.meta.datasets.find((item) => item.id === "tumor") ?? null;
  const deliverySummary =
    dataset.meta.deliverySummary ||
    healthyMeta?.deliverySummary ||
    tumorMeta?.deliverySummary ||
    "";

  featureCount.textContent = String(dataset.meta.featureDim);
  datasetDescription.textContent = dataset.meta.description;
  viewerTitle.textContent =
    meshMode
      ? `Homology Maps / ${formatParcellationName(dataset.meta)} Parcellation`
      : "Voxel Homology Maps";
  activeFamiliesText.textContent = `${state.activeSimilarityGroups.size} of ${
    dataset.meta.similarityGroups.length
  } contrast families active.`;
  contrastLabel.textContent = getDisplayOption(dataset.meta).label;
  interactionHint.textContent =
    dataset.meta.interactionHint ||
    (meshMode
      ? "Drag the 3D cortex to rotate it. Wheel to zoom. Click once to pick a cortical parcel."
      : "Drag across any subject slice to scrub the reference voxel continuously.");

  if (dataset.meta.id === "healthy") {
    healthyPrepText.textContent = healthyMeta?.preprocessingSummary ?? "";
    healthyPrepText.hidden = !healthyMeta?.preprocessingSummary;
    tumorPrepText.textContent = tumorMeta?.preprocessingSummary ?? "";
    tumorPrepText.hidden = !tumorMeta?.preprocessingSummary;
  } else if (dataset.meta.id === "tumor") {
    healthyPrepText.textContent = tumorMeta?.preprocessingSummary ?? "";
    healthyPrepText.hidden = !tumorMeta?.preprocessingSummary;
    tumorPrepText.hidden = true;
  } else {
    healthyPrepText.textContent = dataset.meta.preprocessingSummary ?? "";
    healthyPrepText.hidden = !dataset.meta.preprocessingSummary;
    tumorPrepText.hidden = true;
  }
  deliveryText.textContent = deliverySummary;
  deliveryText.hidden = !deliverySummary;

  const isMesh = meshMode;
  contrastControlStack.hidden = isMesh || cortexMode;
  zoomControlStack.hidden = isMesh;
  regionStripCard.hidden = isMesh || dataset.meta.regions.length === 0 || cortexMode;
  similarityStripCard.hidden = dataset.meta.similarityGroups.length === 0;
  cortexViewToggle.hidden = !cortexMode;
  cortexView2dBtn.classList.toggle("is-active", cortexMode && state.cortexViewMode === "slice");
  cortexView3dBtn.classList.toggle("is-active", cortexMode && state.cortexViewMode === "mesh");
}

function renderRegionControls() {
  const dataset = getActiveDataset();
  regionControls.innerHTML = "";
  if (!dataset) {
    return;
  }
  if (!dataset.meta.regions.length || isCortexMeshMode(dataset) || isCortexDataset(dataset)) {
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

async function setCortexViewMode(nextMode) {
  const dataset = getActiveDataset();
  if (!dataset || !isCortexDataset(dataset)) {
    return;
  }
  if (nextMode !== "slice" && nextMode !== "mesh") {
    return;
  }
  if (state.cortexViewMode === nextMode) {
    return;
  }

  state.cortexViewMode = nextMode;
  state.selected = null;
  state.meshPickedLabel = "";
  invalidateSimilarityCache();
  state.displayIndex = firstFeatureDisplayIndex(dataset.meta);

  syncContrastControls();
  syncZoomControls();
  updateStats();
  updateSelectionSummary();
  statusText.textContent =
    nextMode === "mesh"
      ? "Preparing 3D cortical meshes..."
      : "Preparing 2D cortical slices...";

  await buildPanels(dataset);
  renderRegionControls();
  renderSimilarityControls();
  syncContrastControls();
  syncZoomControls();
  updateStats();
  updateSelectionSummary();

  statusText.textContent =
    dataset.featuresReadyCount < dataset.featuresTotal
      ? loadingStatusText(dataset)
      : isCortexMeshMode(dataset)
      ? "Drag the cortex to rotate it. Click one cortical parcel to choose a reference."
      : "Drag any subject panel to choose a reference voxel.";
}

function renderDatasetTabs() {
  datasetTabs.innerHTML = "";
  const tabOrder = new Map([
    ["healthy", 0],
    ["cortex", 1],
    ["tumor", 2],
  ]);
  const datasets = [...state.meta.datasets].sort(
    (a, b) => (tabOrder.get(a.id) ?? 99) - (tabOrder.get(b.id) ?? 99)
  );

  for (const dataset of datasets) {
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
  if (isCortexMeshMode(dataset)) {
    contrastSlider.min = "0";
    contrastSlider.max = "0";
    contrastSlider.value = "0";
    contrastLabel.textContent = "Inflated cortical surface";
    return;
  }
  contrastSlider.min = "0";
  contrastSlider.max = String(dataset.meta.displayOptions.length - 1);
  contrastSlider.value = String(state.displayIndex);
  contrastLabel.textContent = getDisplayOption(dataset.meta).label;
}

function syncZoomControls() {
  if (isCortexMeshMode()) {
    zoomValue.textContent = "orbit";
    return;
  }
  zoomSlider.value = String(Math.round(state.zoom * 100));
  zoomValue.textContent = `${state.zoom.toFixed(1)}x`;
}

function setZoom(nextZoom) {
  if (isCortexMeshMode()) {
    return;
  }
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
  if (dataset.meta.viewerKind === "mesh") {
    const subject = dataset.subjects[idx];
    const subjectMeta = dataset.meta.subjects[idx];
    const manifestUrl = joinUrl(baseDir, subjectMeta.mesh_manifest_file);
    const manifestDir = dirnameUrl(manifestUrl);
    await Promise.all(
      ["lh", "rh"].map(async (hemi) => {
        const hemiData = subject.hemis[hemi];
        const sampleLength = hemiData.sampleCount * dataset.meta.featureDim;
        hemiData.sampleFeatures = await loadTypedArray(
          joinUrl(manifestDir, hemiData.files.sample_features),
          dataset.meta.dtype.feature,
          sampleLength
        );
      })
    );
    const sliceSubject = subject.sliceSubject;
    const featureLength = sliceSubject.pixelCount * dataset.meta.featureDim;
    const features = await loadTypedArray(
      joinUrl(baseDir, sliceSubject.featureFile),
      dataset.meta.dtype.feature,
      featureLength
    );
    const norms = new Float32Array(sliceSubject.pixelCount);
    for (let flat = 0; flat < sliceSubject.pixelCount; flat++) {
      let norm2 = 0;
      const base = flat * sliceSubject.k;
      for (let j = 0; j < sliceSubject.k; j++) {
        const value = features[base + j];
        norm2 += value * value;
      }
      norms[flat] = Math.sqrt(norm2);
    }
    sliceSubject.features = features;
    sliceSubject.norms = norms;
    sliceSubject.featuresReady = true;
    subject.featuresReady = true;
    return;
  }

  const subject = dataset.subjects[idx];
  const subjectMeta = dataset.meta.subjects[idx];
  const featureLength = subject.pixelCount * dataset.meta.featureDim;
  const features = await loadTypedArray(joinUrl(baseDir, subjectMeta.feature_file), dataset.meta.dtype.feature, featureLength);
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
    const concurrency =
      dataset.meta.viewerKind === "mesh"
        ? Math.min(dataset.subjects.length, 4)
        : Math.min(2, dataset.subjects.length);
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
            statusText.textContent =
              isCortexMeshMode(dataset)
                ? "Drag the cortex to rotate it. Click one cortical parcel to choose a reference."
                : "Drag any subject panel to choose a reference voxel.";
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
  state.meshClickCandidate = null;
  state.meshPickedLabel = "";
  state.activeSimilarityGroups = new Set(dataset.meta.similarityGroups.map((group) => group.id));
  state.displayIndex = isCortexDataset(dataset)
    ? firstFeatureDisplayIndex(dataset.meta)
    : findDisplayIndex(dataset.meta, dataset.meta.defaultDisplay);
  state.zoom = 1;
  pickedText.textContent =
    isCortexMeshMode(dataset)
      ? ""
      : "No reference voxel selected. Drag or click a slice to project voxel homology across subjects.";
}

async function activateDataset(datasetId) {
  const datasetMeta = state.meta.datasets.find((dataset) => dataset.id === datasetId);
  if (!datasetMeta) {
    throw new Error(`Unknown dataset: ${datasetId}`);
  }

  state.datasetId = datasetId;
  state.cortexViewMode = "slice";
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

  await buildPanels(dataset);
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
    statusText.textContent =
      isCortexMeshMode(dataset)
        ? "Drag the cortex to rotate it. Click one cortical parcel to choose a reference."
        : "Drag any subject panel to choose a reference voxel.";
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
cortexView2dBtn.addEventListener("click", () => {
  // Keep cortical controls simple: one click swaps between 2D and 3D on the same page.
  setCortexViewMode("slice").catch((err) => {
    statusText.textContent = "Failed to switch cortical view.";
    pickedText.textContent = String(err.message || err);
  });
});
cortexView3dBtn.addEventListener("click", () => {
  setCortexViewMode("mesh").catch((err) => {
    statusText.textContent = "Failed to switch cortical view.";
    pickedText.textContent = String(err.message || err);
  });
});

alphaSlider.value = "86";
syncZoomControls();
window.addEventListener("resize", syncLegendSizing);
window.addEventListener("pointerup", () => {
  state.dragPicking = null;
});
window.addEventListener("pointercancel", () => {
  state.dragPicking = null;
});

init();
