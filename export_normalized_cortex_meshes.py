#!/usr/bin/env python3
"""Export normalized cortical mesh data for the browser demo.

The browser should not load every vertex feature vector because that would make
the demo very heavy.  Instead we keep the visible mesh at full resolution and
store a smaller set of feature anchors for cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from nibabel.freesurfer.io import read_annot
from scipy.spatial import cKDTree


SUBJECTS = [
    "YHC_10_lncguay",
    "YHC_03_cyumllh",
    "OHC_17_mmixjcu",
    "OHC_11_fxdvnzc",
]

SOURCE_DIR = Path("/home/m/HDD2/vox2/cortical_sheet_tool")
FREESURFER_DIR = Path("/home/m/test_subj_folder")
OUT_DIR = Path(__file__).resolve().parent / "data" / "cortex"
PARCELLATION = "destrieux"
ANNOT_NAME = "aparc.a2009s"
SAMPLE_COUNT = 8192
RNG_SEED = 7


def annotation_lookup(subject_id: str, hemi: str) -> tuple[np.ndarray, list[str]]:
    """Load FreeSurfer colors and parcel names for one hemisphere."""
    annot_path = FREESURFER_DIR / subject_id / "label" / f"{hemi}.{ANNOT_NAME}.annot"
    _, ctab, names = read_annot(str(annot_path))
    colors = np.clip(ctab[:, :3], 0, 255).astype(np.uint8)
    decoded_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]
    return colors, decoded_names


def choose_anchor_vertices(coords: np.ndarray, valid_mask: np.ndarray, sample_count: int, seed: int) -> np.ndarray:
    """Choose deterministic feature anchors from vertices that have features."""
    rng = np.random.default_rng(seed)
    valid_ids = np.flatnonzero(valid_mask)
    n_samples = min(sample_count, valid_ids.size)
    sample_ids = rng.choice(valid_ids, size=n_samples, replace=False)
    return np.sort(sample_ids).astype(np.int32)


def remap_valid_mesh(
    coords: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    """Keep the full inflated mesh and mark which vertices have features."""
    valid_vertices = (counts > 0) & np.all(np.isfinite(features), axis=1)

    return {
        "coords": coords.astype(np.float32),
        "faces": faces.astype(np.uint32),
        "labels": labels.astype(np.int32),
        "valid": valid_vertices.astype(np.uint8),
        "features": np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    }


def export_hemi(subject_id: str, hemi: str, mesh_npz: np.lib.npyio.NpzFile, parcel_names: dict[str, str]) -> dict:
    """Write one hemisphere's binary files and return its manifest entry."""
    coords = mesh_npz[f"{hemi}_coords"]
    faces = mesh_npz[f"{hemi}_faces"].astype(np.int32)
    labels = mesh_npz[f"{hemi}_parcellation"]
    counts = mesh_npz[f"{hemi}_voxel_count"]
    features = mesh_npz[f"{hemi}_vertex_features"]

    visible = remap_valid_mesh(coords, faces, labels, counts, features)
    colors, names = annotation_lookup(subject_id, hemi)

    label_codes = np.where(visible["labels"] < 0, 0, visible["labels"]).astype(np.uint8)
    label_colors = colors[label_codes].astype(np.uint8)
    label_colors[visible["valid"] == 0] = np.array([145, 151, 151], dtype=np.uint8)

    for label_id, name in enumerate(names):
        parcel_names[f"{hemi}:{label_id}"] = name

    seed = RNG_SEED + SUBJECTS.index(subject_id) * 17 + (0 if hemi == "lh" else 1)
    sample_ids = choose_anchor_vertices(visible["coords"], visible["valid"].astype(bool), SAMPLE_COUNT, seed)
    tree = cKDTree(visible["coords"][sample_ids])
    _, full_to_sample = tree.query(visible["coords"], k=1)

    prefix = f"{subject_id}_{hemi}"
    files = {
        "coords": f"{prefix}_coords.f32",
        "faces": f"{prefix}_faces.u32",
        "colors": f"{prefix}_colors.u8",
        "labels": f"{prefix}_labels.u8",
        "valid": f"{prefix}_valid.u8",
        "full_to_sample": f"{prefix}_full_to_sample.u16",
        "sample_features": f"{prefix}_sample_features.f16",
    }

    visible["coords"].astype(np.float32).tofile(OUT_DIR / files["coords"])
    visible["faces"].astype(np.uint32).tofile(OUT_DIR / files["faces"])
    label_colors.astype(np.uint8).tofile(OUT_DIR / files["colors"])
    label_codes.astype(np.uint8).tofile(OUT_DIR / files["labels"])
    visible["valid"].astype(np.uint8).tofile(OUT_DIR / files["valid"])
    full_to_sample.astype(np.uint16).tofile(OUT_DIR / files["full_to_sample"])
    visible["features"][sample_ids].astype(np.float16).tofile(OUT_DIR / files["sample_features"])

    return {
        "vertex_count": int(visible["coords"].shape[0]),
        "face_count": int(visible["faces"].shape[0]),
        "sample_count": int(sample_ids.size),
        "files": files,
    }


def export_subject(subject_id: str, parcel_names: dict[str, str]) -> dict:
    """Export one subject manifest from the normalized mesh archive."""
    mesh_path = SOURCE_DIR / f"{subject_id}_{PARCELLATION}_cortical_mesh_ind.npz"
    with np.load(mesh_path, allow_pickle=False) as mesh_npz:
        feature_dim = int(mesh_npz["lh_vertex_features"].shape[1])
        manifest = {
            "subject_id": subject_id,
            "feature_dim": feature_dim,
            "parcellation": str(mesh_npz["parcellation"]),
            "annot_name": str(mesh_npz["annot_name"]),
            "normalization_mode": str(mesh_npz["normalization_mode"]),
            "block_weight_mode": str(mesh_npz["block_weight_mode"]),
            "hemis": {},
        }
        for hemi in ["lh", "rh"]:
            manifest["hemis"][hemi] = export_hemi(subject_id, hemi, mesh_npz, parcel_names)

    manifest_path = OUT_DIR / f"{subject_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def update_metadata(parcel_names: dict[str, str]) -> None:
    """Point the demo metadata at the refreshed Destrieux cortex dataset."""
    metadata_path = Path(__file__).resolve().parent / "data" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["default_dataset_id"] = "cortex"

    for dataset in metadata["datasets"]:
        if dataset["id"] != "cortex":
            continue
        dataset["name"] = "Cortical Homology"
        dataset["description"] = "Four subject-native inflated cortical meshes with volume-normalized, block-weighted ribbon signatures."
        dataset["dtype"]["valid"] = "uint8"
        dataset["parcellation"] = PARCELLATION
        dataset["annot_name"] = ANNOT_NAME
        dataset["preprocessing_summary"] = (
            "Cortex preprocessing: each 4D volume is normalized across all masked voxels first, feature blocks are weighted, "
            "the cortical ribbon is sampled to pial vertices, and cosine similarity is computed on those vertex signatures."
        )
        dataset["delivery_summary"] = (
            "Web demo delivery: full inflated meshes are shown. Cortex outside the sampled ribbon slab is gray, "
            "and sampled Destrieux parcels are colored. Each hemisphere stores 8192 feature anchors; "
            "similarity is painted back onto sampled cortex by nearest anchor."
        )
        dataset["interaction_hint"] = "Drag the 3D cortex to rotate it. Wheel to zoom. Click once to pick a cortical parcel."
        dataset["parcel_names"] = parcel_names

    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    """Refresh all cortex files used by the web demo."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parcel_names: dict[str, str] = {}
    for subject_id in SUBJECTS:
        manifest = export_subject(subject_id, parcel_names)
        print(f"exported {subject_id}: {manifest['hemis']['lh']['vertex_count']} LH vertices, {manifest['hemis']['rh']['vertex_count']} RH vertices")
    update_metadata(parcel_names)
    print(f"updated {OUT_DIR.parent / 'metadata.json'}")


if __name__ == "__main__":
    main()
