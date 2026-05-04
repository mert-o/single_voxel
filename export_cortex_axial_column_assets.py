#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cortical_sheet_tool.functions import sample_ribbon_voxel_ids, surface_tkr_to_world_transform
from nibabel.freesurfer.io import read_geometry


SUBJECTS = [
    "YHC_10_lncguay",
    "YHC_03_cyumllh",
    "OHC_17_mmixjcu",
    "OHC_11_fxdvnzc",
]
TARGET_SLICES = {
    "YHC_10_lncguay": 255,
    "YHC_03_cyumllh": 259,
    "OHC_17_mmixjcu": 253,
    "OHC_11_fxdvnzc": 252,
}

DATA_ROOT = Path("/home/m/HDD/nl_subjects")
FS_ROOT = Path("/home/m/test_subj_folder")
DEMO_DIR = Path(__file__).resolve().parent
OUT_DIR = DEMO_DIR / "data" / "cortex"
ROTATE_K = 1
MPRAGE_INDEX = 343
INVALID_SAMPLE = np.uint16(65535)
INVALID_LABEL = np.uint8(255)


def normalize_u8(values, mask):
    inside = values[mask > 0]
    lo, hi = np.quantile(inside, [0.01, 0.995])
    out = np.clip((values - lo) / (hi - lo + 1e-8), 0, 1)
    out = (out * 255).astype(np.uint8)
    out[mask == 0] = 0
    return out


def load_full_to_sample(subject_id, hemi):
    manifest = json.loads((OUT_DIR / f"{subject_id}_manifest.json").read_text())
    hemi_meta = manifest["hemis"][hemi]
    full_to_sample = np.fromfile(OUT_DIR / hemi_meta["files"]["full_to_sample"], dtype=np.uint16)
    labels = np.fromfile(OUT_DIR / hemi_meta["files"]["labels"], dtype=np.uint8)
    return full_to_sample, labels, hemi_meta["sample_count"]


def build_hemi_maps(subject_id, hemi, volume_shape, slab_mask, volume_affine, surf_to_world, dense_mask):
    white, _ = read_geometry(str(FS_ROOT / subject_id / "surf" / f"{hemi}.white"))
    pial, _ = read_geometry(str(FS_ROOT / subject_id / "surf" / f"{hemi}.pial"))
    ribbon_voxel_ids = sample_ribbon_voxel_ids(white, pial, surf_to_world, volume_affine, slab_mask, n_depths=7)
    full_to_sample, labels, sample_count = load_full_to_sample(subject_id, hemi)
    pial_world = nib.affines.apply_affine(surf_to_world, pial)
    pial_vox = nib.affines.apply_affine(np.linalg.inv(volume_affine), pial_world)

    vertex_ids = np.repeat(np.arange(pial.shape[0], dtype=np.int32), ribbon_voxel_ids.shape[1])
    voxel_ids = ribbon_voxel_ids.reshape(-1)
    valid = voxel_ids >= 0
    voxel_ids = voxel_ids[valid]
    vertex_ids = vertex_ids[valid]
    sample_ids = full_to_sample[vertex_ids]
    keep = sample_ids < sample_count
    voxel_ids = voxel_ids[keep]
    vertex_ids = vertex_ids[keep]
    sample_ids = sample_ids[keep]

    xyz = np.column_stack(np.unravel_index(voxel_ids, volume_shape)).astype(np.float32)
    pial_sel = pial_vox[vertex_ids]
    dist2 = ((xyz - pial_sel) ** 2).sum(axis=1)
    order = np.lexsort((dist2, voxel_ids))
    xyz = xyz[order]
    voxel_ids = voxel_ids[order]
    vertex_ids = vertex_ids[order]
    sample_ids = sample_ids[order]
    uniq = np.concatenate(([True], voxel_ids[1:] != voxel_ids[:-1]))
    xyz = xyz[uniq]
    voxel_ids = voxel_ids[uniq]
    vertex_ids = vertex_ids[uniq]
    sample_ids = sample_ids[uniq]

    dense_voxel_ids = np.flatnonzero(dense_mask.reshape(-1))
    dense_xyz = np.column_stack(np.unravel_index(dense_voxel_ids, volume_shape)).astype(np.float32)
    tree = cKDTree(xyz)
    _, nearest = tree.query(dense_xyz, k=1)
    sample_map = np.full(np.prod(volume_shape), INVALID_SAMPLE, dtype=np.uint16)
    label_map = np.full(np.prod(volume_shape), INVALID_LABEL, dtype=np.uint8)
    hemi_map = np.zeros(np.prod(volume_shape), dtype=np.uint8)
    sample_map[dense_voxel_ids] = sample_ids[nearest]
    label_map[dense_voxel_ids] = labels[vertex_ids[nearest]]
    hemi_map[dense_voxel_ids] = 1 if hemi == "lh" else 2
    return sample_map, label_map, hemi_map


def export_subject(subject_id):
    feature_img = nib.load(str(DATA_ROOT / subject_id / "new_data_original_space_poly2_residuals.nii.gz"))
    feature_vol = np.asarray(feature_img.dataobj, dtype=np.float32)
    mask_img = nib.load(str(DATA_ROOT / subject_id / "mask.nii.gz"))
    slab_mask = np.asarray(mask_img.dataobj).astype(bool)
    aseg = np.asarray(nib.load(str(DATA_ROOT / subject_id / "aseg_resampled.nii.gz")).dataobj)
    dense_hemi = np.zeros_like(aseg, dtype=np.uint8)
    dense_hemi[aseg == 3] = 1
    dense_hemi[aseg == 42] = 2
    dense_hemi[~slab_mask] = 0
    t1_vol = feature_vol[..., MPRAGE_INDEX]

    surf_to_world = surface_tkr_to_world_transform(nib.load(str(FS_ROOT / subject_id / "mri" / "orig.mgz")))
    lh_sample, lh_label, lh_hemi = build_hemi_maps(subject_id, "lh", slab_mask.shape, slab_mask, feature_img.affine, surf_to_world, dense_hemi == 1)
    rh_sample, rh_label, rh_hemi = build_hemi_maps(subject_id, "rh", slab_mask.shape, slab_mask, feature_img.affine, surf_to_world, dense_hemi == 2)

    sample_dense = np.where(lh_hemi.reshape(slab_mask.shape) > 0, lh_sample.reshape(slab_mask.shape), rh_sample.reshape(slab_mask.shape))
    label_dense = np.where(lh_hemi.reshape(slab_mask.shape) > 0, lh_label.reshape(slab_mask.shape), rh_label.reshape(slab_mask.shape))
    hemi_dense = lh_hemi.reshape(slab_mask.shape) + rh_hemi.reshape(slab_mask.shape)
    ribbon_mask = dense_hemi > 0
    slice_idx = TARGET_SLICES[subject_id]

    base = np.rot90(t1_vol[:, :, slice_idx], k=ROTATE_K)
    mask2d = np.rot90(ribbon_mask[:, :, slice_idx].astype(np.uint8), k=ROTATE_K)
    hemi2d = np.rot90(hemi_dense[:, :, slice_idx], k=ROTATE_K).astype(np.uint8)
    label2d = np.rot90(label_dense[:, :, slice_idx], k=ROTATE_K).astype(np.uint8)
    sample2d = np.rot90(sample_dense[:, :, slice_idx], k=ROTATE_K).astype(np.uint16)

    base = normalize_u8(base, mask2d)
    sample2d[mask2d == 0] = INVALID_SAMPLE
    label2d[mask2d == 0] = INVALID_LABEL
    hemi2d[mask2d == 0] = 0

    prefix = f"{subject_id}_axial"
    files = {
        "base": f"{prefix}_base.u8",
        "mask": f"{prefix}_mask.u8",
        "hemi": f"{prefix}_hemi.u8",
        "label": f"{prefix}_label.u8",
        "sample": f"{prefix}_sample.u16",
    }
    base.ravel(order="C").tofile(OUT_DIR / files["base"])
    mask2d.ravel(order="C").tofile(OUT_DIR / files["mask"])
    hemi2d.ravel(order="C").tofile(OUT_DIR / files["hemi"])
    label2d.ravel(order="C").tofile(OUT_DIR / files["label"])
    sample2d.ravel(order="C").tofile(OUT_DIR / files["sample"])
    manifest = {"subject_id": subject_id, "width": int(mask2d.shape[1]), "height": int(mask2d.shape[0]), "slice": slice_idx, "files": files}
    (OUT_DIR / f"{subject_id}_axial_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sid in SUBJECTS:
        export_subject(sid)


if __name__ == "__main__":
    main()
