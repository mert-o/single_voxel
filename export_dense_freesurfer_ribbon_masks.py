#!/usr/bin/env python3
"""Export dense cortical ribbon masks for the browser demo.

The browser demo needs one 2D cortical ribbon mask per healthy subject.  The
best source is the already co-registered aseg file, because it lives in the same
space as the demo feature volume.  If that is not available, we fall back to the
native FreeSurfer ribbon and resample by affine.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


# The demo folder contains this script, the config, and the browser data folder.
DEMO_DIR = Path(__file__).resolve().parent

# The config records the anatomy sources for each healthy subject.
CONFIG_PATH = DEMO_DIR / "demo_config.json"

# The browser reads this metadata file at runtime.
METADATA_PATH = DEMO_DIR / "data" / "metadata.json"

# The region id is used in config, metadata, and the output filenames.
REGION_ID = "cortical_ribbon"

# This is the label shown in the anatomical focus buttons.
REGION_LABEL = "Cortical ribbon"


def rotate_slice(arr: np.ndarray, rotate_k: int) -> np.ndarray:
    """Rotate a 2D slice exactly like the main demo exporter."""
    if rotate_k % 4 == 0:
        return arr
    return np.rot90(arr, k=rotate_k, axes=(0, 1))


def downsample_indices(n_src: int, n_dst: int) -> np.ndarray:
    """Pick nearest-neighbor source indices for one image axis."""
    if n_src == n_dst:
        return np.arange(n_src, dtype=np.int64)
    return np.round(np.linspace(0, n_src - 1, n_dst)).astype(np.int64)


def downsample_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Downsample a 2D mask using the same nearest-neighbor rule as the demo."""
    rows = downsample_indices(arr.shape[0], out_h)
    cols = downsample_indices(arr.shape[1], out_w)
    return arr[rows][:, cols]


def output_name_for_subject(index: int, subject_name: str) -> str:
    """Build the same simple filename style used by the other demo masks."""
    slug = subject_name.lower().replace("-", "_")
    return f"healthy_s{index}_{slug}_{REGION_ID}.u8"


def cortical_ribbon_on_volume_grid(subject_config: dict, target_grid: tuple) -> np.ndarray:
    """Return the cortical ribbon on the same 3D grid as the feature volume."""
    region_sources = subject_config.get("region_sources", {})

    # Prefer aseg because the config already points to the co-registered aseg.
    if "aseg" in region_sources:
        source_img = nib.load(region_sources["aseg"])
    else:
        source_img = nib.load(region_sources[REGION_ID])

    # FreeSurfer labels 3 and 42 are left and right cerebral cortex.
    source_data = np.isin(np.asarray(source_img.dataobj), [3, 42]).astype(np.uint8)

    # If the source is already on the feature grid, use it directly.
    if source_img.shape[:3] == target_grid[0] and np.allclose(source_img.affine, target_grid[1]):
        return source_data

    # Otherwise resample the binary label image with nearest-neighbor labels.
    source_binary_img = nib.Nifti1Image(source_data, source_img.affine, source_img.header)
    return np.asarray(resample_from_to(source_binary_img, target_grid, order=0).dataobj)


def main() -> None:
    """Resample each FreeSurfer ribbon and write one 2D uint8 mask per subject."""
    config = json.loads(CONFIG_PATH.read_text())
    metadata = json.loads(METADATA_PATH.read_text())

    healthy_config = next(dataset for dataset in config["datasets"] if dataset["id"] == "healthy")
    healthy_meta = next(dataset for dataset in metadata["datasets"] if dataset["id"] == "healthy")

    if not any(region["id"] == REGION_ID for region in healthy_meta["regions"]):
        healthy_meta["regions"].append({"id": REGION_ID, "label": REGION_LABEL})

    out_h = int(healthy_meta["height"])
    out_w = int(healthy_meta["width"])

    for index, (subject_config, subject_meta) in enumerate(
        zip(healthy_config["subjects"], healthy_meta["subjects"]),
        start=1,
    ):
        slice_index = int(subject_config["slice"])
        rotate_k = int(subject_config.get("rotate_k", 0)) % 4

        volume_img = nib.load(subject_config["vol4d"])
        brain_mask_img = nib.load(subject_config["mask"])
        target_grid = (volume_img.shape[:3], volume_img.affine)
        ribbon_on_grid = cortical_ribbon_on_volume_grid(subject_config, target_grid)

        ribbon_slice = ribbon_on_grid[:, :, slice_index] != 0
        brain_slice = np.asarray(brain_mask_img.dataobj[:, :, slice_index]) != 0

        ribbon_slice = rotate_slice(ribbon_slice, rotate_k)
        brain_slice = rotate_slice(brain_slice, rotate_k)

        # Keep only ribbon voxels where the demo has feature values for similarity.
        ribbon_slice = ribbon_slice & brain_slice
        ribbon_downsampled = downsample_2d(ribbon_slice.astype(np.float32), out_h, out_w) >= 0.5

        output_name = output_name_for_subject(index, subject_meta["name"])
        output_path = DEMO_DIR / "data" / output_name
        ribbon_downsampled.astype(np.uint8).ravel(order="C").tofile(output_path)

        subject_meta.setdefault("region_files", {})[REGION_ID] = output_name

        print(
            f"{subject_meta['name']}: wrote {output_name} "
            f"with {int(ribbon_downsampled.sum())} dense ribbon pixels"
        )

    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
