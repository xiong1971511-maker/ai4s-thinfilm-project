"""Validate generated thin-film dataset CSV and associated metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "dataset.csv"
METADATA_PATH = DATA_DIR / "metadata.json"

EXPECTED_SAMPLE_COUNT = 5000
EXPECTED_INPUT_DIM = 4
EXPECTED_OUTPUT_DIM = 41


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {METADATA_PATH}")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    if not header:
        raise ValueError("Dataset CSV header is empty.")

    thickness_columns = [name for name in header if name.startswith("d") and name.endswith("_nm")]
    reflectance_columns = [name for name in header if name.startswith("reflectance_") and name.endswith("_nm")]

    if len(thickness_columns) != EXPECTED_INPUT_DIM:
        raise ValueError(f"Expected {EXPECTED_INPUT_DIM} thickness columns, found {len(thickness_columns)}.")
    if len(reflectance_columns) != EXPECTED_OUTPUT_DIM:
        raise ValueError(f"Expected {EXPECTED_OUTPUT_DIM} reflectance columns, found {len(reflectance_columns)}.")
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SAMPLE_COUNT} rows, found {len(rows)}.")

    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != EXPECTED_INPUT_DIM + EXPECTED_OUTPUT_DIM:
        raise ValueError(
            f"Expected a 2D dataset with shape ({EXPECTED_SAMPLE_COUNT}, {EXPECTED_INPUT_DIM + EXPECTED_OUTPUT_DIM}), "
            f"found {data.shape}."
        )

    thicknesses = data[:, :EXPECTED_INPUT_DIM]
    spectra = data[:, EXPECTED_INPUT_DIM:]

    if not np.all(np.isfinite(thicknesses)):
        raise ValueError("Thickness values contain NaN or Inf.")
    if not np.all(np.isfinite(spectra)):
        raise ValueError("Reflectance values contain NaN or Inf.")

    if not np.all((thicknesses >= 40.0) & (thicknesses <= 180.0)):
        raise ValueError("Thickness values fall outside the valid 40–180 nm range.")
    if not np.all((spectra >= 0.0) & (spectra <= 1.0)):
        raise ValueError("Reflectance values fall outside the physical range [0,1].")

    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    split_sizes = metadata.get("split_sizes", {})
    train_size = int(split_sizes.get("train", 0))
    validation_size = int(split_sizes.get("validation", 0))
    test_size = int(split_sizes.get("test", 0))

    if train_size + validation_size + test_size != EXPECTED_SAMPLE_COUNT:
        raise ValueError("Split sizes do not sum to the expected sample count.")

    summary = {
        "sample_count": int(data.shape[0]),
        "input_dim": int(thicknesses.shape[1]),
        "output_dim": int(spectra.shape[1]),
        "thickness_range_nm": [float(thicknesses.min()), float(thicknesses.max())],
        "reflectance_range": [float(spectra.min()), float(spectra.max())],
        "all_reflectances_bounded": bool(np.all((spectra >= 0.0) & (spectra <= 1.0))),
        "split_sizes": {"train": train_size, "validation": validation_size, "test": test_size},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
