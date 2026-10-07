"""Generate the assignment dataset as CSV plus split metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.dataset import generate_dataset, split_indices


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "dataset.csv"
METADATA_PATH = DATA_DIR / "metadata.json"


def main() -> None:
    thicknesses_nm, spectra = generate_dataset(sample_count=5000, config=STUDY_CONFIG)
    train_indices, validation_indices, test_indices = split_indices(
        5000,
        STUDY_CONFIG.seed,
        (4000, 500, 500),
    )

    headers = ["d1_nm", "d2_nm", "d3_nm", "d4_nm"] + [
        f"reflectance_{wavelength_nm}_nm" for wavelength_nm in STUDY_CONFIG.wavelengths_nm
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row_index in range(thicknesses_nm.shape[0]):
            row = np.concatenate((thicknesses_nm[row_index], spectra[row_index]))
            writer.writerow(row.tolist())

    metadata = {
        "project": "AI4S Thin Film Project",
        "seed": int(STUDY_CONFIG.seed),
        "target_wavelength_nm": int(STUDY_CONFIG.target_wavelength_nm),
        "layer_sequence": "Air/H/L/H/L/Glass",
        "layer_indices": [float(value) for value in STUDY_CONFIG.layer_indices],
        "incident_index": float(STUDY_CONFIG.incident_index),
        "substrate_index": float(STUDY_CONFIG.substrate_index),
        "thickness_range_nm": [
            int(STUDY_CONFIG.thickness_min_nm),
            int(STUDY_CONFIG.thickness_max_nm),
        ],
        "wavelength_grid_nm": {
            "start": int(STUDY_CONFIG.wavelength_start_nm),
            "stop": int(STUDY_CONFIG.wavelength_stop_nm),
            "step": int(STUDY_CONFIG.wavelength_step_nm),
            "count": len(STUDY_CONFIG.wavelengths_nm),
        },
        "sample_count": 5000,
        "split_sizes": {"train": 4000, "validation": 500, "test": 500},
        "train_indices": train_indices.astype(int).tolist(),
        "validation_indices": validation_indices.astype(int).tolist(),
        "test_indices": test_indices.astype(int).tolist(),
        "output_format": {
            "rows": 5000,
            "input_columns": 4,
            "output_columns": 41,
        },
        "generation_method": "continuous uniform random thickness sampling + TMM reflectance spectrum",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Generated {CSV_PATH}")
    print(f"Generated {METADATA_PATH}")
    print(f"Samples={thicknesses_nm.shape[0]}")
    print(f"Inputs={thicknesses_nm.shape[1]}")
    print(f"Outputs={spectra.shape[1]}")
    print(f"Reflectance_range={float(np.min(spectra)), float(np.max(spectra))}")


if __name__ == "__main__":
    main()
