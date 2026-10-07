"""Generate the assignment's fixed 5000-row TMM dataset."""

import argparse
import json
import platform

import numpy as np

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.dataset import generate_dataset, save_dataset, split_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/thinfilm_dataset.npz",
        help="destination NPZ file (default: %(default)s)",
    )
    args = parser.parse_args()

    config = STUDY_CONFIG
    thicknesses_nm, spectra = generate_dataset(sample_count=5000, config=config)
    splits = split_indices(5000, config.seed, (4000, 500, 500))
    metadata = {
        "seed": config.seed,
        "design_seed": config.design_seed,
        "target_wavelength_nm": config.target_wavelength_nm,
        "sample_count": 5000,
        "split_sizes": {"train": 4000, "validation": 500, "test": 500},
        "layer_sequence": "Air/H/L/H/L/Glass",
        "layer_indices": config.layer_indices,
        "substrate_index": config.substrate_index,
        "thickness_range_nm": [config.thickness_min_nm, config.thickness_max_nm],
        "wavelengths_nm": [config.wavelength_start_nm, config.wavelength_stop_nm,
                            config.wavelength_step_nm],
        "generation_method": "uniform continuous thickness sampling + repository TMM",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    destination = save_dataset(
        args.output,
        thicknesses_nm,
        spectra,
        splits,
        metadata=metadata,
        config=config,
    )
    report = {
        "status": "success",
        "summary": "Generated 5000 TMM spectra with a fixed 4000/500/500 split",
        "next_actions": ["Review dataset summary, then run scripts/train_mlp.py"],
        "artifacts": [str(destination)],
        "samples": len(thicknesses_nm),
        "wavelength_points_per_spectrum": len(config.wavelengths_nm),
        "split_sizes": [len(indices) for indices in splits],
        "thickness_range_nm": [float(np.min(thicknesses_nm)), float(np.max(thicknesses_nm))],
        "reflectance_range": [float(np.min(spectra)), float(np.max(spectra))],
        "all_reflectances_finite": bool(np.isfinite(spectra).all()),
        "all_reflectances_bounded": bool(((spectra >= 0.0) & (spectra <= 1.0)).all()),
        "metadata": metadata,
    }
    print(json.dumps(report, indent=2))
    if not report["all_reflectances_finite"] or not report["all_reflectances_bounded"]:
        raise SystemExit("Dataset reflectance validation failed")


if __name__ == "__main__":
    main()
