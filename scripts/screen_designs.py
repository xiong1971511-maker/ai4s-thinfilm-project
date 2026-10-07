"""Rank 10000 new candidates with the MLP and verify the top 10 using TMM."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import torch

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.design import generate_candidates, rank_candidates, verify_spectra
from ai4s_thinfilm.model import MLPRegressor


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mlp_rank",
        "tmm_rank_within_mlp_top10",
        "candidate_index",
        "d1_nm",
        "d2_nm",
        "d3_nm",
        "d4_nm",
        "mlp_reflectance_at_target",
        "tmm_reflectance_at_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/thinfilm_dataset.npz")
    parser.add_argument("--checkpoint", default="models/mlp_train_4000.pt")
    parser.add_argument("--candidates", type=int, default=10_000)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.candidates < 10:
        parser.error("--candidates must be at least 10")

    config = STUDY_CONFIG
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        training_thicknesses = archive["thicknesses_nm"]
        wavelengths_nm = archive["wavelengths_nm"]
    if not np.array_equal(wavelengths_nm, np.asarray(config.wavelengths_nm)):
        raise ValueError("dataset wavelength grid does not match current study config")
    target_matches = np.flatnonzero(wavelengths_nm == config.target_wavelength_nm)
    if len(target_matches) != 1:
        raise ValueError("target wavelength must occur exactly once in the dataset grid")
    target_column = int(target_matches[0])

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    output_activation = checkpoint.get("output_activation", "linear")
    model = MLPRegressor(output_activation=output_activation)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(1)

    candidates = generate_candidates(args.candidates, config)
    existing_designs = {tuple(row) for row in training_thicknesses}
    candidate_dataset_matches = sum(tuple(row) in existing_designs for row in candidates)
    scale = config.thickness_max_nm - config.thickness_min_nm
    candidates_scaled = ((candidates - config.thickness_min_nm) / scale).astype(np.float32)
    predicted_spectra = np.empty(
        (len(candidates), len(config.wavelengths_nm)), dtype=np.float32
    )
    with torch.no_grad():
        for start in range(0, len(candidates), 1024):
            stop = min(start + 1024, len(candidates))
            predicted_spectra[start:stop] = model(
                torch.from_numpy(candidates_scaled[start:stop])
            ).cpu().numpy()
    if not np.isfinite(predicted_spectra).all():
        raise ValueError("MLP produced non-finite candidate predictions")

    predictions_path = output_dir / "design_screening_predictions.npz"
    np.savez_compressed(
        predictions_path,
        thicknesses_nm=candidates,
        predicted_reflectance=predicted_spectra,
        wavelengths_nm=wavelengths_nm,
        target_wavelength_nm=np.asarray(config.target_wavelength_nm),
        design_seed=np.asarray(config.design_seed),
    )

    outputs_by_objective: dict[str, object] = {}
    verified_spectra: dict[str, np.ndarray] = {}
    for objective in ("min", "max"):
        ranked_indices = rank_candidates(predicted_spectra, target_column, objective)
        mlp_top10 = ranked_indices[:10]
        tmm_spectra = verify_spectra(candidates[mlp_top10], config)
        verified_spectra[objective] = tmm_spectra
        true_targets = tmm_spectra[:, target_column]
        tmm_order_local = np.argsort(
            true_targets if objective == "min" else -true_targets,
            kind="stable",
        )
        tmm_rank_by_local_index = np.empty(10, dtype=np.int64)
        tmm_rank_by_local_index[tmm_order_local] = np.arange(1, 11)
        rows: list[dict[str, object]] = []
        for local_index, candidate_index in enumerate(mlp_top10):
            rows.append(
                {
                    "mlp_rank": local_index + 1,
                    "tmm_rank_within_mlp_top10": int(tmm_rank_by_local_index[local_index]),
                    "candidate_index": int(candidate_index),
                    "d1_nm": float(candidates[candidate_index, 0]),
                    "d2_nm": float(candidates[candidate_index, 1]),
                    "d3_nm": float(candidates[candidate_index, 2]),
                    "d4_nm": float(candidates[candidate_index, 3]),
                    "mlp_reflectance_at_target": float(
                        predicted_spectra[candidate_index, target_column]
                    ),
                    "tmm_reflectance_at_target": float(true_targets[local_index]),
                }
            )
        top10_path = output_dir / f"design_top10_{objective}.csv"
        _write_rows(top10_path, rows)
        top5_rows = [rows[int(local_index)] for local_index in tmm_order_local[:5]]
        top5_path = output_dir / f"design_top5_{objective}.csv"
        _write_rows(top5_path, top5_rows)
        outputs_by_objective[objective] = {
            "objective": "minimize target reflectance" if objective == "min"
                         else "maximize target reflectance",
            "mlp_top10_csv": str(top10_path),
            "tmm_verified_top5_csv": str(top5_path),
            "top5": top5_rows,
        }

    verified_path = output_dir / "design_top10_tmm_spectra.npz"
    np.savez_compressed(
        verified_path,
        wavelengths_nm=wavelengths_nm,
        min_objective_spectra=verified_spectra["min"],
        max_objective_spectra=verified_spectra["max"],
    )
    target_predictions = predicted_spectra[:, target_column]
    report = {
        "status": "success",
        "summary": (
            f"Screened new candidates with the {output_activation}-output "
            "4000-sample MLP and recalculated each objective's top 10 with TMM"
        ),
        "next_actions": ["Review the objective direction before using either top-five list"],
        "artifacts": [
            str(predictions_path),
            str(verified_path),
            str(output_dir / "design_top10_min.csv"),
            str(output_dir / "design_top10_max.csv"),
            str(output_dir / "design_top5_min.csv"),
            str(output_dir / "design_top5_max.csv"),
        ],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": "cpu",
        "model_output_activation": output_activation,
        "candidate_count": len(candidates),
        "exact_candidate_matches_in_training_dataset": candidate_dataset_matches,
        "design_seed": config.design_seed,
        "target_wavelength_nm": config.target_wavelength_nm,
        "candidate_thickness_range_nm": [
            float(candidates.min()),
            float(candidates.max()),
        ],
        "prediction_range_at_target": [
            float(target_predictions.min()),
            float(target_predictions.max()),
        ],
        "predictions_outside_physical_reflectance_bounds": int(
            np.count_nonzero((target_predictions < 0.0) | (target_predictions > 1.0))
        ),
        "objectives": outputs_by_objective,
    }
    report_path = output_dir / "design_screening_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
