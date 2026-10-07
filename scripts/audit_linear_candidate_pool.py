"""Audit the existing linear-model candidate pool with full TMM evaluation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.model import MLPRegressor
from ai4s_thinfilm.tmm import reflectance_spectrum
from make_figures import Svg, _header, _legend, _panel, _plot_axes

LINEAR_DIR = ROOT / "outputs_convergence_linear"
CHECKPOINT_PATH = ROOT / "models_convergence_linear" / "mlp_train_4000.pt"
METRICS_PATH = LINEAR_DIR / "mlp_metrics.json"
CANDIDATES_PATH = LINEAR_DIR / "design_screening_predictions.npz"
DATASET_PATH = ROOT / "data" / "thinfilm_dataset.npz"
OUTPUT_DIR = LINEAR_DIR / "global_tmm_audit"


def _load_top10(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _top_rows(
    order: np.ndarray,
    candidates: np.ndarray,
    predictions: np.ndarray,
    tmm_r700: np.ndarray,
    opposite_ranks: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for rank, index_value in enumerate(order[:5], start=1):
        index = int(index_value)
        rows.append({
            "candidate_index": index,
            "d1_nm": float(candidates[index, 0]),
            "d2_nm": float(candidates[index, 1]),
            "d3_nm": float(candidates[index, 2]),
            "d4_nm": float(candidates[index, 3]),
            "mlp_reflectance_at_target": float(predictions[index]),
            "tmm_reflectance_at_target": float(tmm_r700[index]),
            "tmm_rank_overall": rank,
            "tmm_rank_opposite_objective": int(opposite_ranks[index]),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_figure3(
    output_path: Path,
    wavelengths: np.ndarray,
    true_test: np.ndarray,
    predicted_test: np.ndarray,
    predicted_candidates: np.ndarray,
    tmm_spectra: np.ndarray,
    min_order: np.ndarray,
    max_order: np.ndarray,
    tmm_r700: np.ndarray,
) -> None:
    svg = Svg(1500, 1110)
    _header(
        svg,
        "03",
        "Linear predictions checked against full-pool TMM",
        "Held-out test  ·  all 10,000 candidates  ·  global physical ranking",
    )

    _panel(svg, 45, 105, 680, 400, "(a) Held-out test set at λ = 700 nm")
    target_column = int(np.flatnonzero(wavelengths == STUDY_CONFIG.target_wavelength_nm)[0])
    actual = true_test[:, target_column]
    predicted = predicted_test[:, target_column]
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    sx, sy = _plot_axes(
        svg, 65, 145, 640, 300, low, high, low, high,
        "TMM reflectance", "Linear MLP reflectance",
        [(low, f"{low:.2f}"), ((low + high) / 2, f"{(low + high) / 2:.2f}"), (high, f"{high:.2f}")],
        [(low, f"{low:.2f}"), ((low + high) / 2, f"{(low + high) / 2:.2f}"), (high, f"{high:.2f}")],
    )
    svg.line(sx(low), sy(low), sx(high), sy(high), "#94a3b8", 2, "6,5")
    for actual_value, predicted_value in zip(actual, predicted):
        svg.circle(sx(float(actual_value)), sy(float(predicted_value)), 2.6, "#087e8b")
    svg.text(85, 480, f"n = {len(actual)} held-out samples; test RMSE = 0.011375 (full spectrum).", 13, "#526579")

    _panel(svg, 755, 105, 700, 400, "(b) Predictions vs. TMM for the same full candidate pool")
    predicted_r700 = predicted_candidates[:, target_column]
    x_min = min(0.0, float(predicted_r700.min()))
    x_max = max(float(predicted_r700.max()), float(tmm_r700.max())) * 1.04
    y_min = min(0.0, float(tmm_r700.min()))
    y_max = float(tmm_r700.max()) * 1.04
    sx_all, sy_all = _plot_axes(
        svg, 775, 145, 660, 300, x_min, x_max, y_min, y_max,
        "Linear MLP prediction R(700 nm)", "Full TMM R(700 nm)",
        [(x_min, f"{x_min:.2f}"), (x_max, f"{x_max:.2f}")],
        [(y_min, f"{y_min:.2f}"), (y_max, f"{y_max:.2f}")],
    )
    for predicted_value, tmm_value in zip(predicted_r700, tmm_r700):
        svg.parts.append(
            f'<circle cx="{sx_all(float(predicted_value)):.1f}" '
            f'cy="{sy_all(float(tmm_value)):.1f}" r="2.0" '
            'fill="#64748b" fill-opacity="0.23"/>'
        )
    for indices, color in ((min_order[:5], "#087e8b"), (max_order[:5], "#d97706")):
        for index_value in indices:
            index = int(index_value)
            svg.circle(sx_all(float(predicted_r700[index])), sy_all(float(tmm_r700[index])), 4.2, color)
    _legend(svg, 1120, 155, [("Global TMM min Top 5", "#087e8b"), ("Global TMM max Top 5", "#d97706")])
    svg.text(790, 480, "Each gray point is one candidate; colored points are global TMM Top 5.", 12, "#526579")
    svg.text(
        790, 496,
        f"Min: #{int(min_order[0])}, R={tmm_r700[int(min_order[0])]:.7f}   "
        f"Max: #{int(max_order[0])}, R={tmm_r700[int(max_order[0])]:.7f}",
        12, "#334155", weight="bold",
    )

    _panel(svg, 45, 530, 1410, 520, "03C   Global TMM-best candidates: linear prediction and TMM spectrum")
    left, top, plot_width, plot_height = 125, 615, 1240, 295
    wavelength_min, wavelength_max = float(wavelengths[0]), float(wavelengths[-1])
    sx_spectrum = lambda value: left + (value - wavelength_min) / (wavelength_max - wavelength_min) * plot_width
    spectrum_min = min(-0.04, float(predicted_candidates.min()))
    spectrum_max = max(0.70, float(predicted_candidates.max()))
    sy_spectrum = lambda value: top + plot_height - (value - spectrum_min) / (spectrum_max - spectrum_min) * plot_height
    for wavelength in (400, 500, 600, 700, 800):
        px = sx_spectrum(wavelength)
        svg.line(px, top, px, top + plot_height, "#e8edf3", 1)
        svg.text(px, top + plot_height + 24, str(wavelength), 13, "#64748b", "middle")
    for value in (0.0, 0.2, 0.4, 0.6):
        py = sy_spectrum(value)
        svg.line(left, py, left + plot_width, py, "#e8edf3", 1)
        svg.text(left - 12, py + 5, f"{value:.1f}", 13, "#64748b", "end")
    svg.line(left, top + plot_height, left + plot_width, top + plot_height, "#64748b", 1.5)
    svg.line(left, top, left, top + plot_height, "#64748b", 1.5)
    svg.line(sx_spectrum(700), top, sx_spectrum(700), top + plot_height, "#e6a23c", 1.5, "5,5")

    for index_value, objective, tmm_color, pred_color in (
        (min_order[0], "min", "#087e8b", "#5eead4"),
        (max_order[0], "max", "#b45309", "#fdba74"),
    ):
        index = int(index_value)
        svg.path(
            [(sx_spectrum(float(wave)), sy_spectrum(float(value))) for wave, value in zip(wavelengths, predicted_candidates[index])],
            pred_color,
            2.5,
        )
        svg.path(
            [(sx_spectrum(float(wave)), sy_spectrum(float(value))) for wave, value in zip(wavelengths, tmm_spectra[index])],
            tmm_color,
            3,
        )
        svg.text(
            105, 965 if objective == "min" else 990,
            f"Global {objective}: candidate {index}; linear R700={predicted_r700[index]:.6f}; TMM R700={tmm_r700[index]:.7f}",
            13, tmm_color,
        )
    _legend(
        svg, 1050, 565,
        [("Global min · TMM", "#087e8b"), ("Global min · linear MLP", "#5eead4"),
         ("Global max · TMM", "#b45309"), ("Global max · linear MLP", "#fdba74")],
    )
    svg.text(750, 1025, "All rankings and highlighted designs use the same existing 10,000-candidate linear pool.", 14, "#526579", "middle")
    svg.save(output_path)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing independent audit directory: {OUTPUT_DIR}")

    with METRICS_PATH.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model_metrics = metrics["results_by_training_size"]["4000"]
    if checkpoint.get("training_size") != 4000 or checkpoint.get("output_activation") != "linear":
        raise ValueError("The checkpoint is not the 4000-sample linear-output model")
    if not np.isclose(checkpoint.get("validation_mse"), model_metrics["validation_mse"], rtol=0, atol=1e-12):
        raise ValueError("Checkpoint validation MSE does not match the linear metrics report")

    with np.load(CANDIDATES_PATH, allow_pickle=False) as archive:
        candidates = np.asarray(archive["thicknesses_nm"], dtype=np.float64)
        saved_predictions = np.asarray(archive["predicted_reflectance"], dtype=np.float32)
        wavelengths = np.asarray(archive["wavelengths_nm"], dtype=np.float64)
        design_seed = int(archive["design_seed"])
    if candidates.shape != (10_000, 4) or saved_predictions.shape != (10_000, len(wavelengths)):
        raise ValueError("Linear screening archive has unexpected candidate or prediction dimensions")
    if design_seed != int(metrics["design_seed"]):
        raise ValueError("Linear screening archive design seed does not match its metrics report")
    if not np.array_equal(wavelengths, np.asarray(STUDY_CONFIG.wavelengths_nm, dtype=np.float64)):
        raise ValueError("Candidate and TMM wavelength grids do not match")

    model = MLPRegressor(output_activation="linear")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(1)
    candidate_inputs = ((candidates - STUDY_CONFIG.thickness_min_nm) /
                        (STUDY_CONFIG.thickness_max_nm - STUDY_CONFIG.thickness_min_nm)).astype(np.float32)
    with torch.no_grad():
        predicted_candidates = model(torch.from_numpy(candidate_inputs)).cpu().numpy()
    if not np.allclose(predicted_candidates, saved_predictions, rtol=1e-6, atol=1e-7):
        raise ValueError("Saved linear predictions do not match inference from the selected checkpoint")
    max_prediction_difference = float(np.max(np.abs(predicted_candidates - saved_predictions)))

    target_column = int(np.flatnonzero(wavelengths == STUDY_CONFIG.target_wavelength_nm)[0])
    tmm_spectra = np.empty((len(candidates), len(wavelengths)), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        tmm_spectra[index] = reflectance_spectrum(
            wavelengths,
            STUDY_CONFIG.layer_indices,
            candidate,
            STUDY_CONFIG.incident_index,
            STUDY_CONFIG.substrate_index,
        )

    tmm_r700 = tmm_spectra[:, target_column]
    min_order = np.argsort(tmm_r700, kind="stable")
    max_order = np.argsort(-tmm_r700, kind="stable")
    min_ranks = np.empty(len(candidates), dtype=np.int64)
    min_ranks[min_order] = np.arange(1, len(candidates) + 1)
    max_ranks = np.empty(len(candidates), dtype=np.int64)
    max_ranks[max_order] = np.arange(1, len(candidates) + 1)
    predicted_r700 = predicted_candidates[:, target_column]

    min_rows = _top_rows(min_order, candidates, predicted_r700, tmm_r700, max_ranks)
    max_rows = _top_rows(max_order, candidates, predicted_r700, tmm_r700, min_ranks)
    rank_fields = [
        "tmm_rank_overall", "candidate_index", "d1_nm", "d2_nm", "d3_nm", "d4_nm",
        "mlp_reflectance_at_target", "tmm_reflectance_at_target",
    ]
    all_rows = []
    for index_value in min_order:
        index = int(index_value)
        all_rows.append({
            "tmm_rank_overall": int(min_ranks[index]),
            "candidate_index": index,
            "d1_nm": float(candidates[index, 0]),
            "d2_nm": float(candidates[index, 1]),
            "d3_nm": float(candidates[index, 2]),
            "d4_nm": float(candidates[index, 3]),
            "mlp_reflectance_at_target": float(predicted_r700[index]),
            "tmm_reflectance_at_target": float(tmm_r700[index]),
            "tmm_rank_min": int(min_ranks[index]),
            "tmm_rank_max": int(max_ranks[index]),
        })

    linear_min_top10 = _load_top10(LINEAR_DIR / "design_top10_min.csv")
    linear_max_top10 = _load_top10(LINEAR_DIR / "design_top10_max.csv")
    min_overlap = sorted({int(row["candidate_index"]) for row in linear_min_top10} & {int(i) for i in min_order[:5]})
    max_overlap = sorted({int(row["candidate_index"]) for row in linear_max_top10} & {int(i) for i in max_order[:5]})

    with np.load(DATASET_PATH, allow_pickle=False) as archive:
        dataset_spectra = archive["reflectance"]
        test_indices = archive["test_indices"]
        dataset_wavelengths = archive["wavelengths_nm"]
        thicknesses = archive["thicknesses_nm"]
    if not np.array_equal(dataset_wavelengths, wavelengths):
        raise ValueError("Dataset and candidate wavelength grids do not match")
    test_inputs = ((thicknesses[test_indices] - STUDY_CONFIG.thickness_min_nm) /
                   (STUDY_CONFIG.thickness_max_nm - STUDY_CONFIG.thickness_min_nm)).astype(np.float32)
    with torch.no_grad():
        predicted_test = model(torch.from_numpy(test_inputs)).cpu().numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        OUTPUT_DIR / "linear_candidate_full_tmm.npz",
        candidate_index=np.arange(len(candidates), dtype=np.int64),
        thicknesses_nm=candidates,
        linear_predicted_reflectance=predicted_candidates,
        tmm_reflectance_spectra=tmm_spectra,
        tmm_reflectance_at_target=tmm_r700,
        tmm_rank_min=min_ranks,
        tmm_rank_max=max_ranks,
        wavelengths_nm=wavelengths,
        target_wavelength_nm=np.asarray(STUDY_CONFIG.target_wavelength_nm),
        design_seed=np.asarray(design_seed),
    )
    _write_csv(OUTPUT_DIR / "linear_candidate_tmm_ranked.csv", all_rows, rank_fields + ["tmm_rank_min", "tmm_rank_max"])
    _write_csv(OUTPUT_DIR / "linear_global_top5_min.csv", min_rows, list(min_rows[0]))
    _write_csv(OUTPUT_DIR / "linear_global_top5_max.csv", max_rows, list(max_rows[0]))
    _make_figure3(
        OUTPUT_DIR / "figure_3_linear_global_tmm.svg",
        wavelengths,
        dataset_spectra[test_indices],
        predicted_test,
        predicted_candidates,
        tmm_spectra,
        min_order,
        max_order,
        tmm_r700,
    )

    report = {
        "status": "success",
        "model_output_activation": "linear",
        "model_training_size": 4000,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "metrics": str(METRICS_PATH.relative_to(ROOT)),
        "candidate_source": str(CANDIDATES_PATH.relative_to(ROOT)),
        "candidate_count": int(len(candidates)),
        "design_seed": design_seed,
        "target_wavelength_nm": int(STUDY_CONFIG.target_wavelength_nm),
        "saved_prediction_matches_checkpoint_inference": True,
        "max_abs_prediction_difference": max_prediction_difference,
        "full_tmm_evaluation_count": int(len(candidates)),
        "minimum": {
            "candidate_index": int(min_order[0]),
            "tmm_reflectance_at_target": float(tmm_r700[min_order[0]]),
        },
        "maximum": {
            "candidate_index": int(max_order[0]),
            "tmm_reflectance_at_target": float(tmm_r700[max_order[0]]),
        },
        "comparison_with_linear_mlp_top10": {
            "min_overlap_count_with_global_top5": len(min_overlap),
            "min_overlap_candidate_ids": min_overlap,
            "max_overlap_count_with_global_top5": len(max_overlap),
            "max_overlap_candidate_ids": max_overlap,
        },
        "artifacts": {
            "full_candidate_npz": "linear_candidate_full_tmm.npz",
            "all_candidate_ranked_csv": "linear_candidate_tmm_ranked.csv",
            "global_top5_min_csv": "linear_global_top5_min.csv",
            "global_top5_max_csv": "linear_global_top5_max.csv",
            "updated_figure3": "figure_3_linear_global_tmm.svg",
        },
        "global_tmm_top5_min": min_rows,
        "global_tmm_top5_max": max_rows,
    }
    (OUTPUT_DIR / "linear_global_tmm_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()