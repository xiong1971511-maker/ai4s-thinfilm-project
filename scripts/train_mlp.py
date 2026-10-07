"""Train the assignment MLPs for the four prescribed training-set sizes."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.training import TrainingConfig, train_model


TRAINING_SIZES = (500, 1000, 2000, 4000)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("epoch", "training_mse", "validation_mse"),
        )
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/thinfilm_dataset.npz")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--training-sizes",
        type=int,
        nargs="+",
        choices=TRAINING_SIZES,
        default=TRAINING_SIZES,
        help="subset of assignment training sizes to run (default: all four)",
    )
    parser.add_argument(
        "--output-activation",
        choices=("linear", "sigmoid"),
        default="sigmoid",
        help="linear for unconstrained regression or sigmoid to enforce 0-1 reflectance",
    )
    args = parser.parse_args()
    if len(set(args.training_sizes)) != len(args.training_sizes):
        parser.error("--training-sizes must not contain duplicates")

    config = STUDY_CONFIG
    training_config = TrainingConfig(
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as archive:
        thicknesses_nm = archive["thicknesses_nm"]
        spectra = archive["reflectance"]
        train_indices = archive["train_indices"]
        validation_indices = archive["validation_indices"]
        test_indices = archive["test_indices"]
        wavelengths_nm = archive["wavelengths_nm"]
        dataset_metadata = json.loads(str(archive["metadata"].item()))

    if len(train_indices) < TRAINING_SIZES[-1]:
        raise ValueError("dataset training split must contain at least 4000 rows")
    expected_split_sizes = (4000, 500, 500)
    actual_split_sizes = (len(train_indices), len(validation_indices), len(test_indices))
    if actual_split_sizes != expected_split_sizes:
        raise ValueError(
            f"expected fixed 4000/500/500 split, got {actual_split_sizes}"
        )
    thickness_scale = config.thickness_max_nm - config.thickness_min_nm
    x_scaled = (thicknesses_nm - config.thickness_min_nm) / thickness_scale
    x_scaled = x_scaled.astype(np.float32)
    y = spectra.astype(np.float32)
    x_validation = x_scaled[validation_indices]
    y_validation = y[validation_indices]
    x_test = x_scaled[test_indices]
    y_test = y[test_indices]

    results: dict[str, object] = {}
    for training_size in args.training_sizes:
        started = time.perf_counter()
        result = train_model(
            x_scaled[train_indices[:training_size]],
            y[train_indices[:training_size]],
            x_validation,
            y_validation,
            x_test,
            y_test,
            config=training_config,
            seed=config.seed,
            output_activation=args.output_activation,
        )
        elapsed_seconds = time.perf_counter() - started

        checkpoint_path = models_dir / f"mlp_train_{training_size}.pt"
        torch.save(
            {
                "model_state_dict": result.model.state_dict(),
                "layer_sizes": result.model.layer_sizes,
                "output_activation": result.model.output_activation,
                "seed": config.seed,
                "training_size": training_size,
                "best_epoch": result.best_epoch,
                "validation_mse": result.validation_mse,
                "input_scaling": {
                    "minimum_nm": config.thickness_min_nm,
                    "maximum_nm": config.thickness_max_nm,
                    "formula": "(thickness_nm - minimum_nm) / (maximum_nm - minimum_nm)",
                },
                "wavelengths_nm": wavelengths_nm.tolist(),
                "dataset_metadata": dataset_metadata,
                "training_config": {
                    "optimizer": "Adam",
                    "learning_rate": training_config.learning_rate,
                    "loss": "mean squared error",
                    "batch_size": training_config.batch_size,
                    "max_epochs": training_config.max_epochs,
                    "early_stopping_patience": training_config.patience,
                    "activation": "ReLU",
                    "output_activation": args.output_activation,
                    "device": "cpu",
                },
            },
            checkpoint_path,
        )
        history_path = outputs_dir / f"loss_history_train_{training_size}.csv"
        _write_history(history_path, result.history)
        results[str(training_size)] = {
            "training_size": training_size,
            "best_epoch": result.best_epoch,
            "epochs_run": len(result.history),
            "validation_mse": result.validation_mse,
            "test_mse": result.test_mse,
            "test_rmse": result.test_rmse,
            "test_mae": result.test_mae,
            "test_r2": result.test_r2,
            "elapsed_seconds": elapsed_seconds,
            "checkpoint": str(checkpoint_path),
            "loss_history": str(history_path),
        }
        print(
            f"train_size={training_size:4d}  best_epoch={result.best_epoch:3d}  "
            f"test_RMSE={result.test_rmse:.6f}  test_MAE={result.test_mae:.6f}  "
            f"test_R2={result.test_r2:.6f}  elapsed={elapsed_seconds:.1f}s",
            flush=True,
        )

    report = {
        "status": "success",
        "summary": (
            "Trained MLPs at assignment training sizes "
            f"{', '.join(str(size) for size in args.training_sizes)} "
            "using fixed train/validation/test splits"
        ),
        "next_actions": ["Review metrics and loss histories before design screening"],
        "artifacts": [str(outputs_dir / "mlp_metrics.json"), str(models_dir)],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": config.seed,
        "design_seed": config.design_seed,
        "target_wavelength_nm": config.target_wavelength_nm,
        "split_sizes": {
            "train_pool": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "model": {
            "layer_sizes": [4, 128, 128, 64, 41],
            "activation": "ReLU",
            "output_activation": args.output_activation,
            "optimizer": "Adam",
            "learning_rate": training_config.learning_rate,
            "loss": "mean squared error",
            "batch_size": training_config.batch_size,
            "max_epochs": training_config.max_epochs,
            "early_stopping_patience": training_config.patience,
            "input_scaling": "linear map of 40-180 nm to 0-1",
        },
        "results_by_training_size": results,
    }
    report_path = outputs_dir / "mlp_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
