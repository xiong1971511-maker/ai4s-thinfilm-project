"""Train nested training-size ablation subsets while preserving the original split."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.training import TrainingConfig, train_model

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "thinfilm_dataset.npz"
RESULT_ROOT = ROOT / "results" / "ablation"
MODEL_DIR = RESULT_ROOT / "models"
LOSS_DIR = RESULT_ROOT / "losses"
SUMMARY_PATH = RESULT_ROOT / "summary.json"
PLOT_PATH = RESULT_ROOT / "training_size_ablation.png"

TRAINING_SIZES = (500, 1000, 2000, 4000)
TRAINING_CONFIG = TrainingConfig(max_epochs=250, patience=50)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "training_mse", "validation_mse"))
        writer.writeheader()
        writer.writerows(history)


def _nested_training_subsets(train_pool: np.ndarray, sizes: tuple[int, ...], seed: int) -> dict[int, np.ndarray]:
    order = np.random.default_rng(seed).permutation(len(train_pool))
    return {size: train_pool[order[:size]] for size in sizes}


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOSS_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(DATASET_PATH, allow_pickle=False) as archive:
        thicknesses_nm = archive["thicknesses_nm"]
        spectra = archive["reflectance"]
        train_pool = np.asarray(archive["train_indices"], dtype=np.int64)
        validation_indices = np.asarray(archive["validation_indices"], dtype=np.int64)
        test_indices = np.asarray(archive["test_indices"], dtype=np.int64)

    if len(train_pool) != 4000:
        raise ValueError(f"Training pool must contain 4000 rows, found {len(train_pool)}")

    nested_subsets = _nested_training_subsets(train_pool, TRAINING_SIZES, STUDY_CONFIG.seed)
    scale = STUDY_CONFIG.thickness_max_nm - STUDY_CONFIG.thickness_min_nm
    x_scaled = (thicknesses_nm - STUDY_CONFIG.thickness_min_nm) / scale
    x_scaled = x_scaled.astype(np.float32)
    y = spectra.astype(np.float32)

    summary: dict[str, dict[str, float | int | str | list[float]]] = {}
    validation_mse_values: list[float] = []
    test_mse_values: list[float] = []

    for size in TRAINING_SIZES:
        subset = nested_subsets[size]
        result = train_model(
            x_scaled[subset],
            y[subset],
            x_scaled[validation_indices],
            y[validation_indices],
            x_scaled[test_indices],
            y[test_indices],
            config=TRAINING_CONFIG,
            seed=STUDY_CONFIG.seed,
            output_activation="sigmoid",
        )

        checkpoint_path = MODEL_DIR / f"mlp_train_{size}.pt"
        torch.save(
            {
                "model_state_dict": result.model.state_dict(),
                "layer_sizes": result.model.layer_sizes,
                "output_activation": result.model.output_activation,
                "seed": STUDY_CONFIG.seed,
                "training_size": size,
                "best_epoch": result.best_epoch,
                "validation_mse": result.validation_mse,
                "input_scaling": {
                    "minimum_nm": STUDY_CONFIG.thickness_min_nm,
                    "maximum_nm": STUDY_CONFIG.thickness_max_nm,
                    "formula": "(thickness_nm - minimum_nm) / (maximum_nm - minimum_nm)",
                },
                "optimizer": "Adam",
                "loss": "MSELoss",
                "batch_size": TRAINING_CONFIG.batch_size,
                "max_epochs": TRAINING_CONFIG.max_epochs,
                "early_stopping_patience": TRAINING_CONFIG.patience,
            },
            checkpoint_path,
        )

        history_path = LOSS_DIR / f"loss_history_train_{size}.csv"
        _write_history(history_path, result.history)

        summary[str(size)] = {
            "training_size": int(size),
            "best_epoch": int(result.best_epoch),
            "validation_mse": float(result.validation_mse),
            "test_mse": float(result.test_mse),
            "test_mae": float(result.test_mae),
            "test_rmse": float(result.test_rmse),
            "checkpoint": str(checkpoint_path),
            "loss_history": str(history_path),
        }
        validation_mse_values.append(float(result.validation_mse))
        test_mse_values.append(float(result.test_mse))

        print(
            f"size={size:4d} best_epoch={result.best_epoch:3d} "
            f"val_mse={result.validation_mse:.8f} test_mse={result.test_mse:.8f} "
            f"test_mae={result.test_mae:.8f} test_rmse={result.test_rmse:.8f}"
        )

    summary_payload = {
        "seed": int(STUDY_CONFIG.seed),
        "source_dataset": str(DATASET_PATH),
        "split_sizes": {"train": 4000, "validation": 500, "test": 500},
        "training_sizes": TRAINING_SIZES,
        "model": {
            "layer_sizes": [4, 128, 128, 64, 41],
            "activation": "ReLU",
            "output_activation": "sigmoid",
            "optimizer": "Adam",
            "learning_rate": TRAINING_CONFIG.learning_rate,
            "loss": "MSELoss",
            "batch_size": TRAINING_CONFIG.batch_size,
            "max_epochs": TRAINING_CONFIG.max_epochs,
            "early_stopping_patience": TRAINING_CONFIG.patience,
            "init_seed": STUDY_CONFIG.seed,
        },
        "results_by_training_size": summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    plt.figure(figsize=(9, 6))
    plt.plot(TRAINING_SIZES, [summary[str(size)]["validation_mse"] for size in TRAINING_SIZES], marker="o", color="tab:blue", label="Best validation MSE")
    plt.plot(TRAINING_SIZES, [summary[str(size)]["test_mse"] for size in TRAINING_SIZES], marker="s", color="tab:orange", label="Test MSE")
    plt.xscale("log", base=2)
    plt.xlabel("Training set size")
    plt.ylabel("MSE")
    plt.title("Training-size ablation: validation and test error")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Plot written to {PLOT_PATH}")


if __name__ == "__main__":
    main()
