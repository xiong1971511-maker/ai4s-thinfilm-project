"""Train nested-size MLP ablations while keeping validation and test splits fixed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.training import TrainingConfig, train_model


TRAINING_SIZES = (500, 1000, 2000, 4000)
OUTPUT_ACTIVATION = "sigmoid"
MODEL_DIR = Path("models_ablation")
OUTPUT_DIR = Path("outputs_ablation")
DATASET_PATH = Path("data/thinfilm_dataset.npz")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(DATASET_PATH, allow_pickle=False) as archive:
        thicknesses_nm = archive["thicknesses_nm"]
        spectra = archive["reflectance"]
        train_indices = archive["train_indices"]
        validation_indices = archive["validation_indices"]
        test_indices = archive["test_indices"]

    if len(train_indices) < max(TRAINING_SIZES):
        raise ValueError("The dataset does not contain enough training rows for the requested ablation")
    if (len(validation_indices), len(test_indices)) != (500, 500):
        raise ValueError("Validation/test splits must remain fixed at 500 each")

    x_min = float(STUDY_CONFIG.thickness_min_nm)
    x_max = float(STUDY_CONFIG.thickness_max_nm)
    x_scaled = ((thicknesses_nm - x_min) / (x_max - x_min)).astype(np.float32)
    y = spectra.astype(np.float32)

    x_validation = x_scaled[validation_indices]
    y_validation = y[validation_indices]
    x_test = x_scaled[test_indices]
    y_test = y[test_indices]

    report: dict[str, object] = {
        "status": "success",
        "summary": "Nested training-size ablation with fixed validation and test splits",
        "seed": int(STUDY_CONFIG.seed),
        "fixed_split_sizes": {"train": 4000, "validation": 500, "test": 500},
        "training_settings": {
            "layer_sizes": [4, 128, 128, 64, 41],
            "output_activation": OUTPUT_ACTIVATION,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "loss": "MSELoss",
            "batch_size": 128,
            "max_epochs": 250,
            "early_stopping_patience": 50,
            "activation": "ReLU",
            "input_scaling": "(thickness_nm - 40) / 140",
            "dataset_path": str(DATASET_PATH),
        },
        "results_by_training_size": {},
    }

    for training_size in TRAINING_SIZES:
        subset = train_indices[:training_size]
        result = train_model(
            x_scaled[subset],
            y[subset],
            x_validation,
            y_validation,
            x_test,
            y_test,
            config=TrainingConfig(
                learning_rate=1e-3,
                batch_size=128,
                max_epochs=250,
                patience=50,
                min_delta=1e-8,
            ),
            seed=STUDY_CONFIG.seed,
            output_activation=OUTPUT_ACTIVATION,
        )

        checkpoint_path = MODEL_DIR / f"mlp_train_{training_size}.pt"
        torch.save(
            {
                "model_state_dict": result.model.state_dict(),
                "layer_sizes": result.model.layer_sizes,
                "output_activation": result.model.output_activation,
                "training_size": int(training_size),
                "seed": int(STUDY_CONFIG.seed),
                "best_epoch": int(result.best_epoch),
                "validation_mse": float(result.validation_mse),
                "test_mse": float(result.test_mse),
                "test_mae": float(result.test_mae),
                "test_rmse": float(result.test_rmse),
                "test_r2": float(result.test_r2),
            },
            checkpoint_path,
        )

        history_path = OUTPUT_DIR / f"loss_history_train_{training_size}.csv"
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("epoch,training_mse,validation_mse\n")
            for row in result.history:
                handle.write(f"{int(row['epoch'])},{row['training_mse']},{row['validation_mse']}\n")

        metrics = {
            "training_size": int(training_size),
            "best_epoch": int(result.best_epoch),
            "validation_mse": float(result.validation_mse),
            "test_mse": float(result.test_mse),
            "test_mae": float(result.test_mae),
            "test_rmse": float(result.test_rmse),
            "test_r2": float(result.test_r2),
            "checkpoint": str(checkpoint_path),
            "loss_history": str(history_path),
        }
        report["results_by_training_size"][str(training_size)] = metrics

        print(
            f"size={training_size:4d} best_epoch={result.best_epoch:3d} "
            f"val_mse={result.validation_mse:.8f} test_mse={result.test_mse:.8f} "
            f"test_mae={result.test_mae:.8f} test_rmse={result.test_rmse:.8f}"
        )

    report_path = OUTPUT_DIR / "training_size_ablation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    sizes = [int(size) for size in report["results_by_training_size"]]
    test_mse = [float(report["results_by_training_size"][str(size)]["test_mse"]) for size in sizes]
    val_mse = [float(report["results_by_training_size"][str(size)]["validation_mse"]) for size in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(sizes, val_mse, marker="o", color="tab:blue")
    ax1.set_title("Validation MSE vs training size")
    ax1.set_xlabel("Training samples")
    ax1.set_ylabel("Validation MSE")
    ax1.set_xscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    ax2.plot(sizes, test_mse, marker="o", color="tab:orange")
    ax2.set_title("Test MSE vs training size")
    ax2.set_xlabel("Training samples")
    ax2.set_ylabel("Test MSE")
    ax2.set_xscale("log")
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_size_vs_error.png", dpi=200)
    plt.close(fig)

    print(f"Saved report to {report_path}")
    print(f"Saved plot to {OUTPUT_DIR / 'training_size_vs_error.png'}")


if __name__ == "__main__":
    main()
