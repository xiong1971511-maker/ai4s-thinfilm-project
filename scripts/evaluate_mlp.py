"""Evaluate a surrogate MLP on a held-out split without long training."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.model import MLPRegressor


def load_split(dataset_path: str, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(dataset_path, allow_pickle=False) as archive:
        thicknesses_nm = archive["thicknesses_nm"]
        spectra = archive["reflectance"]
        indices = archive[f"{split_name}_indices"]

    x = thicknesses_nm[indices]
    y = spectra[indices].astype(np.float32)
    x_min = float(STUDY_CONFIG.thickness_min_nm)
    x_max = float(STUDY_CONFIG.thickness_max_nm)
    x_scaled = (x - x_min) / (x_max - x_min)
    return x_scaled.astype(np.float32), y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/thinfilm_dataset.npz")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--checkpoint", default="models_convergence_linear/mlp_train_4000.pt")
    args = parser.parse_args()

    x, y = load_split(args.dataset, args.split)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MLPRegressor(output_activation=checkpoint.get("output_activation", "linear"))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x)).numpy()

    residual = predictions - y
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    print(f"split={args.split}")
    print(f"x_shape={tuple(x.shape)}")
    print(f"y_shape={tuple(y.shape)}")
    print(f"prediction_shape={tuple(predictions.shape)}")
    print(f"mse={mse:.8f}")
    print(f"mae={mae:.8f}")
    print(f"rmse={rmse:.8f}")


if __name__ == "__main__":
    main()
