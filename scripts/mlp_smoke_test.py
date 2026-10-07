"""Minimal one-epoch smoke test for the assignment MLP surrogate."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.model import MLPRegressor


def main() -> None:
    with np.load("data/thinfilm_dataset.npz", allow_pickle=False) as archive:
        thicknesses_nm = archive["thicknesses_nm"]
        spectra = archive["reflectance"]
        train_indices = archive["train_indices"]

    sample_count = 64
    train_idx = train_indices[:sample_count]
    x = thicknesses_nm[train_idx].astype(np.float32)
    y = spectra[train_idx].astype(np.float32)

    x_min = float(STUDY_CONFIG.thickness_min_nm)
    x_max = float(STUDY_CONFIG.thickness_max_nm)
    x_scaled = (x - x_min) / (x_max - x_min)

    model = MLPRegressor(output_activation="sigmoid")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    inputs = torch.from_numpy(x_scaled)
    targets = torch.from_numpy(y)

    print(f"input_shape={tuple(inputs.shape)}")
    print(f"target_shape={tuple(targets.shape)}")

    model.train()
    predictions = model(inputs)
    print(f"prediction_shape={tuple(predictions.shape)}")
    loss = criterion(predictions, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"epoch_1_mse={float(loss.item()):.8f}")
    print("smoke_test=PASS")


if __name__ == "__main__":
    main()
