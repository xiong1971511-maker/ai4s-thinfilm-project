"""Deterministic CPU training utilities for the assignment MLP."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .model import MLPRegressor


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 1200
    patience: int = 100
    min_delta: float = 1e-8


@dataclass
class TrainingResult:
    model: MLPRegressor
    best_epoch: int
    validation_mse: float
    test_mse: float
    test_mae: float
    test_rmse: float
    test_r2: float
    predictions: np.ndarray
    history: list[dict[str, float]]


def _as_tensor(values: np.ndarray, name: str) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return torch.from_numpy(np.ascontiguousarray(array))


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: TrainingConfig = TrainingConfig(),
    seed: int = 270256,
    output_activation: str = "sigmoid",
) -> TrainingResult:
    """Train with Adam/MSE and select the checkpoint by validation MSE.

    Input thicknesses must be normalized to [0, 1] by the caller. The test
    split is used only after checkpoint selection and early stopping.
    """

    arrays = {
        "x_train": _as_tensor(x_train, "x_train"),
        "y_train": _as_tensor(y_train, "y_train"),
        "x_validation": _as_tensor(x_validation, "x_validation"),
        "y_validation": _as_tensor(y_validation, "y_validation"),
        "x_test": _as_tensor(x_test, "x_test"),
        "y_test": _as_tensor(y_test, "y_test"),
    }
    for split in ("train", "validation", "test"):
        inputs = arrays[f"x_{split}"]
        targets = arrays[f"y_{split}"]
        if inputs.shape[0] == 0 or targets.shape[0] == 0:
            raise ValueError(f"{split} split must not be empty")
        if inputs.shape != (inputs.shape[0], 4):
            raise ValueError(f"x_{split} must have four columns")
        if targets.shape != (inputs.shape[0], 41):
            raise ValueError(f"y_{split} must have 41 columns and match x_{split}")
    if config.max_epochs <= 0 or config.batch_size <= 0 or config.patience <= 0:
        raise ValueError("max_epochs, batch_size and patience must be positive")

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if output_activation not in ("linear", "sigmoid"):
        raise ValueError("output_activation must be 'linear' or 'sigmoid'")
    model = MLPRegressor(output_activation=output_activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    train_x = arrays["x_train"]
    train_y = arrays["y_train"]
    validation_x = arrays["x_validation"]
    validation_y = arrays["y_validation"]
    test_x = arrays["x_test"]
    test_y = arrays["y_test"]

    shuffle_generator = torch.Generator(device="cpu").manual_seed(seed)
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(train_x.shape[0], generator=shuffle_generator)
        accumulated_loss = 0.0
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            accumulated_loss += float(loss.detach()) * len(batch_indices)
        training_mse = accumulated_loss / train_x.shape[0]

        model.eval()
        with torch.no_grad():
            validation_mse = float(loss_function(model(validation_x), validation_y))
        history.append(
            {
                "epoch": float(epoch),
                "training_mse": training_mse,
                "validation_mse": validation_mse,
            }
        )

        if validation_mse < best_validation - config.min_delta:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(test_x).cpu().numpy()
    targets = test_y.cpu().numpy()
    residual = predictions - targets
    test_mse = float(np.mean(np.square(residual)))
    test_mae = float(np.mean(np.abs(residual)))
    test_rmse = float(np.sqrt(test_mse))
    total_sum_squares = float(np.sum(np.square(targets - np.mean(targets))))
    test_r2 = (
        1.0 - float(np.sum(np.square(residual))) / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        validation_mse=best_validation,
        test_mse=test_mse,
        test_mae=test_mae,
        test_rmse=test_rmse,
        test_r2=test_r2,
        predictions=predictions,
        history=history,
    )
