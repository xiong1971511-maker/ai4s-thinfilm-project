"""Reproducible TMM dataset generation and fixed assignment split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import STUDY_CONFIG, StudyConfig
from .tmm import reflectance_spectrum

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]


def generate_dataset(
    sample_count: int = 5000,
    config: StudyConfig = STUDY_CONFIG,
) -> tuple[FloatArray, FloatArray]:
    """Generate uniform random thicknesses and their TMM spectra.

    The provided study seed controls thickness generation. Thicknesses are
    sampled continuously and independently within the inclusive assignment
    range. Reflectance is calculated directly with this repository's TMM.
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")

    rng = np.random.default_rng(config.seed)
    thicknesses_nm = rng.uniform(
        config.thickness_min_nm,
        config.thickness_max_nm,
        size=(sample_count, len(config.layer_indices)),
    ).astype(np.float64)
    spectra = np.empty((sample_count, len(config.wavelengths_nm)), dtype=np.float64)
    for row, thicknesses in enumerate(thicknesses_nm):
        spectra[row] = reflectance_spectrum(
            config.wavelengths_nm,
            config.layer_indices,
            thicknesses,
            config.incident_index,
            config.substrate_index,
        )
    return thicknesses_nm, spectra


def split_indices(
    sample_count: int,
    seed: int,
    split_sizes: tuple[int, int, int],
) -> tuple[IndexArray, IndexArray, IndexArray]:
    """Create one seeded permutation split into train/validation/test rows."""

    if sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if len(split_sizes) != 3 or any(size < 0 for size in split_sizes):
        raise ValueError("split_sizes must contain three non-negative sizes")
    if sum(split_sizes) != sample_count:
        raise ValueError("split_sizes must sum to sample_count")

    shuffled = np.random.default_rng(seed).permutation(sample_count).astype(np.int64)
    train_end = split_sizes[0]
    validation_end = train_end + split_sizes[1]
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )


def save_dataset(
    destination: str | Path,
    thicknesses_nm: FloatArray,
    spectra: FloatArray,
    splits: tuple[IndexArray, IndexArray, IndexArray],
    metadata: dict[str, Any] | None = None,
    config: StudyConfig = STUDY_CONFIG,
) -> Path:
    """Save arrays, fixed splits, wavelengths and JSON metadata in NPZ format."""

    if thicknesses_nm.ndim != 2 or thicknesses_nm.shape[1] != 4:
        raise ValueError("thicknesses_nm must have shape (sample_count, 4)")
    expected_spectrum_shape = (thicknesses_nm.shape[0], len(config.wavelengths_nm))
    if spectra.shape != expected_spectrum_shape:
        raise ValueError(f"spectra must have shape {expected_spectrum_shape}")
    if len(splits) != 3 or sum(len(indices) for indices in splits) != len(thicknesses_nm):
        raise ValueError("splits must cover every dataset row exactly once")
    joined = np.concatenate(splits)
    if len(np.unique(joined)) != len(thicknesses_nm) or np.any(
        (joined < 0) | (joined >= len(thicknesses_nm))
    ):
        raise ValueError("splits must contain unique valid row indices")

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    np.savez_compressed(
        path,
        thicknesses_nm=thicknesses_nm,
        reflectance=spectra,
        wavelengths_nm=np.asarray(config.wavelengths_nm, dtype=np.int64),
        train_indices=splits[0],
        validation_indices=splits[1],
        test_indices=splits[2],
        metadata=np.asarray(metadata_json),
    )
    return path

