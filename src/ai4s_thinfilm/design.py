"""Candidate generation, target-wavelength ranking and TMM verification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import STUDY_CONFIG, StudyConfig
from .tmm import reflectance_spectrum

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]


def generate_candidates(
    sample_count: int = 10_000,
    config: StudyConfig = STUDY_CONFIG,
) -> FloatArray:
    """Generate new continuous thickness candidates using design_seed."""

    if sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    rng = np.random.default_rng(config.design_seed)
    return rng.uniform(
        config.thickness_min_nm,
        config.thickness_max_nm,
        size=(sample_count, len(config.layer_indices)),
    ).astype(np.float64)


def rank_candidates(
    predicted_spectra: NDArray[np.floating],
    target_column: int,
    objective: str,
) -> IndexArray:
    """Return stable candidate indices ordered by predicted target reflectance."""

    spectra = np.asarray(predicted_spectra)
    if spectra.ndim != 2:
        raise ValueError("predicted_spectra must be a two-dimensional array")
    if not 0 <= target_column < spectra.shape[1]:
        raise ValueError("target_column is outside the predicted wavelength grid")
    if not np.isfinite(spectra).all():
        raise ValueError("predicted_spectra must contain only finite values")
    if objective == "min":
        return np.argsort(spectra[:, target_column], kind="stable").astype(np.int64)
    if objective == "max":
        return np.argsort(-spectra[:, target_column], kind="stable").astype(np.int64)
    raise ValueError("objective must be 'min' or 'max'")


def verify_spectra(
    thicknesses_nm: NDArray[np.floating],
    config: StudyConfig = STUDY_CONFIG,
) -> FloatArray:
    """Recalculate candidate spectra with the repository's physical TMM."""

    thicknesses = np.asarray(thicknesses_nm, dtype=np.float64)
    if thicknesses.ndim != 2 or thicknesses.shape[1] != len(config.layer_indices):
        raise ValueError("thicknesses_nm must have shape (candidate_count, 4)")
    if not np.isfinite(thicknesses).all():
        raise ValueError("thicknesses_nm must contain only finite values")
    if np.any(thicknesses < config.thickness_min_nm) or np.any(
        thicknesses > config.thickness_max_nm
    ):
        raise ValueError("thicknesses_nm must lie within the configured bounds")
    spectra = np.empty((len(thicknesses), len(config.wavelengths_nm)), dtype=np.float64)
    for row, candidate in enumerate(thicknesses):
        spectra[row] = reflectance_spectrum(
            config.wavelengths_nm,
            config.layer_indices,
            candidate,
            config.incident_index,
            config.substrate_index,
        )
    return spectra

