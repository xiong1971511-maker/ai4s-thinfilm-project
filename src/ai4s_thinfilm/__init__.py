"""Physics and machine-learning tools for the AI4S thin-film study."""

from .config import STUDY_CONFIG, StudyConfig
from .tmm import reflectance, reflectance_spectrum, transmittance

__all__ = [
    "STUDY_CONFIG",
    "StudyConfig",
    "reflectance",
    "reflectance_spectrum",
    "transmittance",
]

