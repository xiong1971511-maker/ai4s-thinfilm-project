"""Single source of truth for the student-specific study parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StudyConfig:
    """Immutable parameters specified by the AI4S course assignment."""

    target_wavelength_nm: int = 700
    seed: int = 270256
    design_seed: int = 270257
    incident_index: float = 1.0
    high_index: float = 2.30
    low_index: float = 1.45
    substrate_index: float = 1.52
    thickness_min_nm: int = 40
    thickness_max_nm: int = 180
    wavelength_start_nm: int = 400
    wavelength_stop_nm: int = 800
    wavelength_step_nm: int = 10

    @property
    def layer_indices(self) -> tuple[float, float, float, float]:
        """Refractive-index sequence for Air/H/L/H/L/Glass."""

        return (
            self.high_index,
            self.low_index,
            self.high_index,
            self.low_index,
        )

    @property
    def wavelengths_nm(self) -> tuple[int, ...]:
        """Inclusive wavelength sampling grid required by the assignment."""

        return tuple(
            range(
                self.wavelength_start_nm,
                self.wavelength_stop_nm + 1,
                self.wavelength_step_nm,
            )
        )


STUDY_CONFIG = StudyConfig()

