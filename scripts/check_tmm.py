"""Run small deterministic physics checks for the current TMM phase."""

import json
import platform

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.tmm import (
    reflectance,
    reflectance_spectrum,
    transmittance,
)


def main() -> None:
    config = STUDY_CONFIG
    target = float(config.target_wavelength_nm)
    d_high = target / (4.0 * config.high_index)
    d_low = target / (4.0 * config.low_index)
    thicknesses_nm = (d_high, d_low, d_high, d_low)

    spectrum = reflectance_spectrum(
        config.wavelengths_nm,
        config.layer_indices,
        thicknesses_nm,
        config.incident_index,
        config.substrate_index,
    )
    bare_expected = (
        (config.incident_index - config.substrate_index)
        / (config.incident_index + config.substrate_index)
    ) ** 2
    bare_actual = reflectance(
        550.0,
        (),
        (),
        config.incident_index,
        config.substrate_index,
    )
    energy_errors = []
    for wavelength_nm in config.wavelengths_nm:
        r_value = reflectance(
            wavelength_nm,
            config.layer_indices,
            thicknesses_nm,
            config.incident_index,
            config.substrate_index,
        )
        t_value = transmittance(
            wavelength_nm,
            config.layer_indices,
            thicknesses_nm,
            config.incident_index,
            config.substrate_index,
        )
        energy_errors.append(abs(r_value + t_value - 1.0))

    checks = {
        "wavelength_count_is_41": len(spectrum) == 41,
        "wavelength_endpoints_are_400_and_800_nm": (
            config.wavelengths_nm[0] == 400
            and config.wavelengths_nm[-1] == 800
        ),
        "reflectance_is_finite_and_bounded": all(
            0.0 <= value <= 1.0 for value in spectrum
        ),
        "bare_interface_matches_fresnel": abs(bare_actual - bare_expected) < 1e-14,
        "lossless_energy_error_below_1e-12": max(energy_errors) < 1e-12,
    }
    passed = all(checks.values())
    report = {
        "status": "success" if passed else "error",
        "summary": (
            "TMM physical checks passed"
            if passed
            else "One or more TMM physical checks failed"
        ),
        "next_actions": (
            ["Run scripts/verify_submission.py to verify the published article artifacts"]
            if passed
            else ["Inspect the failed check before continuing"]
        ),
        "artifacts": [
            "src/ai4s_thinfilm/tmm.py",
            "tests/test_tmm.py",
        ],
        "python_version": platform.python_version(),
        "parameters": {
            "target_wavelength_nm": config.target_wavelength_nm,
            "seed": config.seed,
            "design_seed": config.design_seed,
            "layer_indices": config.layer_indices,
            "quarter_wave_thicknesses_nm": thicknesses_nm,
        },
        "spectrum": {
            "point_count": len(spectrum),
            "first_wavelength_nm": config.wavelengths_nm[0],
            "last_wavelength_nm": config.wavelengths_nm[-1],
            "minimum_reflectance": min(spectrum),
            "maximum_reflectance": max(spectrum),
            "target_reflectance": spectrum[
                config.wavelengths_nm.index(config.target_wavelength_nm)
            ],
        },
        "physics_checks": {
            "bare_air_glass_reflectance": bare_actual,
            "max_energy_conservation_error": max(energy_errors),
            "checks": checks,
        },
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

