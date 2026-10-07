import math
import unittest

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.tmm import (
    reflectance,
    reflectance_spectrum,
    transmittance,
)


class StudyConfigTests(unittest.TestCase):
    def test_student_parameters_and_wavelength_grid(self) -> None:
        self.assertEqual(STUDY_CONFIG.target_wavelength_nm, 700)
        self.assertEqual(STUDY_CONFIG.seed, 270256)
        self.assertEqual(STUDY_CONFIG.design_seed, 270257)
        self.assertEqual(STUDY_CONFIG.layer_indices, (2.30, 1.45, 2.30, 1.45))
        self.assertEqual(len(STUDY_CONFIG.wavelengths_nm), 41)
        self.assertEqual(STUDY_CONFIG.wavelengths_nm[0], 400)
        self.assertEqual(STUDY_CONFIG.wavelengths_nm[-1], 800)


class TMMPhysicsTests(unittest.TestCase):
    def test_bare_air_glass_interface_matches_fresnel_formula(self) -> None:
        expected = ((1.0 - 1.52) / (1.0 + 1.52)) ** 2
        actual = reflectance(
            wavelength_nm=550.0,
            layer_indices=(),
            thicknesses_nm=(),
            incident_index=1.0,
            substrate_index=1.52,
        )
        self.assertAlmostEqual(actual, expected, places=14)

    def test_zero_thickness_layers_reduce_to_bare_interface(self) -> None:
        bare = ((1.0 - 1.52) / (1.0 + 1.52)) ** 2
        actual = reflectance(
            wavelength_nm=700.0,
            layer_indices=(2.30, 1.45, 2.30, 1.45),
            thicknesses_nm=(0.0, 0.0, 0.0, 0.0),
            incident_index=1.0,
            substrate_index=1.52,
        )
        self.assertAlmostEqual(actual, bare, places=14)

    def test_lossless_stack_conserves_energy(self) -> None:
        for wavelength_nm in STUDY_CONFIG.wavelengths_nm:
            with self.subTest(wavelength_nm=wavelength_nm):
                r_value = reflectance(
                    wavelength_nm=wavelength_nm,
                    layer_indices=STUDY_CONFIG.layer_indices,
                    thicknesses_nm=(80.0, 120.0, 90.0, 110.0),
                    incident_index=STUDY_CONFIG.incident_index,
                    substrate_index=STUDY_CONFIG.substrate_index,
                )
                t_value = transmittance(
                    wavelength_nm=wavelength_nm,
                    layer_indices=STUDY_CONFIG.layer_indices,
                    thicknesses_nm=(80.0, 120.0, 90.0, 110.0),
                    incident_index=STUDY_CONFIG.incident_index,
                    substrate_index=STUDY_CONFIG.substrate_index,
                )
                self.assertAlmostEqual(r_value + t_value, 1.0, places=12)

    def test_two_quarter_wave_pairs_raise_target_reflectance(self) -> None:
        target = STUDY_CONFIG.target_wavelength_nm
        d_high = target / (4.0 * STUDY_CONFIG.high_index)
        d_low = target / (4.0 * STUDY_CONFIG.low_index)
        quarter_wave_r = reflectance(
            wavelength_nm=target,
            layer_indices=STUDY_CONFIG.layer_indices,
            thicknesses_nm=(d_high, d_low, d_high, d_low),
            incident_index=STUDY_CONFIG.incident_index,
            substrate_index=STUDY_CONFIG.substrate_index,
        )
        bare_r = ((1.0 - STUDY_CONFIG.substrate_index) /
                  (1.0 + STUDY_CONFIG.substrate_index)) ** 2
        self.assertGreater(quarter_wave_r, bare_r)
        self.assertGreaterEqual(quarter_wave_r, 0.0)
        self.assertLessEqual(quarter_wave_r, 1.0)

    def test_spectrum_has_one_finite_bounded_value_per_wavelength(self) -> None:
        spectrum = reflectance_spectrum(
            wavelengths_nm=STUDY_CONFIG.wavelengths_nm,
            layer_indices=STUDY_CONFIG.layer_indices,
            thicknesses_nm=(80.0, 120.0, 80.0, 120.0),
            incident_index=STUDY_CONFIG.incident_index,
            substrate_index=STUDY_CONFIG.substrate_index,
        )
        self.assertEqual(len(spectrum), 41)
        self.assertTrue(all(math.isfinite(value) for value in spectrum))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in spectrum))

    def test_mismatched_layer_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            reflectance(
                wavelength_nm=550.0,
                layer_indices=(2.30, 1.45),
                thicknesses_nm=(100.0,),
                incident_index=1.0,
                substrate_index=1.52,
            )


if __name__ == "__main__":
    unittest.main()
