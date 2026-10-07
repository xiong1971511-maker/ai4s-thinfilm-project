import unittest

import numpy as np

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.design import generate_candidates, rank_candidates, verify_spectra


class DesignScreeningTests(unittest.TestCase):
    def test_design_candidates_are_seeded_and_within_thickness_bounds(self) -> None:
        first = generate_candidates(sample_count=16)
        second = generate_candidates(sample_count=16)

        self.assertEqual(first.shape, (16, 4))
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(float(first.min()), STUDY_CONFIG.thickness_min_nm)
        self.assertLessEqual(float(first.max()), STUDY_CONFIG.thickness_max_nm)
        self.assertFalse(
            np.array_equal(
                first,
                np.random.default_rng(STUDY_CONFIG.seed).uniform(40, 180, (16, 4)),
            )
        )

    def test_candidate_ranking_supports_both_reflectance_objectives(self) -> None:
        predictions = np.array(
            [[0.1, 0.6], [0.4, 0.2], [0.3, 0.9]],
            dtype=np.float64,
        )
        low_first = rank_candidates(predictions, target_column=0, objective="min")
        high_first = rank_candidates(predictions, target_column=0, objective="max")

        np.testing.assert_array_equal(low_first, np.array([0, 2, 1]))
        np.testing.assert_array_equal(high_first, np.array([1, 2, 0]))
        with self.assertRaisesRegex(ValueError, "objective"):
            rank_candidates(predictions, target_column=0, objective="middle")

    def test_tmm_verification_returns_one_bounded_value_per_wavelength(self) -> None:
        candidates = np.array(
            [[76.0, 120.0, 76.0, 120.0], [100.0, 100.0, 100.0, 100.0]],
            dtype=np.float64,
        )
        spectra = verify_spectra(candidates)

        self.assertEqual(spectra.shape, (2, 41))
        self.assertTrue(np.isfinite(spectra).all())
        self.assertTrue(((spectra >= 0.0) & (spectra <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
