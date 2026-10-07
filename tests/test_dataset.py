import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.dataset import generate_dataset, save_dataset, split_indices


class DatasetTests(unittest.TestCase):
    def test_generation_is_reproducible_and_has_expected_shapes(self) -> None:
        first_x, first_y = generate_dataset(sample_count=8)
        second_x, second_y = generate_dataset(sample_count=8)

        self.assertEqual(first_x.shape, (8, 4))
        self.assertEqual(first_y.shape, (8, 41))
        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y, second_y)
        self.assertTrue(np.all(first_x >= 40.0))
        self.assertTrue(np.all(first_x <= 180.0))
        self.assertTrue(np.all(first_y >= 0.0))
        self.assertTrue(np.all(first_y <= 1.0))

    def test_assignment_split_is_fixed_disjoint_and_covers_all_rows(self) -> None:
        train, validation, test = split_indices(
            sample_count=5000,
            seed=STUDY_CONFIG.seed,
            split_sizes=(4000, 500, 500),
        )

        self.assertEqual((len(train), len(validation), len(test)), (4000, 500, 500))
        all_indices = np.concatenate((train, validation, test))
        self.assertEqual(len(np.unique(all_indices)), 5000)
        np.testing.assert_array_equal(
            train,
            split_indices(5000, STUDY_CONFIG.seed, (4000, 500, 500))[0],
        )

    def test_dataset_archive_round_trips_splits_and_metadata(self) -> None:
        thicknesses, spectra = generate_dataset(sample_count=8)
        splits = split_indices(8, STUDY_CONFIG.seed, (4, 2, 2))
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "small_dataset.npz"
            save_dataset(destination, thicknesses, spectra, splits)

            with np.load(destination, allow_pickle=False) as archive:
                self.assertEqual(archive["thicknesses_nm"].shape, (8, 4))
                self.assertEqual(archive["reflectance"].shape, (8, 41))
                self.assertEqual(archive["train_indices"].shape, (4,))
                self.assertEqual(archive["validation_indices"].shape, (2,))
                self.assertEqual(archive["test_indices"].shape, (2,))
                self.assertEqual(archive["wavelengths_nm"].tolist(), list(STUDY_CONFIG.wavelengths_nm))
                self.assertEqual(str(archive["metadata"].item()), "{}")


if __name__ == "__main__":
    unittest.main()
