import unittest

import numpy as np
import torch

from ai4s_thinfilm.model import MLPRegressor
from ai4s_thinfilm.training import TrainingConfig, train_model


class MLPTests(unittest.TestCase):
    def test_model_has_assignment_architecture_and_returns_41_outputs(self) -> None:
        model = MLPRegressor()
        prediction = model(torch.zeros((3, 4), dtype=torch.float32))

        self.assertEqual(model.layer_sizes, (4, 128, 128, 64, 41))
        self.assertEqual(tuple(prediction.shape), (3, 41))

    def test_model_predictions_remain_inside_physical_reflectance_bounds(self) -> None:
        model = MLPRegressor(output_activation="sigmoid")
        final_layer = model.network[-1]
        with torch.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.fill_(8.0)

        prediction = model(torch.zeros((2, 4), dtype=torch.float32))

        self.assertTrue(torch.all(prediction >= 0.0))
        self.assertTrue(torch.all(prediction <= 1.0))

    def test_model_can_select_linear_output_for_fair_architecture_comparison(self) -> None:
        model = MLPRegressor(output_activation="linear")
        final_layer = model.network[-1]
        with torch.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.fill_(8.0)

        prediction = model(torch.zeros((1, 4), dtype=torch.float32))

        self.assertTrue(torch.all(prediction == 8.0))

    def test_training_returns_finite_validation_and_test_metrics(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.uniform(0.0, 1.0, size=(48, 4)).astype(np.float32)
        y = np.repeat(x[:, :1], 41, axis=1).astype(np.float32)
        result = train_model(
            x[:32], y[:32], x[32:40], y[32:40], x[40:], y[40:],
            config=TrainingConfig(max_epochs=3, patience=3, batch_size=8),
            seed=7,
        )

        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertLessEqual(result.best_epoch, 3)
        self.assertTrue(np.isfinite(result.validation_mse))
        self.assertTrue(np.isfinite(result.test_mse))
        self.assertEqual(tuple(result.predictions.shape), (8, 41))


if __name__ == "__main__":
    unittest.main()
