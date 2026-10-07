"""Compare linear and sigmoid runs using validation MSE, without test leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_report(path: str) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "success":
        raise ValueError(f"training report is not successful: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", default="outputs_long_linear/mlp_metrics.json")
    parser.add_argument("--sigmoid", default="outputs_long_sigmoid/mlp_metrics.json")
    parser.add_argument("--output", default="outputs_long_comparison/model_variant_comparison.json")
    args = parser.parse_args()

    linear = _load_report(args.linear)
    sigmoid = _load_report(args.sigmoid)
    if linear["split_sizes"] != sigmoid["split_sizes"]:
        raise ValueError("the two runs used different train/validation/test splits")
    linear_model = dict(linear["model"])
    sigmoid_model = dict(sigmoid["model"])
    linear_model.pop("output_activation", None)
    sigmoid_model.pop("output_activation", None)
    if linear_model != sigmoid_model:
        raise ValueError("the two runs differ in settings beyond output activation")

    linear_results = linear["results_by_training_size"]
    sigmoid_results = sigmoid["results_by_training_size"]
    sizes = sorted(set(linear_results) & set(sigmoid_results), key=int)
    comparisons = {}
    for size in sizes:
        linear_metrics = linear_results[size]
        sigmoid_metrics = sigmoid_results[size]
        linear_validation = float(linear_metrics["validation_mse"])
        sigmoid_validation = float(sigmoid_metrics["validation_mse"])
        comparisons[size] = {
            "linear_validation_mse": linear_validation,
            "sigmoid_validation_mse": sigmoid_validation,
            "validation_mse_relative_reduction_linear_vs_sigmoid": (
                (sigmoid_validation - linear_validation) / sigmoid_validation
            ),
            "linear_best_epoch": linear_metrics["best_epoch"],
            "sigmoid_best_epoch": sigmoid_metrics["best_epoch"],
            "linear_test_rmse_diagnostic_only": linear_metrics["test_rmse"],
            "sigmoid_test_rmse_diagnostic_only": sigmoid_metrics["test_rmse"],
        }

    selection_size = "4000"
    selected = min(
        ("linear", "sigmoid"),
        key=lambda variant: comparisons[selection_size][f"{variant}_validation_mse"],
    )
    report = {
        "status": "success",
        "summary": "Compared output activations under matched training settings",
        "selection_rule": "lowest validation MSE at 4000 training samples; test metrics are diagnostic only",
        "selected_output_activation": selected,
        "training_settings": {
            key: value
            for key, value in linear["model"].items()
            if key != "output_activation"
        },
        "comparisons_by_training_size": comparisons,
        "source_reports": {"linear": args.linear, "sigmoid": args.sigmoid},
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

