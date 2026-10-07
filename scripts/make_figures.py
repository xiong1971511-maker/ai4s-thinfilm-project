"""Create assignment figures as self-contained SVGs from repository outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import torch

from ai4s_thinfilm.config import STUDY_CONFIG
from ai4s_thinfilm.model import MLPRegressor


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<defs><marker id="arrow" markerWidth="10" markerHeight="8" '
            'refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" '
            'fill="#536579"/></marker>'
            '<linearGradient id="header" x1="0" y1="0" x2="1" y2="0">'
            '<stop offset="0%" stop-color="#102a43"/><stop offset="100%" stop-color="#1d4e89"/>'
            '</linearGradient>'
            '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">'
            '<feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#15324b" flood-opacity="0.10"/>'
            '</filter></defs>',
            '<rect width="100%" height="100%" fill="#f3f7fb"/>',
            '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif}</style>',
        ]

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int = 18,
        color: str = "#172b4d",
        anchor: str = "start",
        weight: str = "normal",
    ) -> None:
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(value))}</text>'
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: str,
        stroke: str = "#cbd5e1",
        radius: int = 12,
        shadow: bool = False,
    ) -> None:
        filter_attr = ' filter="url(#shadow)"' if shadow else ""
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
            f'height="{height:.1f}" rx="{radius}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5"{filter_attr}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#64748b",
        width: float = 2,
        dash: str = "",
        arrow: bool = False,
    ) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        end = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
            f'y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"'
            f'{extra}{end}/>'
        )

    def circle(self, x: float, y: float, radius: float, fill: str) -> None:
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}"/>'
        )

    def path(self, points: list[tuple[float, float]], color: str, width: float = 3) -> None:
        if len(points) < 2:
            return
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def save(self, path: Path) -> None:
        path.write_text("\n".join([*self.parts, "</svg>"]), encoding="utf-8")


def _panel(svg: Svg, x: int, y: int, width: int, height: int, title: str) -> None:
    svg.rect(x, y, width, height, "#ffffff", "#d8e1eb", 16, shadow=True)
    svg.rect(x + 2, y + 2, width - 4, 5, "#2b7a9b", "#2b7a9b", 3)
    svg.text(x + 24, y + 37, title, 20, "#17365d", weight="bold")


def _header(svg: Svg, number: str, title: str, subtitle: str) -> None:
    svg.parts.append('<rect x="0" y="0" width="100%" height="96" fill="url(#header)"/>')
    svg.text(48, 31, f"AI4S  /  THIN-FILM STUDY     ·     FIGURE {number}", 13, "#b9d7ea", weight="bold")
    svg.text(48, 63, title, 27, "#ffffff", weight="bold")
    svg.text(1450, 60, subtitle, 14, "#dbeafe", "end")


def _metric_card(svg: Svg, x: int, y: int, width: int, value: str, label: str, accent: str) -> None:
    svg.rect(x, y, width, 66, "#f8fbff", "#dce6f0", 11)
    svg.rect(x, y, 5, 66, accent, accent, 3)
    svg.text(x + 18, y + 29, value, 21, "#17365d", weight="bold")
    svg.text(x + 18, y + 51, label, 12, "#64748b")


def _legend(svg: Svg, x: int, y: int, entries: list[tuple[str, str]]) -> None:
    for index, (label, color) in enumerate(entries):
        yy = y + index * 25
        svg.line(x, yy, x + 24, yy, color, 3)
        svg.text(x + 33, yy + 6, label, 14, "#334155")


def _plot_axes(
    svg: Svg,
    x: int,
    y: int,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
) -> tuple[object, object]:
    left, top = x + 70, y + 65
    plot_w, plot_h = width - 100, height - 125
    sx = lambda value: left + (value - x_min) / (x_max - x_min) * plot_w
    sy = lambda value: top + plot_h - (value - y_min) / (y_max - y_min) * plot_h
    for value, label in x_ticks:
        px = sx(value)
        svg.line(px, top, px, top + plot_h, "#e8edf3", 1)
        svg.text(px, top + plot_h + 25, label, 13, "#64748b", "middle")
    for value, label in y_ticks:
        py = sy(value)
        svg.line(left, py, left + plot_w, py, "#e8edf3", 1)
        svg.text(left - 10, py + 5, label, 13, "#64748b", "end")
    svg.line(left, top + plot_h, left + plot_w, top + plot_h, "#64748b", 1.5)
    svg.line(left, top, left, top + plot_h, "#64748b", 1.5)
    svg.text(left + plot_w / 2, top + plot_h + 53, x_label, 15, "#334155", "middle")
    svg.text(x + 20, top + plot_h / 2, y_label, 15, "#334155", "middle")
    return sx, sy


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _figure1(output: Path, sample: dict[str, np.ndarray], max_epochs: int) -> None:
    svg = Svg(1500, 900)
    _header(svg, "01", "From thin-film physics to inverse design", "Forward model  →  surrogate  →  verified candidates")

    _panel(svg, 45, 118, 1410, 310, "01A   Physical stack and TMM-generated spectral data")
    svg.text(82, 190, "NORMAL-INCIDENCE LIGHT", 12, "#64748b", weight="bold")
    svg.line(92, 226, 137, 226, "#e6a23c", 4, arrow=True)
    layer_items = [
        ("Air", 72, "#e8f1fb"),
        ("H", 112, "#f9d7d4"),
        ("L", 112, "#d6e8fb"),
        ("H", 112, "#f9d7d4"),
        ("L", 112, "#d6e8fb"),
        ("Glass", 96, "#dce3ea"),
    ]
    x0, stack_y, gap = 145, 198, 6
    thickness_index = 0
    for label, width, fill in layer_items:
        svg.rect(x0, stack_y, width, 58, fill, "#9aabba", 6)
        svg.text(x0 + width / 2, stack_y + 36, label, 17, "#17365d", "middle", "bold")
        if label in ("H", "L"):
            svg.text(x0 + width / 2, 282, f"d{thickness_index + 1} = {sample['thicknesses_nm'][thickness_index]:.1f} nm", 12, "#526579", "middle")
            thickness_index += 1
        elif label == "Glass":
            svg.text(x0 + width / 2, 282, "nₛ = 1.52", 12, "#526579", "middle")
        x0 += width + gap
    svg.text(470, 316, "H: n = 2.30", 12, "#b4534a", "middle")
    svg.text(585, 316, "L: n = 1.45", 12, "#2563a6", "middle")

    svg.line(787, 228, 835, 228, "#536579", 2.5, arrow=True)
    svg.rect(850, 186, 182, 105, "#e8f6f6", "#8fc8c6", 13, shadow=True)
    svg.text(941, 222, "TMM", 22, "#0b6e75", "middle", "bold")
    svg.text(941, 248, "n, d, λ → R(λ)", 14, "#526579", "middle")
    svg.text(941, 270, "400–800 nm · 10 nm step", 11, "#64748b", "middle")

    wavelengths = sample["wavelengths_nm"]
    reflectance = sample["reflectance"]
    chart_left, chart_top, chart_w, chart_h = 1090, 188, 300, 132
    svg.text(chart_left, 174, "EXAMPLE TMM SPECTRUM", 12, "#64748b", weight="bold")
    sx = lambda value: chart_left + (value - 400) / 400 * chart_w
    sy = lambda value: chart_top + chart_h - value * chart_h
    for wave in (400, 500, 600, 700, 800):
        px = sx(wave)
        svg.line(px, chart_top, px, chart_top + chart_h, "#e8edf3", 1)
        svg.text(px, chart_top + chart_h + 18, str(wave), 10, "#64748b", "middle")
    for value in (0, 0.5, 1):
        py = sy(value)
        svg.line(chart_left, py, chart_left + chart_w, py, "#e8edf3", 1)
        svg.text(chart_left - 9, py + 4, f"{value:g}", 10, "#64748b", "end")
    svg.path([(sx(float(w)), sy(float(r))) for w, r in zip(wavelengths, reflectance)], "#087e8b", 3)
    svg.line(sx(700), chart_top, sx(700), chart_top + chart_h, "#e6a23c", 1.5, "5,4")
    svg.text(sx(700) + 5, chart_top + 14, "700", 10, "#b45309")

    cards = [
        (82, "5,000", "TMM-labeled stacks", "#087e8b"),
        (411, "41", "wavelength points", "#2563eb"),
        (740, "4,000 / 500 / 500", "train / validation / test", "#8b5cf6"),
        (1069, "700 nm", "inverse-design target", "#d97706"),
    ]
    for x, value, label, accent in cards:
        _metric_card(svg, x, 341, 300, value, label, accent)

    _panel(svg, 45, 452, 1410, 388, "01B   Reproducible modeling and design workflow")
    workflow = [
        (92, "01", "Sample", ["4 layer thicknesses", "40–180 nm · seed 270256"], "#fff5e8", "#d97706"),
        (355, "02", "Simulate", ["TMM forward calculation", "41-point reflectance labels"], "#eaf8f7", "#0b858b"),
        (618, "03", "Learn", ["MLP surrogate model", "4 → 128 → 128 → 64 → 41"], "#edf3ff", "#3568b8"),
        (881, "04", "Search", ["10,000 new candidates", "design_seed 270257"], "#f4efff", "#8056b3"),
        (1144, "05", "Verify", ["TMM recomputes shortlist", "physical result, not proxy"], "#ecf8ed", "#34834a"),
    ]
    for x, number, title, details, fill, accent in workflow:
        svg.rect(x, 535, 222, 144, fill, "#d4e0eb", 15, shadow=True)
        svg.circle(x + 28, 564, 15, accent)
        svg.text(x + 28, 569, number, 11, "#ffffff", "middle", "bold")
        svg.text(x + 52, 570, title, 18, "#17365d", weight="bold")
        svg.line(x + 18, 592, x + 204, 592, accent, 2)
        svg.text(x + 18, 620, details[0], 13, "#334155")
        svg.text(x + 18, 645, details[1], 12, "#64748b")
    for x in (319, 582, 845, 1108):
        svg.line(x, 606, x + 29, 606, "#536579", 2.5, arrow=True)
    svg.rect(95, 716, 1310, 76, "#eef5fb", "#d0deeb", 13)
    svg.text(120, 747, "MODEL SELECTION", 12, "#64748b", weight="bold")
    svg.text(120, 775, "Linear output chosen by lowest validation MSE at N = 4000", 17, "#17365d", weight="bold")
    svg.text(930, 755, f"Adam · CPU · max {max_epochs} epochs · patience 100", 13, "#526579")
    svg.save(output)


def _figure2(
    output: Path,
    linear: dict,
    sigmoid: dict,
    histories: dict[str, list[dict[str, str]]],
    validation_by_size: dict[str, dict[int, float]],
) -> None:
    svg = Svg(1500, 970)
    _header(svg, "02", "The surrogate model and its learning curve", "Architecture  ·  validation loss  ·  data-volume effect")

    _panel(svg, 45, 105, 1410, 250, "02A   Fully connected MLP  ·  schematic (representative neurons shown)")
    centers = [185, 465, 745, 1025, 1305]
    neuron_y = [198, 222, 246, 270, 294]
    layer_names = ["Input", "Dense + ReLU", "Dense + ReLU", "Dense + ReLU", "Output"]
    layer_sizes = ["4 normalized dᵢ", "128 units", "128 units", "64 units", "41 wavelengths"]
    layer_colors = ["#3b82f6", "#6489df", "#6489df", "#6489df", "#0f8b8d"]
    svg.rect(567, 163, 356, 162, "#f7faff", "#e0e8f0", 14)
    for layer in range(len(centers) - 1):
        for y0 in neuron_y:
            for y1 in neuron_y:
                svg.line(centers[layer] + 10, y0, centers[layer + 1] - 10, y1, "#d8e2ed", 1)
    for cx, ys, name, size, color in zip(centers, [neuron_y] * 5, layer_names, layer_sizes, layer_colors):
        svg.text(cx, 174, name, 14, "#334155", "middle", "bold")
        for cy in ys:
            svg.circle(cx, cy, 8, color)
        svg.text(cx, 322, size, 13, "#526579", "middle")
    svg.text(750, 345, "Adam · lr 0.001 · batch 128 · MSE · fixed 4000 / 500 / 500 split · max " + str(linear["model"]["max_epochs"]) + " epochs", 13, "#526579", "middle")

    # Validation MSE by training set size.
    _panel(svg, 45, 380, 680, 535, "(b) Validation MSE vs. training-set size")
    x_values = [500, 1000, 2000, 4000]
    left, top, plot_w, plot_h = 135, 490, 540, 280
    sx = lambda value: left + (value - 500) / 3500 * plot_w
    sy = lambda value: top + plot_h - (np.log10(value) - np.log10(0.0001)) / (np.log10(0.003) - np.log10(0.0001)) * plot_h
    for size in x_values:
        px = sx(size)
        svg.line(px, top, px, top + plot_h, "#e8edf3", 1)
        svg.text(px, top + plot_h + 25, str(size), 13, "#64748b", "middle")
    for value in (0.0001, 0.0003, 0.001, 0.003):
        py = sy(value)
        svg.line(left, py, left + plot_w, py, "#e8edf3", 1)
        svg.text(left - 10, py + 5, f"{value:.0e}", 13, "#64748b", "end")
    svg.line(left, top + plot_h, left + plot_w, top + plot_h, "#64748b", 1.5)
    svg.line(left, top, left, top + plot_h, "#64748b", 1.5)
    svg.text(left + plot_w / 2, top + plot_h + 53, "Training samples", 15, "#334155", "middle")
    svg.text(85, top + plot_h / 2, "Validation MSE (log scale)", 15, "#334155", "middle")
    colors = {"linear": "#087e8b", "sigmoid": "#d97706"}
    for name in ("linear", "sigmoid"):
        values = validation_by_size[name]
        pts = [(sx(size), sy(float(values[size]))) for size in x_values]
        svg.path(pts, colors[name], 3)
        for px, py in pts:
            svg.circle(px, py, 5, colors[name])
    _legend(svg, 435, 460, [("Linear output", colors["linear"]), ("Sigmoid output", colors["sigmoid"])])
    lval = float(linear["results_by_training_size"]["4000"]["validation_mse"])
    sval = float(sigmoid["results_by_training_size"]["4000"]["validation_mse"])
    svg.text(95, 870, f"At N=4000: Linear {lval:.6g}; Sigmoid {sval:.6g} (Linear is {(sval-lval)/sval:.1%} lower).", 14, "#334155")

    # Epoch-level validation trajectory at the selected dataset size.
    _panel(svg, 755, 380, 700, 535, "(c) Validation-loss history at N = 4000")
    h_linear = histories["linear"]
    h_sigmoid = histories["sigmoid"]
    all_vals = [float(row["validation_mse"]) for row in [*h_linear, *h_sigmoid]]
    y_min = max(min(all_vals) * 0.65, 1e-6)
    y_max = max(all_vals) * 1.3
    sx2, sy2 = _plot_axes(svg, 775, 425, 660, 405, 1, max(len(h_linear), len(h_sigmoid)), y_min, y_max, "Epoch", "Validation MSE (log scale)", [(1, "1"), (300, "300"), (600, "600"), (900, "900"), (1200, "1200")], [])
    # Log map is used explicitly for epoch-loss visual clarity.
    def log_sy(value: float) -> float:
        top, bottom = 490, 765
        return bottom - (np.log10(value) - np.log10(y_min)) / (np.log10(y_max) - np.log10(y_min)) * (bottom - top)
    # Replace the generic vertical grid with labeled logarithmic ticks.
    for value in np.geomspace(y_min, y_max, 5):
        py = log_sy(float(value))
        svg.line(845, py, 1408, py, "#e8edf3", 1)
        svg.text(835, py + 5, f"{value:.1e}", 12, "#64748b", "end")
    for name, history in (("linear", h_linear), ("sigmoid", h_sigmoid)):
        pts = [(sx2(float(row["epoch"])), log_sy(float(row["validation_mse"]))) for row in history[::4]]
        if history and history[-1] not in history[::4]:
            pts.append((sx2(float(history[-1]["epoch"])), log_sy(float(history[-1]["validation_mse"]))))
        svg.path(pts, colors[name], 2.5)
    _legend(svg, 1190, 460, [("Linear", colors["linear"]), ("Sigmoid", colors["sigmoid"])])
    linear_4000 = linear["results_by_training_size"]["4000"]
    sigmoid_4000 = sigmoid["results_by_training_size"]["4000"]
    svg.text(790, 870, f"At N=4000 both runs early-stopped: Linear best/stop {linear_4000['best_epoch']}/{linear_4000['epochs_run']}; Sigmoid {sigmoid_4000['best_epoch']}/{sigmoid_4000['epochs_run']}.", 13, "#334155")
    svg.text(790, 890, "Some smaller-N runs reached the 2500-epoch cap; interpret their final losses as cap-limited.", 13, "#9a3412")
    svg.save(output)


def _figure3(
    output: Path,
    wavelengths: np.ndarray,
    true_test: np.ndarray,
    pred_test: np.ndarray,
    candidate_predictions: np.ndarray,
    candidate_rows: dict[str, list[dict[str, str]]],
    verified_spectra: dict[str, np.ndarray],
    candidate_spectra: np.ndarray,
    oob_count: int,
) -> None:
    svg = Svg(1500, 1040)
    _header(svg, "03", "Predictions are checked against physics", "Held-out test  ·  candidate ranking  ·  fresh TMM calculations")

    # Test parity at the task's target wavelength.
    _panel(svg, 45, 105, 680, 420, "(a) Held-out test set at λ = 700 nm")
    column = int(np.flatnonzero(wavelengths == 700)[0])
    actual = true_test[:, column]
    predicted = pred_test[:, column]
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    sx, sy = _plot_axes(svg, 65, 150, 640, 320, low, high, low, high, "TMM reflectance", "MLP reflectance", [(low, f"{low:.2f}"), ((low+high)/2, f"{(low+high)/2:.2f}"), (high, f"{high:.2f}")], [(low, f"{low:.2f}"), ((low+high)/2, f"{(low+high)/2:.2f}"), (high, f"{high:.2f}")])
    svg.line(sx(low), sy(low), sx(high), sy(high), "#94a3b8", 2, "6,5")
    for xval, yval in zip(actual, predicted):
        svg.circle(sx(float(xval)), sy(float(yval)), 2.6, "#087e8b")
    svg.text(90, 493, f"n = {len(actual)} held-out samples; full-spectrum test RMSE is reported in the metrics JSON.", 13, "#526579")

    # Top-ten surrogate ranking errors at target wavelength.
    _panel(svg, 755, 105, 700, 420, "(b) MLP-ranked candidates: target reflectance check")
    xy_vals = []
    for objective in ("min", "max"):
        for row in candidate_rows[objective]:
            xy_vals.append((float(row["mlp_reflectance_at_target"]), float(row["tmm_reflectance_at_target"]), objective))
    xmin = min(0.0, *(x for x, _, _ in xy_vals))
    xmax = max(x for x, _, _ in xy_vals) * 1.05
    ymin = min(0.0, *(y for _, y, _ in xy_vals))
    ymax = max(y for _, y, _ in xy_vals) * 1.05
    sx2, sy2 = _plot_axes(svg, 775, 150, 660, 320, xmin, xmax, ymin, ymax, "MLP prediction at 700 nm", "TMM verification at 700 nm", [(xmin, f"{xmin:.2f}"), (xmax, f"{xmax:.2f}")], [(ymin, f"{ymin:.2f}"), (ymax, f"{ymax:.2f}")])
    grid_min, grid_max = max(xmin, ymin), min(xmax, ymax)
    if grid_min < grid_max:
        svg.line(sx2(grid_min), sy2(grid_min), sx2(grid_max), sy2(grid_max), "#94a3b8", 2, "6,5")
    for px, py, objective in xy_vals:
        color = "#2563eb" if objective == "min" else "#d97706"
        svg.circle(sx2(px), sy2(py), 5, color)
    _legend(svg, 1185, 185, [("Top 10: minimize R", "#2563eb"), ("Top 10: maximize R", "#d97706"), ("Ideal agreement", "#94a3b8")])
    svg.text(790, 493, "R̂ outside [0,1]:", 13, "#526579")
    svg.rect(915, 472, 260, 34, "#fff4e5", "#f0c98a", 9)
    svg.text(1045, 495, f"{oob_count} / 10,000 candidates  ·  {oob_count/10000:.2%}", 13, "#9a3412", "middle", "bold")
    svg.text(1190, 493, "These are surrogate predictions, not physical reflectance.", 11, "#64748b")

    # Full spectra for the best TMM-verified candidate under both objectives.
    _panel(svg, 45, 550, 1410, 430, "03C   Best TMM-verified design under each objective")
    left, top, plot_w, plot_h = 125, 625, 1240, 255
    sx3 = lambda value: left + (value - 400) / 400 * plot_w
    sy3 = lambda value: top + plot_h - value * plot_h
    for wavelength in (400, 500, 600, 700, 800):
        px = sx3(wavelength)
        svg.line(px, top, px, top + plot_h, "#e8edf3", 1)
        svg.text(px, top + plot_h + 25, str(wavelength), 13, "#64748b", "middle")
    for value in (0, 0.25, 0.5, 0.75, 1):
        py = sy3(value)
        svg.line(left, py, left + plot_w, py, "#e8edf3", 1)
        svg.text(left - 12, py + 5, f"{value:.2f}", 13, "#64748b", "end")
    svg.line(left, top + plot_h, left + plot_w, top + plot_h, "#64748b", 1.5)
    svg.line(left, top, left, top + plot_h, "#64748b", 1.5)
    svg.line(sx3(700), top, sx3(700), top + plot_h, "#e6a23c", 1.5, "5,5")
    for objective, color_tmm, color_mlp in (("min", "#087e8b", "#5eead4"), ("max", "#b45309", "#fdba74")):
        rows = candidate_rows[objective]
        best = next(row for row in rows if int(row["tmm_rank_within_mlp_top10"]) == 1)
        local_index = int(best["mlp_rank"]) - 1
        mlp_spectrum = candidate_spectra[int(best["candidate_index"])]
        tmm_spectrum = verified_spectra[objective][local_index]
        svg.path([(sx3(float(w)), sy3(float(r))) for w, r in zip(wavelengths, mlp_spectrum)], color_mlp, 2.5)
        svg.path([(sx3(float(w)), sy3(float(r))) for w, r in zip(wavelengths, tmm_spectrum)], color_tmm, 3)
        svg.text(105, 920, f"{objective}: d = ({float(best['d1_nm']):.1f}, {float(best['d2_nm']):.1f}, {float(best['d3_nm']):.1f}, {float(best['d4_nm']):.1f}) nm; TMM R700 = {float(best['tmm_reflectance_at_target']):.4f}", 13, color_tmm)
    svg.text(750, 955, "Solid: TMM verification · pale: selected MLP prediction · colors distinguish low- and high-reflectance objectives", 14, "#526579", "middle")
    _legend(svg, 1150, 580, [("Min objective · TMM", "#087e8b"), ("Min objective · MLP", "#5eead4"), ("Max objective · TMM", "#b45309"), ("Max objective · MLP", "#fdba74")])
    svg.save(output)


def _load_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_design_rows(path: Path) -> list[dict[str, str]]:
    return _read_csv(path)


def _validation_values_by_size(
    report: dict,
    models_dir: Path,
    expected_activation: str,
) -> dict[int, float]:
    """Read validation MSE from reports, falling back to saved partial-run checkpoints."""
    values: dict[int, float] = {}
    for size in (500, 1000, 2000, 4000):
        row = report["results_by_training_size"].get(str(size))
        if row is not None:
            values[size] = float(row["validation_mse"])
            continue
        checkpoint_path = models_dir / f"mlp_train_{size}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing validation metrics and checkpoint for training size {size}: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("training_size") != size:
            raise ValueError(f"checkpoint has an unexpected training size: {checkpoint_path}")
        if checkpoint.get("output_activation") != expected_activation:
            raise ValueError(f"checkpoint has an unexpected output activation: {checkpoint_path}")
        values[size] = float(checkpoint["validation_mse"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/thinfilm_dataset.npz")
    parser.add_argument("--linear-metrics", default="outputs_convergence_linear/mlp_metrics.json")
    parser.add_argument("--sigmoid-metrics", default="outputs_convergence_sigmoid/mlp_metrics.json")
    parser.add_argument("--checkpoint", default="models_convergence_linear/mlp_train_4000.pt")
    parser.add_argument("--linear-output-dir", default="outputs_convergence_linear")
    parser.add_argument("--sigmoid-output-dir", default="outputs_convergence_sigmoid")
    parser.add_argument("--linear-models-dir", default="models_convergence_linear")
    parser.add_argument("--sigmoid-models-dir", default="models_convergence_sigmoid")
    parser.add_argument("--output-dir", default="outputs_convergence_linear/figures")
    args = parser.parse_args()

    config = STUDY_CONFIG
    data_path = Path(args.dataset)
    linear_output = Path(args.linear_output_dir)
    output_dir = Path(args.output_dir)
    design_required = [
        linear_output / "design_screening_predictions.npz",
        linear_output / "design_top10_tmm_spectra.npz",
        linear_output / "design_top10_min.csv",
        linear_output / "design_top10_max.csv",
    ]
    missing = [str(path) for path in design_required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run scripts/screen_designs.py with the selected 4000-sample linear "
            f"checkpoint first; missing: {', '.join(missing)}"
        )

    with np.load(data_path, allow_pickle=False) as archive:
        thicknesses = archive["thicknesses_nm"]
        spectra = archive["reflectance"]
        train_indices = archive["train_indices"]
        test_indices = archive["test_indices"]
        wavelengths = archive["wavelengths_nm"]
    linear_report = _load_report(Path(args.linear_metrics))
    sigmoid_report = _load_report(Path(args.sigmoid_metrics))
    if linear_report["model"]["output_activation"] != "linear":
        raise ValueError("linear metrics report does not contain a linear model")
    if sigmoid_report["model"]["output_activation"] != "sigmoid":
        raise ValueError("sigmoid metrics report does not contain a sigmoid model")
    if linear_report["split_sizes"] != sigmoid_report["split_sizes"]:
        raise ValueError("linear and sigmoid metrics use different data splits")
    linear_settings = dict(linear_report["model"])
    sigmoid_settings = dict(sigmoid_report["model"])
    linear_settings.pop("output_activation", None)
    sigmoid_settings.pop("output_activation", None)
    if linear_settings != sigmoid_settings:
        raise ValueError("linear and sigmoid training settings do not match")

    sample_index = int(train_indices[0])
    sample = {
        "thicknesses_nm": thicknesses[sample_index],
        "wavelengths_nm": wavelengths,
        "reflectance": spectra[sample_index],
    }

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("output_activation") != "linear" or checkpoint.get("training_size") != 4000:
        raise ValueError("selected checkpoint must be the 4000-sample linear model")
    model = MLPRegressor(output_activation="linear")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(1)
    x = ((thicknesses - config.thickness_min_nm) / (config.thickness_max_nm - config.thickness_min_nm)).astype(np.float32)
    with torch.no_grad():
        pred_test = model(torch.from_numpy(x[test_indices])).cpu().numpy()

    with np.load(linear_output / "design_screening_predictions.npz", allow_pickle=False) as archive:
        candidates = archive["thicknesses_nm"]
        candidate_spectra = archive["predicted_reflectance"]
        design_wavelengths = archive["wavelengths_nm"]
    if not np.array_equal(wavelengths, design_wavelengths):
        raise ValueError("design-screening and dataset wavelength grids differ")
    candidate_rows = {
        objective: _load_design_rows(linear_output / f"design_top10_{objective}.csv")
        for objective in ("min", "max")
    }
    with np.load(linear_output / "design_top10_tmm_spectra.npz", allow_pickle=False) as archive:
        verified_spectra = {
            "min": archive["min_objective_spectra"],
            "max": archive["max_objective_spectra"],
        }
    target_column = int(np.flatnonzero(wavelengths == config.target_wavelength_nm)[0])
    oob_count = int(np.count_nonzero((candidate_spectra[:, target_column] < 0) | (candidate_spectra[:, target_column] > 1)))

    output_dir.mkdir(parents=True, exist_ok=True)
    _figure1(
        output_dir / "figure_1_workflow_and_tmm.svg",
        sample,
        int(linear_report["model"]["max_epochs"]),
    )
    _figure2(
        output_dir / "figure_2_mlp_and_training.svg",
        linear_report,
        sigmoid_report,
        {
            "linear": _read_csv(Path(args.linear_output_dir) / "loss_history_train_4000.csv"),
            "sigmoid": _read_csv(Path(args.sigmoid_output_dir) / "loss_history_train_4000.csv"),
        },
        {
            "linear": _validation_values_by_size(
                linear_report, Path(args.linear_models_dir), "linear"
            ),
            "sigmoid": _validation_values_by_size(
                sigmoid_report, Path(args.sigmoid_models_dir), "sigmoid"
            ),
        },
    )
    _figure3(
        output_dir / "figure_3_prediction_and_design.svg",
        wavelengths,
        spectra[test_indices],
        pred_test,
        candidate_spectra,
        candidate_rows,
        verified_spectra,
        candidate_spectra,
        oob_count,
    )
    manifest = {
        "status": "success",
        "summary": "Three SVG assignment figures generated from repository artifacts",
        "files": [
            "figure_1_workflow_and_tmm.svg",
            "figure_2_mlp_and_training.svg",
            "figure_3_prediction_and_design.svg",
        ],
        "sources": [
            str(data_path),
            args.linear_metrics,
            args.sigmoid_metrics,
            args.checkpoint,
            str(Path(args.sigmoid_models_dir) / "mlp_train_500.pt"),
            str(Path(args.sigmoid_models_dir) / "mlp_train_1000.pt"),
            str(Path(args.sigmoid_models_dir) / "mlp_train_2000.pt"),
            *(str(path) for path in design_required),
        ],
        "seed": config.seed,
        "design_seed": config.design_seed,
        "target_wavelength_nm": config.target_wavelength_nm,
        "selected_output_activation": "linear",
        "selected_training_size": 4000,
        "verified_screening_candidates": len(candidates),
        "candidate_predictions_outside_0_1_at_target": oob_count,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
