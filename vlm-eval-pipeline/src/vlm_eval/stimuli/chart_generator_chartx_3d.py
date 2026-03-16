"""
3D chart generator using real-world data from the ChartX dataset.

Renders bar, line, area, and heatmap in 3D matplotlib from ChartX CSV data,
falling back to synthetic 3D generation for scatter and stacked_bar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from vlm_eval.stimuli.chart_generator import (
    BenchmarkItem,
    ChartGroundTruth,
    save_chart_with_metadata,
)
from vlm_eval.stimuli.chart_generator_3d import (
    _setup_3d_axes,
    generate_scatter_chart_3d,
    generate_stacked_bar_3d,
)
from vlm_eval.stimuli.chartx_loader import load_chartx


# ---------------------------------------------------------------------------
# 3D Bar chart from ChartX data
# ---------------------------------------------------------------------------

def generate_bar_chart_chartx_3d(
    chart_id: str, output_dir: str | Path, data: dict[str, Any],
) -> BenchmarkItem:
    output_dir = Path(output_dir)
    categories = data["categories"]
    first_series = data["series_names"][0]
    values = np.array(data["series_data"][first_series])
    n_cats = len(categories)

    colors = sns.color_palette("Set2", n_cats)
    fig = plt.figure(figsize=(9, 6))
    ax = _setup_3d_axes(fig, elev=20, azim=-50)

    x_pos = np.arange(n_cats)
    dx, dy = 0.6, 0.4
    for i in range(n_cats):
        ax.bar3d(x_pos[i], 0, 0, dx, dy, float(values[i]),
                 color=colors[i], alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.text(x_pos[i] + dx / 2, dy / 2, float(values[i]) + max(values) * 0.02,
                f"{values[i]:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x_pos + dx / 2)
    ax.set_xticklabels(categories, fontsize=7, rotation=15)
    ax.set_yticks([])
    ax.set_zlabel("Value", fontsize=9)
    ax.set_title(data.get("title", "3D Category Comparison"), fontsize=11, pad=15)
    ax.set_zlim(0, max(values) * 1.2)

    # Ground truth
    tallest_idx = int(np.argmax(values))
    tallest_cat = categories[tallest_idx]
    tallest_val = float(values[tallest_idx])
    rng = np.random.default_rng(hash(chart_id) % (2**31))
    retrieval_idx = int(rng.integers(0, n_cats))
    retrieval_cat = categories[retrieval_idx]
    retrieval_val = float(values[retrieval_idx])
    cmp_indices = rng.choice(n_cats, size=2, replace=False)
    cmp_a, cmp_b = int(cmp_indices[0]), int(cmp_indices[1])
    larger_cat = categories[cmp_a] if values[cmp_a] > values[cmp_b] else categories[cmp_b]

    gt = ChartGroundTruth(
        chart_type="bar_3d",
        data_values={"categories": categories, "values": values.tolist()},
        ground_truth_answers={
            "extremum_detection": {"answer": tallest_cat, "value": tallest_val,
                                    "description": f"The tallest bar is {tallest_cat} with value {tallest_val}"},
            "value_retrieval": {"category": retrieval_cat, "answer": retrieval_val,
                                "description": f"The value of {retrieval_cat} is {retrieval_val}"},
            "value_comparison": {"bar_a": categories[cmp_a], "bar_b": categories[cmp_b],
                                  "answer": larger_cat, "description": f"{larger_cat} is larger"},
        },
        metadata={"n_categories": n_cats, "source": "chartx"},
    )
    image_path = save_chart_with_metadata(fig, chart_id, "bar_3d", gt, output_dir)
    questions = [
        {"task_type": "extremum_detection", "question": "In this 3D bar chart, which bar is the tallest, and what is its value?"},
        {"task_type": "value_retrieval", "question": f"In this 3D bar chart, what is the value of the bar labeled '{retrieval_cat}'?"},
        {"task_type": "value_comparison", "question": f"In this 3D bar chart, which bar is larger: '{categories[cmp_a]}' or '{categories[cmp_b]}'?"},
    ]
    return BenchmarkItem(chart_id=chart_id, chart_type="bar_3d", image_path=str(image_path), ground_truth=gt, questions=questions)


# ---------------------------------------------------------------------------
# 3D Line chart from ChartX data
# ---------------------------------------------------------------------------

def generate_line_chart_chartx_3d(
    chart_id: str, output_dir: str | Path, data: dict[str, Any],
) -> BenchmarkItem:
    output_dir = Path(output_dir)
    x_labels = data["x_labels"]
    series_names = data["series_names"]
    series_data = data["series_data"]
    n_points = len(x_labels)
    x = np.arange(n_points)

    palette = sns.color_palette("tab10", len(series_names))
    markers = ["o", "s", "^", "D"]
    fig = plt.figure(figsize=(10, 6))
    ax = _setup_3d_axes(fig, elev=25, azim=-55, focal_length=0.25)

    for i, name in enumerate(series_names):
        y = np.array(series_data[name])
        y_depth = np.full(n_points, i * 2.0)
        ax.plot(x, y_depth, y, marker=markers[i % len(markers)], markersize=4,
                label=name, color=palette[i], linewidth=1.5)

    ax.set_xticks(x[::max(1, n_points // 6)])
    ax.set_xticklabels([x_labels[j] for j in range(0, n_points, max(1, n_points // 6))], fontsize=7)
    if len(series_names) > 1:
        ax.set_yticks([i * 2.0 for i in range(len(series_names))])
        ax.set_yticklabels(series_names, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_zlabel("Value", fontsize=9)
    ax.set_title(data.get("title", "3D Trend Over Time"), fontsize=11, pad=15)
    if len(series_names) > 1:
        ax.legend(fontsize=8, loc="upper left")

    primary = series_names[0]
    primary_vals = np.array(series_data[primary])
    peak_idx = int(np.argmax(primary_vals))
    peak_val = float(primary_vals[peak_idx])
    peak_time = x_labels[peak_idx]
    slope = np.polyfit(x, primary_vals, 1)[0]
    trend = "increasing" if slope > 0.5 else ("decreasing" if slope < -0.5 else "stable")
    rng = np.random.default_rng(hash(chart_id) % (2**31))
    retrieval_idx = int(rng.integers(0, n_points))
    retrieval_time = x_labels[retrieval_idx]
    retrieval_val = float(primary_vals[retrieval_idx])

    gt = ChartGroundTruth(
        chart_type="line_3d",
        data_values={"x_labels": x_labels, "series": {n: series_data[n] for n in series_names}},
        ground_truth_answers={
            "trend_identification": {"series": primary, "answer": trend, "slope": round(float(slope), 3),
                                      "description": f"The overall trend of {primary} is {trend}"},
            "extremum_detection": {"series": primary, "answer": peak_time, "value": peak_val,
                                    "description": f"Peak value of {primary} is {peak_val} at {peak_time}"},
            "value_retrieval": {"series": primary, "time_point": retrieval_time, "answer": retrieval_val,
                                "description": f"Value of {primary} at {retrieval_time} is {retrieval_val}"},
        },
        metadata={"n_series": len(series_names), "n_points": n_points, "source": "chartx"},
    )
    image_path = save_chart_with_metadata(fig, chart_id, "line_3d", gt, output_dir)
    questions = [
        {"task_type": "trend_identification", "question": f"In this 3D line chart, what is the overall trend of '{primary}'?"},
        {"task_type": "extremum_detection", "question": f"In this 3D line chart, at which time point does '{primary}' reach its peak value, and what is that value?"},
        {"task_type": "value_retrieval", "question": f"In this 3D line chart, what is the value of '{primary}' at '{retrieval_time}'?"},
    ]
    return BenchmarkItem(chart_id=chart_id, chart_type="line_3d", image_path=str(image_path), ground_truth=gt, questions=questions)


# ---------------------------------------------------------------------------
# 3D Heatmap (surface) from ChartX data
# ---------------------------------------------------------------------------

def generate_heatmap_chartx_3d(
    chart_id: str, output_dir: str | Path, data: dict[str, Any],
) -> BenchmarkItem:
    output_dir = Path(output_dir)
    row_labels = data["row_labels"]
    col_labels = data["col_labels"]
    values = np.array(data["values"])
    n_rows, n_cols = values.shape

    max_idx = np.unravel_index(int(np.argmax(values)), values.shape)
    max_row, max_col = row_labels[max_idx[0]], col_labels[max_idx[1]]
    max_val = float(values[max_idx])
    rng = np.random.default_rng(hash(chart_id) % (2**31))
    r_idx, c_idx = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    retrieval_row, retrieval_col = row_labels[r_idx], col_labels[c_idx]
    retrieval_val = float(values[r_idx, c_idx])
    r1, c1 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    r2, c2 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    while r1 == r2 and c1 == c2:
        r2, c2 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    cell_a, cell_b = (row_labels[r1], col_labels[c1]), (row_labels[r2], col_labels[c2])
    val_a, val_b = float(values[r1, c1]), float(values[r2, c2])
    higher_cell = cell_a if val_a >= val_b else cell_b

    fig = plt.figure(figsize=(9, 7))
    ax = _setup_3d_axes(fig, elev=30, azim=-45)
    X, Y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    surf = ax.plot_surface(X, Y, values, cmap="YlOrRd", alpha=0.85, edgecolor="gray", linewidth=0.3)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="Value")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=30)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_zlabel("Value", fontsize=9)
    ax.set_title(data.get("title", "3D Surface Plot"), fontsize=11, pad=15)

    gt = ChartGroundTruth(
        chart_type="heatmap_3d",
        data_values={"row_labels": row_labels, "col_labels": col_labels, "values": values.tolist()},
        ground_truth_answers={
            "max_value_cell": {"row": max_row, "col": max_col, "answer": max_val,
                                "description": f"The maximum value is {max_val} at ({max_row}, {max_col})"},
            "value_retrieval": {"row": retrieval_row, "col": retrieval_col, "answer": retrieval_val,
                                "description": f"The value at ({retrieval_row}, {retrieval_col}) is {retrieval_val}"},
            "comparison": {"cell_a": {"row": cell_a[0], "col": cell_a[1], "value": val_a},
                           "cell_b": {"row": cell_b[0], "col": cell_b[1], "value": val_b},
                           "answer": f"({higher_cell[0]}, {higher_cell[1]})",
                           "description": f"Cell ({higher_cell[0]}, {higher_cell[1]}) is higher"},
        },
        metadata={"n_rows": n_rows, "n_cols": n_cols, "source": "chartx"},
    )
    image_path = save_chart_with_metadata(fig, chart_id, "heatmap_3d", gt, output_dir)
    questions = [
        {"task_type": "max_value_cell", "question": "In this 3D surface plot, which position (row, column) has the highest peak, and what is its value?"},
        {"task_type": "value_retrieval", "question": f"In this 3D surface plot, what is the approximate value at position ({retrieval_row}, {retrieval_col})?"},
        {"task_type": "comparison", "question": f"In this 3D surface plot, which position has a higher value: ({cell_a[0]}, {cell_a[1]}) or ({cell_b[0]}, {cell_b[1]})?"},
    ]
    return BenchmarkItem(chart_id=chart_id, chart_type="heatmap_3d", image_path=str(image_path), ground_truth=gt, questions=questions)


# ---------------------------------------------------------------------------
# 3D Area chart from ChartX data
# ---------------------------------------------------------------------------

def generate_area_chart_chartx_3d(
    chart_id: str, output_dir: str | Path, data: dict[str, Any],
) -> BenchmarkItem:
    output_dir = Path(output_dir)
    x_labels = data["x_labels"]
    series_names = data["series_names"][:2]
    series_data = data["series_data"]
    n_points = len(x_labels)
    x = np.arange(n_points)

    palette = sns.color_palette("pastel", len(series_names))
    fig = plt.figure(figsize=(10, 6))
    ax = _setup_3d_axes(fig, elev=25, azim=-55, focal_length=0.25)

    for i, name in enumerate(series_names):
        y = np.array(series_data[name])
        y_depth = i * 3.0
        ax.plot(x, np.full(n_points, y_depth), y, color=palette[i], linewidth=1.5, label=name)
        verts = []
        for j in range(n_points - 1):
            verts.append([(x[j], y_depth, 0), (x[j], y_depth, float(y[j])),
                          (x[j+1], y_depth, float(y[j+1])), (x[j+1], y_depth, 0)])
        poly = Poly3DCollection(verts, alpha=0.35, facecolor=palette[i], edgecolor="none")
        ax.add_collection3d(poly)

    ax.set_xticks(x[::max(1, n_points // 6)])
    ax.set_xticklabels([x_labels[j] for j in range(0, n_points, max(1, n_points // 6))], fontsize=7)
    if len(series_names) > 1:
        ax.set_yticks([i * 3.0 for i in range(len(series_names))])
        ax.set_yticklabels(series_names, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_zlabel("Value", fontsize=9)
    ax.set_title(data.get("title", "3D Area Chart"), fontsize=11, pad=15)
    if len(series_names) > 1:
        ax.legend(fontsize=8, loc="upper left")

    primary = series_names[0]
    primary_vals = np.array(series_data[primary])
    slope = np.polyfit(x, primary_vals, 1)[0]
    trend_dir = "increasing" if slope > 0.5 else ("decreasing" if slope < -0.5 else "stable")
    max_idx = int(np.argmax(primary_vals))
    max_val = float(primary_vals[max_idx])
    max_time = x_labels[max_idx]

    if len(series_names) == 2:
        secondary = series_names[1]
        sec_vals = np.array(series_data[secondary])
        primary_mean, secondary_mean = float(np.mean(primary_vals)), float(np.mean(sec_vals))
        larger_series = primary if primary_mean >= secondary_mean else secondary
    else:
        primary_mean = float(np.mean(primary_vals))
        larger_series, secondary_mean = primary, None

    gt = ChartGroundTruth(
        chart_type="area_3d",
        data_values={"x_labels": x_labels, "series": {n: series_data[n] for n in series_names}},
        ground_truth_answers={
            "trend_identification": {"series": primary, "answer": trend_dir, "slope": round(float(slope), 3),
                                      "description": f"The trend of {primary} is {trend_dir}"},
            "max_value": {"series": primary, "answer": max_val, "time_point": max_time,
                          "description": f"The max value of {primary} is {max_val} at {max_time}"},
            "magnitude_comparison": {"answer": larger_series, "primary_mean": round(primary_mean, 2),
                                      "secondary_mean": round(secondary_mean, 2) if secondary_mean is not None else None,
                                      "description": f"{larger_series} has a larger average magnitude"},
        },
        metadata={"n_series": len(series_names), "n_points": n_points, "source": "chartx"},
    )
    image_path = save_chart_with_metadata(fig, chart_id, "area_3d", gt, output_dir)
    questions = [
        {"task_type": "trend_identification", "question": f"In this 3D area chart, what is the overall trend of '{primary}'?"},
        {"task_type": "max_value", "question": f"In this 3D area chart, what is the maximum value of '{primary}' and when does it occur?"},
        {"task_type": "magnitude_comparison", "question": "In this 3D area chart, which series has the larger overall magnitude?"},
    ]
    return BenchmarkItem(chart_id=chart_id, chart_type="area_3d", image_path=str(image_path), ground_truth=gt, questions=questions)


# ---------------------------------------------------------------------------
# Registry and batch generation
# ---------------------------------------------------------------------------

CHARTX_3D_GENERATORS = {
    "bar": generate_bar_chart_chartx_3d,
    "line": generate_line_chart_chartx_3d,
    "heatmap": generate_heatmap_chartx_3d,
    "area": generate_area_chart_chartx_3d,
}


def generate_benchmark_dataset_chartx_3d(
    chart_types: list[str] | None = None,
    n_per_type: int = 50,
    output_base_dir: str | Path = "data/charts_chartx_3d",
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Generate 3D benchmark from ChartX data (bar/line/area/heatmap) + synthetic (scatter/stacked_bar)."""
    if chart_types is None:
        chart_types = ["bar", "line", "scatter", "heatmap", "area", "stacked_bar"]

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    chartx_types = [t for t in chart_types if t in CHARTX_3D_GENERATORS]
    synthetic_types = [t for t in chart_types if t not in CHARTX_3D_GENERATORS]

    chartx_data: dict[str, list] = {}
    if chartx_types:
        chartx_data = load_chartx(n_per_type=n_per_type)

    all_items: list[BenchmarkItem] = []
    rng = np.random.default_rng(seed)

    for chart_type in chartx_types:
        records = chartx_data.get(chart_type, [])
        generator = CHARTX_3D_GENERATORS[chart_type]
        type_dir = output_base_dir / f"{chart_type}_3d"
        for i, data_record in enumerate(records[:n_per_type]):
            chart_id = f"{chart_type}_3d_{i:04d}"
            item = generator(chart_id=chart_id, output_dir=type_dir, data=data_record)
            all_items.append(item)
        print(f"  Generated {min(len(records), n_per_type)} ChartX 3D {chart_type} charts")

    for chart_type in synthetic_types:
        type_dir = output_base_dir / f"{chart_type}_3d"
        if chart_type == "scatter":
            for i in range(n_per_type):
                chart_seed = int(rng.integers(0, 2**31))
                item = generate_scatter_chart_3d(chart_id=f"scatter_3d_{i:04d}", output_dir=type_dir, seed=chart_seed)
                all_items.append(item)
        elif chart_type == "stacked_bar":
            for i in range(n_per_type):
                chart_seed = int(rng.integers(0, 2**31))
                item = generate_stacked_bar_3d(chart_id=f"stacked_bar_3d_{i:04d}", output_dir=type_dir, seed=chart_seed)
                all_items.append(item)
        print(f"  Generated {n_per_type} synthetic 3D {chart_type} charts")

    manifest = {
        "n_items": len(all_items),
        "chart_types": [f"{t}_3d" for t in chart_types],
        "n_per_type": n_per_type,
        "seed": seed,
        "source": "chartx_hybrid",
        "condition": "3d",
        "items": [{"chart_id": item.chart_id, "chart_type": item.chart_type,
                    "image_path": item.image_path, "questions": item.questions} for item in all_items],
    }
    (output_base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return all_items
