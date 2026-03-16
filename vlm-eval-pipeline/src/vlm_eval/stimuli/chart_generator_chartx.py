"""
Chart generator using real-world data from the ChartX dataset.

Renders bar, line, area, and heatmap charts from ChartX CSV data,
falling back to synthetic generation for scatter and stacked_bar.
Produces identical JSON sidecar + manifest format as the synthetic pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from vlm_eval.stimuli.chart_generator import (
    BenchmarkItem,
    ChartGroundTruth,
    generate_scatter_chart,
    generate_stacked_bar,
    save_chart_with_metadata,
    _clean_style,
)
from vlm_eval.stimuli.chartx_loader import load_chartx


# ---------------------------------------------------------------------------
# Bar chart from ChartX data
# ---------------------------------------------------------------------------

def generate_bar_chart_chartx(
    chart_id: str,
    output_dir: str | Path,
    data: dict[str, Any],
) -> BenchmarkItem:
    """Generate a bar chart from ChartX data (first numeric series)."""
    output_dir = Path(output_dir)
    categories = data["categories"]
    # Use the first series for a simple bar chart
    first_series = data["series_names"][0]
    values = np.array(data["series_data"][first_series])

    n_cats = len(categories)
    colors = sns.color_palette("Set2", n_cats)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    title = data.get("title", "Category Comparison")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(list(data.get("series_names", ["Value"]))[0] if len(data.get("series_names", [])) == 1 else "Value", fontsize=10)
    ax.set_ylabel(first_series, fontsize=10)
    ax.set_ylim(0, max(values) * 1.15)
    _clean_style(ax)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()

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
        chart_type="bar",
        data_values={"categories": categories, "values": values.tolist()},
        ground_truth_answers={
            "extremum_detection": {
                "answer": tallest_cat,
                "value": tallest_val,
                "description": f"The tallest bar is {tallest_cat} with value {tallest_val}",
            },
            "value_retrieval": {
                "category": retrieval_cat,
                "answer": retrieval_val,
                "description": f"The value of {retrieval_cat} is {retrieval_val}",
            },
            "value_comparison": {
                "bar_a": categories[cmp_a],
                "bar_b": categories[cmp_b],
                "answer": larger_cat,
                "description": f"{larger_cat} is larger",
            },
        },
        metadata={"n_categories": n_cats, "source": "chartx", "topic": data.get("chartx_topic", "")},
    )

    image_path = save_chart_with_metadata(fig, chart_id, "bar", gt, output_dir)

    questions = [
        {"task_type": "extremum_detection", "question": "Which bar is the tallest, and what is its value?"},
        {"task_type": "value_retrieval", "question": f"What is the value of the bar labeled '{retrieval_cat}'?"},
        {"task_type": "value_comparison", "question": f"Which bar is larger: '{categories[cmp_a]}' or '{categories[cmp_b]}'?"},
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="bar",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Line chart from ChartX data
# ---------------------------------------------------------------------------

def generate_line_chart_chartx(
    chart_id: str,
    output_dir: str | Path,
    data: dict[str, Any],
) -> BenchmarkItem:
    """Generate a line chart from ChartX data."""
    output_dir = Path(output_dir)
    x_labels = data["x_labels"]
    series_names = data["series_names"]
    series_data = data["series_data"]
    n_points = len(x_labels)
    x = np.arange(n_points)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    markers = ["o", "s", "^", "D"]
    palette = sns.color_palette("tab10", len(series_names))

    for i, name in enumerate(series_names):
        y = series_data[name]
        ax.plot(x, y, marker=markers[i % len(markers)], markersize=4,
                label=name, color=palette[i], linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    title = data.get("title", "Trend Over Time")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Time Period", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    if len(series_names) > 1:
        ax.legend(fontsize=8, frameon=False)
    _clean_style(ax)
    plt.tight_layout()

    # Ground truth from first series
    primary = series_names[0]
    primary_vals = np.array(series_data[primary])
    peak_idx = int(np.argmax(primary_vals))
    peak_val = float(primary_vals[peak_idx])
    peak_time = x_labels[peak_idx]

    slope = np.polyfit(x, primary_vals, 1)[0]
    if slope > 0.5:
        trend = "increasing"
    elif slope < -0.5:
        trend = "decreasing"
    else:
        trend = "stable"

    rng = np.random.default_rng(hash(chart_id) % (2**31))
    retrieval_idx = int(rng.integers(0, n_points))
    retrieval_time = x_labels[retrieval_idx]
    retrieval_val = float(primary_vals[retrieval_idx])

    gt = ChartGroundTruth(
        chart_type="line",
        data_values={"x_labels": x_labels, "series": {n: series_data[n] for n in series_names}},
        ground_truth_answers={
            "trend_identification": {
                "series": primary,
                "answer": trend,
                "slope": round(float(slope), 3),
                "description": f"The overall trend of {primary} is {trend}",
            },
            "extremum_detection": {
                "series": primary,
                "answer": peak_time,
                "value": peak_val,
                "description": f"Peak value of {primary} is {peak_val} at {peak_time}",
            },
            "value_retrieval": {
                "series": primary,
                "time_point": retrieval_time,
                "answer": retrieval_val,
                "description": f"Value of {primary} at {retrieval_time} is {retrieval_val}",
            },
        },
        metadata={"n_series": len(series_names), "n_points": n_points, "source": "chartx", "topic": data.get("chartx_topic", "")},
    )

    image_path = save_chart_with_metadata(fig, chart_id, "line", gt, output_dir)

    questions = [
        {"task_type": "trend_identification", "question": f"What is the overall trend of '{primary}'?"},
        {"task_type": "extremum_detection", "question": f"At which time point does '{primary}' reach its peak value, and what is that value?"},
        {"task_type": "value_retrieval", "question": f"What is the value of '{primary}' at '{retrieval_time}'?"},
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="line",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Area chart from ChartX data
# ---------------------------------------------------------------------------

def generate_area_chart_chartx(
    chart_id: str,
    output_dir: str | Path,
    data: dict[str, Any],
) -> BenchmarkItem:
    """Generate an area chart from ChartX data."""
    output_dir = Path(output_dir)
    x_labels = data["x_labels"]
    series_names = data["series_names"][:2]  # use at most 2 series for area
    series_data = data["series_data"]
    n_points = len(x_labels)
    x = np.arange(n_points)

    palette = sns.color_palette("pastel", len(series_names))
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, name in enumerate(series_names):
        y = np.array(series_data[name])
        ax.fill_between(x, y, alpha=0.4, color=palette[i], label=name)
        ax.plot(x, y, color=palette[i], linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    title = data.get("title", "Area Chart")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Category", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    if len(series_names) > 1:
        ax.legend(fontsize=8, frameon=False)
    _clean_style(ax)
    plt.tight_layout()

    # Ground truth from first series
    primary = series_names[0]
    primary_vals = np.array(series_data[primary])

    slope = np.polyfit(x, primary_vals, 1)[0]
    if slope > 0.5:
        trend_dir = "increasing"
    elif slope < -0.5:
        trend_dir = "decreasing"
    else:
        trend_dir = "stable"

    max_idx = int(np.argmax(primary_vals))
    max_val = float(primary_vals[max_idx])
    max_time = x_labels[max_idx]

    if len(series_names) == 2:
        secondary = series_names[1]
        sec_vals = np.array(series_data[secondary])
        primary_mean = float(np.mean(primary_vals))
        secondary_mean = float(np.mean(sec_vals))
        larger_series = primary if primary_mean >= secondary_mean else secondary
    else:
        primary_mean = float(np.mean(primary_vals))
        larger_series = primary
        secondary_mean = None

    gt = ChartGroundTruth(
        chart_type="area",
        data_values={"x_labels": x_labels, "series": {n: series_data[n] for n in series_names}},
        ground_truth_answers={
            "trend_identification": {
                "series": primary,
                "answer": trend_dir,
                "slope": round(float(slope), 3),
                "description": f"The trend of {primary} is {trend_dir}",
            },
            "max_value": {
                "series": primary,
                "answer": max_val,
                "time_point": max_time,
                "description": f"The max value of {primary} is {max_val} at {max_time}",
            },
            "magnitude_comparison": {
                "answer": larger_series,
                "primary_mean": round(primary_mean, 2),
                "secondary_mean": round(secondary_mean, 2) if secondary_mean is not None else None,
                "description": f"{larger_series} has a larger average magnitude",
            },
        },
        metadata={"n_series": len(series_names), "n_points": n_points, "source": "chartx", "topic": data.get("chartx_topic", "")},
    )

    image_path = save_chart_with_metadata(fig, chart_id, "area", gt, output_dir)

    questions = [
        {"task_type": "trend_identification", "question": f"What is the overall trend of '{primary}' in this area chart?"},
        {"task_type": "max_value", "question": f"What is the maximum value of '{primary}' and when does it occur?"},
        {"task_type": "magnitude_comparison", "question": "Which series has the larger overall magnitude?"},
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="area",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Heatmap from ChartX data
# ---------------------------------------------------------------------------

def generate_heatmap_chartx(
    chart_id: str,
    output_dir: str | Path,
    data: dict[str, Any],
) -> BenchmarkItem:
    """Generate a heatmap from ChartX data."""
    output_dir = Path(output_dir)
    row_labels = data["row_labels"]
    col_labels = data["col_labels"]
    values = np.array(data["values"])
    n_rows, n_cols = values.shape

    # Max value cell
    max_idx = np.unravel_index(int(np.argmax(values)), values.shape)
    max_row, max_col = row_labels[max_idx[0]], col_labels[max_idx[1]]
    max_val = float(values[max_idx])

    rng = np.random.default_rng(hash(chart_id) % (2**31))
    r_idx = int(rng.integers(0, n_rows))
    c_idx = int(rng.integers(0, n_cols))
    retrieval_row, retrieval_col = row_labels[r_idx], col_labels[c_idx]
    retrieval_val = float(values[r_idx, c_idx])

    # Comparison: two distinct cells
    r1, c1 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    r2, c2 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    while r1 == r2 and c1 == c2:
        r2, c2 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    cell_a = (row_labels[r1], col_labels[c1])
    cell_b = (row_labels[r2], col_labels[c2])
    val_a, val_b = float(values[r1, c1]), float(values[r2, c2])
    higher_cell = cell_a if val_a >= val_b else cell_b

    # Build figure
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.9), max(5, n_rows * 0.8)))
    sns.heatmap(
        values,
        annot=True,
        fmt=".1f",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 7},
    )
    title = data.get("title", "Value Heatmap")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Column", fontsize=10)
    ax.set_ylabel("Row", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    gt = ChartGroundTruth(
        chart_type="heatmap",
        data_values={
            "row_labels": row_labels,
            "col_labels": col_labels,
            "values": values.tolist(),
        },
        ground_truth_answers={
            "max_value_cell": {
                "row": max_row,
                "col": max_col,
                "answer": max_val,
                "description": f"The maximum value is {max_val} at ({max_row}, {max_col})",
            },
            "value_retrieval": {
                "row": retrieval_row,
                "col": retrieval_col,
                "answer": retrieval_val,
                "description": f"The value at ({retrieval_row}, {retrieval_col}) is {retrieval_val}",
            },
            "comparison": {
                "cell_a": {"row": cell_a[0], "col": cell_a[1], "value": val_a},
                "cell_b": {"row": cell_b[0], "col": cell_b[1], "value": val_b},
                "answer": f"({higher_cell[0]}, {higher_cell[1]})",
                "description": f"Cell ({higher_cell[0]}, {higher_cell[1]}) is higher",
            },
        },
        metadata={"n_rows": n_rows, "n_cols": n_cols, "source": "chartx", "topic": data.get("chartx_topic", "")},
    )

    image_path = save_chart_with_metadata(fig, chart_id, "heatmap", gt, output_dir)

    questions = [
        {"task_type": "max_value_cell", "question": "Which cell contains the maximum value, and what is it?"},
        {"task_type": "value_retrieval", "question": f"What is the value in cell ({retrieval_row}, {retrieval_col})?"},
        {
            "task_type": "comparison",
            "question": (
                f"Which cell has a higher value: ({cell_a[0]}, {cell_a[1]}) "
                f"or ({cell_b[0]}, {cell_b[1]})?"
            ),
        },
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="heatmap",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

CHARTX_GENERATORS = {
    "bar": generate_bar_chart_chartx,
    "line": generate_line_chart_chartx,
    "area": generate_area_chart_chartx,
    "heatmap": generate_heatmap_chartx,
}


# ---------------------------------------------------------------------------
# Batch generation (main entry point)
# ---------------------------------------------------------------------------

def generate_benchmark_dataset_chartx(
    chart_types: list[str] | None = None,
    n_per_type: int = 50,
    output_base_dir: str | Path = "data/charts_chartx",
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Generate a benchmark dataset using ChartX data for bar/line/area/heatmap
    and synthetic data for scatter/stacked_bar.

    Parameters
    ----------
    chart_types:
        Which chart types to generate.  Defaults to all six.
    n_per_type:
        Number of charts per type.
    output_base_dir:
        Root directory for output.
    seed:
        Random seed for synthetic types and shuffling.
    """
    if chart_types is None:
        chart_types = ["bar", "line", "scatter", "heatmap", "area", "stacked_bar"]

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Determine which types come from ChartX vs synthetic
    chartx_types = [t for t in chart_types if t in CHARTX_GENERATORS]
    synthetic_types = [t for t in chart_types if t not in CHARTX_GENERATORS]

    # Load ChartX data for the types we need
    chartx_data: dict[str, list] = {}
    if chartx_types:
        chartx_data = load_chartx(n_per_type=n_per_type)

    all_items: list[BenchmarkItem] = []
    rng = np.random.default_rng(seed)

    # Generate ChartX-sourced charts
    for chart_type in chartx_types:
        records = chartx_data.get(chart_type, [])
        generator = CHARTX_GENERATORS[chart_type]
        type_dir = output_base_dir / chart_type

        for i, data_record in enumerate(records[:n_per_type]):
            chart_id = f"{chart_type}_{i:04d}"
            item = generator(chart_id=chart_id, output_dir=type_dir, data=data_record)
            all_items.append(item)

        print(f"  Generated {min(len(records), n_per_type)} ChartX {chart_type} charts")

    # Generate synthetic charts for unsupported types
    for chart_type in synthetic_types:
        type_dir = output_base_dir / chart_type
        if chart_type == "scatter":
            from vlm_eval.stimuli.chart_generator import generate_scatter_chart
            for i in range(n_per_type):
                chart_seed = int(rng.integers(0, 2**31))
                chart_id = f"{chart_type}_{i:04d}"
                item = generate_scatter_chart(chart_id=chart_id, output_dir=type_dir, seed=chart_seed)
                all_items.append(item)
        elif chart_type == "stacked_bar":
            from vlm_eval.stimuli.chart_generator import generate_stacked_bar
            for i in range(n_per_type):
                chart_seed = int(rng.integers(0, 2**31))
                chart_id = f"{chart_type}_{i:04d}"
                item = generate_stacked_bar(chart_id=chart_id, output_dir=type_dir, seed=chart_seed)
                all_items.append(item)
        print(f"  Generated {n_per_type} synthetic {chart_type} charts")

    # Write manifest
    manifest = {
        "n_items": len(all_items),
        "chart_types": chart_types,
        "n_per_type": n_per_type,
        "seed": seed,
        "source": "chartx_hybrid",
        "items": [
            {
                "chart_id": item.chart_id,
                "chart_type": item.chart_type,
                "image_path": item.image_path,
                "questions": item.questions,
            }
            for item in all_items
        ],
    }
    manifest_path = output_base_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return all_items
