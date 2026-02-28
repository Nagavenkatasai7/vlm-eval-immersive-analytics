"""
Chart stimulus generator for the VLM evaluation pipeline.

Generates chart images (PNG) with ground-truth metadata (JSON sidecars)
for six chart types: bar, line, scatter, heatmap, area, and stacked bar.

CS 692 Course Project
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless generation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChartGroundTruth:
    """Ground-truth information associated with a single chart."""

    chart_type: str
    data_values: dict[str, Any]
    ground_truth_answers: dict[str, Any]  # task_type -> answer
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkItem:
    """A single benchmark item: chart image + ground truth + questions."""

    chart_id: str
    chart_type: str
    image_path: str  # stored as string so it serialises to JSON easily
    ground_truth: ChartGroundTruth
    questions: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: save chart image + JSON sidecar
# ---------------------------------------------------------------------------

def save_chart_with_metadata(
    fig: plt.Figure,
    chart_id: str,
    chart_type: str,
    ground_truth: ChartGroundTruth,
    output_dir: Path,
    dpi: int = 150,
) -> Path:
    """Save a matplotlib figure as PNG and write a JSON sidecar alongside it.

    Returns the path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / f"{chart_id}.png"
    json_path = output_dir / f"{chart_id}.json"

    fig.savefig(str(image_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sidecar = {
        "chart_id": chart_id,
        "chart_type": chart_type,
        "image_file": image_path.name,
        "ground_truth": asdict(ground_truth),
    }
    json_path.write_text(json.dumps(sidecar, indent=2, default=str))

    return image_path


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

_CATEGORY_NAMES = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo",
    "Foxtrot", "Golf", "Hotel", "India", "Juliet",
]

_SERIES_NAMES = ["Series A", "Series B", "Series C", "Series D"]

_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "Jan+", "Feb+", "Mar+",
]


def _clean_style(ax: plt.Axes) -> None:
    """Apply a clean, professional look to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# Generator: Bar chart
# ---------------------------------------------------------------------------

def generate_bar_chart(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a simple vertical bar chart with 5-8 categories."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    n_cats = rng.integers(5, 9)  # 5 to 8 inclusive
    categories = _CATEGORY_NAMES[:n_cats]
    values = np.round(rng.uniform(10, 100, size=n_cats), 1)

    # Compute ground-truth answers
    tallest_idx = int(np.argmax(values))
    tallest_cat = categories[tallest_idx]
    tallest_val = float(values[tallest_idx])

    retrieval_idx = int(rng.integers(0, n_cats))
    retrieval_cat = categories[retrieval_idx]
    retrieval_val = float(values[retrieval_idx])

    # Pick two distinct bars for comparison
    cmp_indices = rng.choice(n_cats, size=2, replace=False)
    cmp_a, cmp_b = int(cmp_indices[0]), int(cmp_indices[1])
    larger_cat = categories[cmp_a] if values[cmp_a] > values[cmp_b] else categories[cmp_b]

    # Build figure
    colors = sns.color_palette("Set2", n_cats)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_title("Category Comparison", fontsize=12, pad=10)
    ax.set_xlabel("Category", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_ylim(0, max(values) * 1.15)
    _clean_style(ax)

    # Ground truth
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
        metadata={"n_categories": n_cats, "seed": seed},
    )

    image_path = save_chart_with_metadata(fig, chart_id, "bar", gt, output_dir)

    questions = [
        {"task_type": "extremum_detection", "question": "Which bar is the tallest, and what is its value?"},
        {"task_type": "value_retrieval", "question": f"What is the value of the bar labeled '{retrieval_cat}'?"},
        {
            "task_type": "value_comparison",
            "question": f"Which bar is larger: '{categories[cmp_a]}' or '{categories[cmp_b]}'?",
        },
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="bar",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Generator: Line chart
# ---------------------------------------------------------------------------

def generate_line_chart(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a line chart with 1-3 series over 8-15 time points."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    n_series = int(rng.integers(1, 4))  # 1 to 3
    n_points = int(rng.integers(8, 16))  # 8 to 15
    x_labels = _MONTH_LABELS[:n_points]
    x = np.arange(n_points)

    series_data: dict[str, list[float]] = {}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    markers = ["o", "s", "^", "D"]
    palette = sns.color_palette("tab10", n_series)

    for i in range(n_series):
        name = _SERIES_NAMES[i]
        # Random walk with a slight trend
        trend_slope = rng.uniform(-2, 2)
        noise = rng.normal(0, 5, size=n_points)
        base = rng.uniform(20, 60)
        y = np.round(base + trend_slope * x + np.cumsum(noise * 0.3), 1)
        series_data[name] = y.tolist()
        ax.plot(
            x, y,
            marker=markers[i % len(markers)],
            markersize=4,
            label=name,
            color=palette[i],
            linewidth=1.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Trend Over Time", fontsize=12, pad=10)
    ax.set_xlabel("Time Period", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    if n_series > 1:
        ax.legend(fontsize=8, frameon=False)
    _clean_style(ax)

    # Ground truth from first series
    primary = _SERIES_NAMES[0]
    primary_vals = np.array(series_data[primary])
    peak_idx = int(np.argmax(primary_vals))
    peak_val = float(primary_vals[peak_idx])
    peak_time = x_labels[peak_idx]

    # Trend direction (simple linear fit)
    slope = np.polyfit(x, primary_vals, 1)[0]
    if slope > 0.5:
        trend = "increasing"
    elif slope < -0.5:
        trend = "decreasing"
    else:
        trend = "stable"

    retrieval_idx = int(rng.integers(0, n_points))
    retrieval_time = x_labels[retrieval_idx]
    retrieval_val = float(primary_vals[retrieval_idx])

    gt = ChartGroundTruth(
        chart_type="line",
        data_values={"x_labels": x_labels, "series": series_data},
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
        metadata={"n_series": n_series, "n_points": n_points, "seed": seed},
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
# Generator: Scatter chart
# ---------------------------------------------------------------------------

def generate_scatter_chart(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a scatter plot with 2-4 clusters (multivariate normal)."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    n_clusters = int(rng.integers(2, 5))  # 2 to 4
    n_total = int(rng.integers(50, 151))  # 50 to 150 total points
    points_per_cluster = n_total // n_clusters

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    cluster_info: dict[str, Any] = {}
    palette = sns.color_palette("Set2", n_clusters)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for c in range(n_clusters):
        mean = rng.uniform(-10, 10, size=2)
        # Random covariance matrix (positive semi-definite)
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(-0.8, 0.8)
        cov = np.array([[a, b * a * 0.5], [b * a * 0.5, a]])
        pts = rng.multivariate_normal(mean, cov, size=points_per_cluster)
        all_x.append(pts[:, 0])
        all_y.append(pts[:, 1])
        label_name = f"Cluster {c + 1}"
        all_labels.append(np.full(points_per_cluster, label_name))
        cluster_info[label_name] = {
            "center": mean.tolist(),
            "n_points": points_per_cluster,
        }
        ax.scatter(
            pts[:, 0], pts[:, 1],
            label=label_name,
            color=palette[c],
            alpha=0.65,
            s=25,
            edgecolors="white",
            linewidths=0.3,
        )

    # Optionally add a few outliers
    n_outliers = int(rng.integers(0, 6))
    outlier_present = n_outliers > 0
    if outlier_present:
        ox = rng.uniform(-15, 15, size=n_outliers)
        oy = rng.uniform(-15, 15, size=n_outliers)
        ax.scatter(ox, oy, color="red", marker="x", s=50, label="Outliers", zorder=5)

    # Overall correlation direction of all points pooled
    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)
    corr = float(np.corrcoef(combined_x, combined_y)[0, 1])
    if corr > 0.3:
        corr_dir = "positive"
    elif corr < -0.3:
        corr_dir = "negative"
    else:
        corr_dir = "none"

    ax.set_title("Scatter Plot with Clusters", fontsize=12, pad=10)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="best")
    _clean_style(ax)

    gt = ChartGroundTruth(
        chart_type="scatter",
        data_values={
            "clusters": cluster_info,
            "n_total_points": int(combined_x.shape[0]),
            "n_outliers": n_outliers,
        },
        ground_truth_answers={
            "cluster_count": {
                "answer": n_clusters,
                "description": f"There are {n_clusters} clusters visible in the scatter plot",
            },
            "correlation_direction": {
                "answer": corr_dir,
                "correlation_coefficient": round(corr, 3),
                "description": f"The overall correlation direction is {corr_dir} (r={corr:.3f})",
            },
            "outlier_presence": {
                "answer": "yes" if outlier_present else "no",
                "n_outliers": n_outliers,
                "description": f"Outliers {'are' if outlier_present else 'are not'} present ({n_outliers} outlier(s))",
            },
        },
        metadata={"n_clusters": n_clusters, "n_total_points": int(combined_x.shape[0]), "seed": seed},
    )

    image_path = save_chart_with_metadata(fig, chart_id, "scatter", gt, output_dir)

    questions = [
        {"task_type": "cluster_count", "question": "How many distinct clusters are visible in this scatter plot?"},
        {"task_type": "correlation_direction", "question": "What is the overall correlation direction of the data points?"},
        {"task_type": "outlier_presence", "question": "Are there any outliers in this scatter plot?"},
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="scatter",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Generator: Heatmap
# ---------------------------------------------------------------------------

def generate_heatmap(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate an annotated heatmap with 5-8 rows and columns."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    n_rows = int(rng.integers(5, 9))
    n_cols = int(rng.integers(5, 9))
    row_labels = [f"R{i + 1}" for i in range(n_rows)]
    col_labels = [f"C{j + 1}" for j in range(n_cols)]
    data = np.round(rng.uniform(0, 100, size=(n_rows, n_cols)), 1)

    # Max value cell
    max_idx = np.unravel_index(int(np.argmax(data)), data.shape)
    max_row, max_col = row_labels[max_idx[0]], col_labels[max_idx[1]]
    max_val = float(data[max_idx])

    # Value retrieval for a specific random cell
    r_idx = int(rng.integers(0, n_rows))
    c_idx = int(rng.integers(0, n_cols))
    retrieval_row, retrieval_col = row_labels[r_idx], col_labels[c_idx]
    retrieval_val = float(data[r_idx, c_idx])

    # Comparison: pick two distinct cells
    r1, c1 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    r2, c2 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    while r1 == r2 and c1 == c2:
        r2, c2 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    cell_a = (row_labels[r1], col_labels[c1])
    cell_b = (row_labels[r2], col_labels[c2])
    val_a, val_b = float(data[r1, c1]), float(data[r2, c2])
    higher_cell = cell_a if val_a >= val_b else cell_b

    # Build figure
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.9), max(5, n_rows * 0.8)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Value Heatmap", fontsize=12, pad=10)
    ax.set_xlabel("Column", fontsize=10)
    ax.set_ylabel("Row", fontsize=10)
    ax.tick_params(labelsize=9)

    gt = ChartGroundTruth(
        chart_type="heatmap",
        data_values={
            "row_labels": row_labels,
            "col_labels": col_labels,
            "values": data.tolist(),
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
        metadata={"n_rows": n_rows, "n_cols": n_cols, "seed": seed},
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
# Generator: Area chart
# ---------------------------------------------------------------------------

def generate_area_chart(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate an area chart with 1-2 series using fill_between."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    n_series = int(rng.integers(1, 3))  # 1 or 2
    n_points = int(rng.integers(10, 16))
    x = np.arange(n_points)
    x_labels = _MONTH_LABELS[:n_points]

    series_data: dict[str, list[float]] = {}
    palette = sns.color_palette("pastel", n_series)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i in range(n_series):
        name = _SERIES_NAMES[i]
        base = rng.uniform(15, 50)
        trend = rng.uniform(-1.0, 1.5)
        noise = rng.normal(0, 3, size=n_points)
        y = np.round(base + trend * x + np.cumsum(noise * 0.4), 1)
        y = np.clip(y, 0, None)  # no negative values
        series_data[name] = y.tolist()
        ax.fill_between(x, y, alpha=0.4, color=palette[i], label=name)
        ax.plot(x, y, color=palette[i], linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Area Chart", fontsize=12, pad=10)
    ax.set_xlabel("Time Period", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    if n_series > 1:
        ax.legend(fontsize=8, frameon=False)
    _clean_style(ax)

    # Ground truth from first series
    primary = _SERIES_NAMES[0]
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

    # Magnitude comparison (only meaningful if 2 series)
    if n_series == 2:
        secondary = _SERIES_NAMES[1]
        sec_vals = np.array(series_data[secondary])
        primary_mean = float(np.mean(primary_vals))
        secondary_mean = float(np.mean(sec_vals))
        larger_series = primary if primary_mean >= secondary_mean else secondary
        magnitude_desc = f"{larger_series} has a larger average magnitude"
    else:
        primary_mean = float(np.mean(primary_vals))
        larger_series = primary
        secondary_mean = None
        magnitude_desc = "Only one series present"

    gt = ChartGroundTruth(
        chart_type="area",
        data_values={"x_labels": x_labels, "series": series_data},
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
                "description": magnitude_desc,
            },
        },
        metadata={"n_series": n_series, "n_points": n_points, "seed": seed},
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
# Generator: Stacked bar chart
# ---------------------------------------------------------------------------

def generate_stacked_bar(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a stacked bar chart with 4-6 categories and 2-3 segments."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    n_cats = int(rng.integers(4, 7))     # 4 to 6
    n_segments = int(rng.integers(2, 4))  # 2 or 3
    categories = _CATEGORY_NAMES[:n_cats]
    segment_names = [f"Segment {chr(65 + i)}" for i in range(n_segments)]  # A, B, C

    # Generate data: rows = segments, cols = categories
    data = np.round(rng.uniform(5, 40, size=(n_segments, n_cats)), 1)
    totals = data.sum(axis=0)

    # Total comparison: which category has the largest total
    max_total_idx = int(np.argmax(totals))
    max_total_cat = categories[max_total_idx]
    max_total_val = float(totals[max_total_idx])

    # Part to whole: pick a random segment and category
    seg_idx = int(rng.integers(0, n_segments))
    cat_idx = int(rng.integers(0, n_cats))
    part_val = float(data[seg_idx, cat_idx])
    whole_val = float(totals[cat_idx])
    fraction = round(part_val / whole_val, 3) if whole_val > 0 else 0.0

    # Build figure
    palette = sns.color_palette("Set2", n_segments)
    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(n_cats)

    for s in range(n_segments):
        ax.bar(
            categories,
            data[s],
            bottom=bottom,
            label=segment_names[s],
            color=palette[s],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += data[s]

    ax.set_title("Stacked Bar Chart", fontsize=12, pad=10)
    ax.set_xlabel("Category", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_ylim(0, max(totals) * 1.12)
    _clean_style(ax)

    # Store data as dict for JSON
    segment_data = {
        segment_names[s]: data[s].tolist() for s in range(n_segments)
    }

    gt = ChartGroundTruth(
        chart_type="stacked_bar",
        data_values={
            "categories": categories,
            "segments": segment_data,
            "totals": totals.tolist(),
        },
        ground_truth_answers={
            "total_comparison": {
                "answer": max_total_cat,
                "total_value": max_total_val,
                "description": f"{max_total_cat} has the largest total ({max_total_val})",
            },
            "part_to_whole": {
                "segment": segment_names[seg_idx],
                "category": categories[cat_idx],
                "part_value": part_val,
                "whole_value": whole_val,
                "answer": fraction,
                "description": (
                    f"{segment_names[seg_idx]} is {fraction:.1%} of the total "
                    f"for {categories[cat_idx]} ({part_val}/{whole_val})"
                ),
            },
        },
        metadata={"n_categories": n_cats, "n_segments": n_segments, "seed": seed},
    )

    image_path = save_chart_with_metadata(fig, chart_id, "stacked_bar", gt, output_dir)

    questions = [
        {"task_type": "total_comparison", "question": "Which category has the largest total across all segments?"},
        {
            "task_type": "part_to_whole",
            "question": (
                f"What fraction of the total for '{categories[cat_idx]}' "
                f"does '{segment_names[seg_idx]}' represent?"
            ),
        },
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="stacked_bar",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Registry of all generators
# ---------------------------------------------------------------------------

CHART_GENERATORS: dict[str, Callable[..., BenchmarkItem]] = {
    "bar": generate_bar_chart,
    "line": generate_line_chart,
    "scatter": generate_scatter_chart,
    "heatmap": generate_heatmap,
    "area": generate_area_chart,
    "stacked_bar": generate_stacked_bar,
}


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_benchmark_dataset(
    chart_types: list[str] | None = None,
    n_per_type: int = 10,
    output_base_dir: str | Path = "benchmark_output",
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Generate a complete benchmark dataset across the requested chart types.

    Parameters
    ----------
    chart_types:
        Which chart types to generate. ``None`` means all available types.
    n_per_type:
        Number of charts to produce for each chart type.
    output_base_dir:
        Root directory; each chart type gets a sub-directory.
    seed:
        Base random seed. Each individual chart receives ``seed + i`` so that
        results are reproducible but varied.

    Returns
    -------
    list[BenchmarkItem]
        All generated benchmark items (images + metadata).
    """
    if chart_types is None:
        chart_types = list(CHART_GENERATORS.keys())

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    all_items: list[BenchmarkItem] = []

    rng = np.random.default_rng(seed)

    for chart_type in chart_types:
        if chart_type not in CHART_GENERATORS:
            raise ValueError(
                f"Unknown chart type '{chart_type}'. "
                f"Available: {list(CHART_GENERATORS.keys())}"
            )

        generator = CHART_GENERATORS[chart_type]
        type_dir = output_base_dir / chart_type

        for i in range(n_per_type):
            chart_seed = int(rng.integers(0, 2**31))
            chart_id = f"{chart_type}_{i:04d}"
            item = generator(chart_id=chart_id, output_dir=type_dir, seed=chart_seed)
            all_items.append(item)

    # Write manifest
    manifest = {
        "n_items": len(all_items),
        "chart_types": chart_types,
        "n_per_type": n_per_type,
        "seed": seed,
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


# ---------------------------------------------------------------------------
# CLI entry point (convenience)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate chart benchmark dataset")
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_output",
        help="Root output directory",
    )
    parser.add_argument(
        "--n-per-type", type=int, default=10,
        help="Number of charts per type",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--chart-types", nargs="*", default=None,
        help="Chart types to generate (default: all)",
    )
    args = parser.parse_args()

    items = generate_benchmark_dataset(
        chart_types=args.chart_types,
        n_per_type=args.n_per_type,
        output_base_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Generated {len(items)} benchmark items in '{args.output_dir}/'")
