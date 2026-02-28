"""
3D chart stimulus generator for the VLM evaluation pipeline.

Generates 3D-rendered chart images (PNG) with ground-truth metadata (JSON sidecars)
for six chart types: bar_3d, line_3d, scatter_3d, heatmap_3d (surface), area_3d, stacked_bar_3d.

These mirror the 2D charts but rendered with matplotlib 3D projections to simulate
an immersive/perspective viewing condition.

CS 692 Course Project
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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
    _CATEGORY_NAMES,
    _SERIES_NAMES,
    _MONTH_LABELS,
)


# ---------------------------------------------------------------------------
# 3D styling helper
# ---------------------------------------------------------------------------

def _setup_3d_axes(
    fig: plt.Figure,
    elev: float = 25,
    azim: float = -60,
    focal_length: float = 0.2,
) -> Any:
    """Create a 3D axes with immersive perspective settings."""
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("persp", focal_length=focal_length)
    # Reduce visual clutter on panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Generator: 3D Bar chart
# ---------------------------------------------------------------------------

def generate_bar_chart_3d(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a 3D bar chart with blocks along the x-axis."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    # --- Data generation (identical to 2D) ---
    n_cats = rng.integers(5, 9)
    categories = _CATEGORY_NAMES[:n_cats]
    values = np.round(rng.uniform(10, 100, size=n_cats), 1)

    tallest_idx = int(np.argmax(values))
    tallest_cat = categories[tallest_idx]
    tallest_val = float(values[tallest_idx])

    retrieval_idx = int(rng.integers(0, n_cats))
    retrieval_cat = categories[retrieval_idx]
    retrieval_val = float(values[retrieval_idx])

    cmp_indices = rng.choice(n_cats, size=2, replace=False)
    cmp_a, cmp_b = int(cmp_indices[0]), int(cmp_indices[1])
    larger_cat = categories[cmp_a] if values[cmp_a] > values[cmp_b] else categories[cmp_b]

    # --- 3D rendering ---
    colors = sns.color_palette("Set2", n_cats)
    fig = plt.figure(figsize=(9, 6))
    ax = _setup_3d_axes(fig, elev=20, azim=-50)

    x_pos = np.arange(n_cats)
    dx, dy = 0.6, 0.4
    for i in range(n_cats):
        ax.bar3d(
            x_pos[i], 0, 0, dx, dy, float(values[i]),
            color=colors[i], alpha=0.85, edgecolor="black", linewidth=0.3,
        )
        # Value label on top
        ax.text(
            x_pos[i] + dx / 2, dy / 2, float(values[i]) + 1.5,
            f"{values[i]:.1f}", ha="center", va="bottom", fontsize=7,
        )

    ax.set_xticks(x_pos + dx / 2)
    ax.set_xticklabels(categories, fontsize=7, rotation=15)
    ax.set_yticks([])
    ax.set_zlabel("Value", fontsize=9)
    ax.set_title("3D Category Comparison", fontsize=12, pad=15)
    ax.set_zlim(0, max(values) * 1.2)

    gt = ChartGroundTruth(
        chart_type="bar_3d",
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
        metadata={"n_categories": int(n_cats), "seed": seed},
    )

    image_path = save_chart_with_metadata(fig, chart_id, "bar_3d", gt, output_dir)

    questions = [
        {"task_type": "extremum_detection", "question": "In this 3D bar chart, which bar is the tallest, and what is its value?"},
        {"task_type": "value_retrieval", "question": f"In this 3D bar chart, what is the value of the bar labeled '{retrieval_cat}'?"},
        {
            "task_type": "value_comparison",
            "question": f"In this 3D bar chart, which bar is larger: '{categories[cmp_a]}' or '{categories[cmp_b]}'?",
        },
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="bar_3d",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Generator: 3D Line chart
# ---------------------------------------------------------------------------

def generate_line_chart_3d(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a 3D line chart with series spread along the y-axis for depth."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    # --- Data generation (identical to 2D) ---
    n_series = int(rng.integers(1, 4))
    n_points = int(rng.integers(8, 16))
    x_labels = _MONTH_LABELS[:n_points]
    x = np.arange(n_points)

    series_data: dict[str, list[float]] = {}
    palette = sns.color_palette("tab10", n_series)
    markers = ["o", "s", "^", "D"]

    fig = plt.figure(figsize=(10, 6))
    ax = _setup_3d_axes(fig, elev=25, azim=-55, focal_length=0.25)

    for i in range(n_series):
        name = _SERIES_NAMES[i]
        trend_slope = rng.uniform(-2, 2)
        noise = rng.normal(0, 5, size=n_points)
        base = rng.uniform(20, 60)
        y = np.round(base + trend_slope * x + np.cumsum(noise * 0.3), 1)
        series_data[name] = y.tolist()

        y_depth = np.full(n_points, i * 2.0)
        ax.plot(
            x, y_depth, y,
            marker=markers[i % len(markers)],
            markersize=4,
            label=name,
            color=palette[i],
            linewidth=1.5,
        )

    ax.set_xticks(x[::max(1, n_points // 6)])
    ax.set_xticklabels(
        [x_labels[j] for j in range(0, n_points, max(1, n_points // 6))],
        fontsize=7,
    )
    if n_series > 1:
        ax.set_yticks([i * 2.0 for i in range(n_series)])
        ax.set_yticklabels([_SERIES_NAMES[i] for i in range(n_series)], fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_zlabel("Value", fontsize=9)
    ax.set_title("3D Trend Over Time", fontsize=12, pad=15)
    if n_series > 1:
        ax.legend(fontsize=8, loc="upper left")

    # Ground truth from first series
    primary = _SERIES_NAMES[0]
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

    retrieval_idx = int(rng.integers(0, n_points))
    retrieval_time = x_labels[retrieval_idx]
    retrieval_val = float(primary_vals[retrieval_idx])

    gt = ChartGroundTruth(
        chart_type="line_3d",
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

    image_path = save_chart_with_metadata(fig, chart_id, "line_3d", gt, output_dir)

    questions = [
        {"task_type": "trend_identification", "question": f"In this 3D line chart, what is the overall trend of '{primary}'?"},
        {"task_type": "extremum_detection", "question": f"In this 3D line chart, at which time point does '{primary}' reach its peak value, and what is that value?"},
        {"task_type": "value_retrieval", "question": f"In this 3D line chart, what is the value of '{primary}' at '{retrieval_time}'?"},
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="line_3d",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Generator: 3D Scatter chart
# ---------------------------------------------------------------------------

def generate_scatter_chart_3d(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a 3D scatter plot with clusters in 3D space."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    # Same initial draws as 2D so n_clusters and n_total match
    n_clusters = int(rng.integers(2, 5))
    n_total = int(rng.integers(50, 151))
    points_per_cluster = n_total // n_clusters

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    cluster_info: dict[str, Any] = {}
    palette = sns.color_palette("Set2", n_clusters)

    fig = plt.figure(figsize=(8, 7))
    ax = _setup_3d_axes(fig, elev=30, azim=-60)

    for c in range(n_clusters):
        mean = rng.uniform(-10, 10, size=3)
        # Random 3D covariance (positive semi-definite)
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(-0.8, 0.8)
        # Construct a valid 3x3 covariance
        L = np.array([
            [a, 0, 0],
            [b * 0.4, a * 0.8, 0],
            [b * 0.3, b * 0.2, a * 0.7],
        ])
        cov = L @ L.T
        pts = rng.multivariate_normal(mean, cov, size=points_per_cluster)
        all_x.append(pts[:, 0])
        all_y.append(pts[:, 1])
        all_z.append(pts[:, 2])
        label_name = f"Cluster {c + 1}"
        cluster_info[label_name] = {
            "center": mean.tolist(),
            "n_points": points_per_cluster,
        }
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            label=label_name,
            color=palette[c],
            alpha=0.65,
            s=20,
            edgecolors="white",
            linewidths=0.3,
        )

    # Outliers
    n_outliers = int(rng.integers(0, 6))
    outlier_present = n_outliers > 0
    if outlier_present:
        ox = rng.uniform(-15, 15, size=n_outliers)
        oy = rng.uniform(-15, 15, size=n_outliers)
        oz = rng.uniform(-15, 15, size=n_outliers)
        ax.scatter(ox, oy, oz, color="red", marker="x", s=50, label="Outliers", zorder=5)

    # Correlation on x-y projection
    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)
    corr = float(np.corrcoef(combined_x, combined_y)[0, 1])
    if corr > 0.3:
        corr_dir = "positive"
    elif corr < -0.3:
        corr_dir = "negative"
    else:
        corr_dir = "none"

    ax.set_title("3D Scatter Plot with Clusters", fontsize=12, pad=15)
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_zlabel("Z", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    gt = ChartGroundTruth(
        chart_type="scatter_3d",
        data_values={
            "clusters": cluster_info,
            "n_total_points": int(combined_x.shape[0]),
            "n_outliers": n_outliers,
        },
        ground_truth_answers={
            "cluster_count": {
                "answer": n_clusters,
                "description": f"There are {n_clusters} clusters visible in the 3D scatter plot",
            },
            "correlation_direction": {
                "answer": corr_dir,
                "correlation_coefficient": round(corr, 3),
                "description": f"The overall correlation direction (x-y) is {corr_dir} (r={corr:.3f})",
            },
            "outlier_presence": {
                "answer": "yes" if outlier_present else "no",
                "n_outliers": n_outliers,
                "description": f"Outliers {'are' if outlier_present else 'are not'} present ({n_outliers} outlier(s))",
            },
        },
        metadata={"n_clusters": n_clusters, "n_total_points": int(combined_x.shape[0]), "seed": seed},
    )

    image_path = save_chart_with_metadata(fig, chart_id, "scatter_3d", gt, output_dir)

    questions = [
        {"task_type": "cluster_count", "question": "How many distinct clusters are visible in this 3D scatter plot?"},
        {"task_type": "correlation_direction", "question": "What is the overall correlation direction of the data points in this 3D scatter plot?"},
        {"task_type": "outlier_presence", "question": "Are there any outliers in this 3D scatter plot?"},
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="scatter_3d",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Generator: 3D Heatmap (Surface plot)
# ---------------------------------------------------------------------------

def generate_heatmap_3d(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a 3D surface plot from heatmap data."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    # --- Data generation (identical to 2D heatmap) ---
    n_rows = int(rng.integers(5, 9))
    n_cols = int(rng.integers(5, 9))
    row_labels = [f"R{i + 1}" for i in range(n_rows)]
    col_labels = [f"C{j + 1}" for j in range(n_cols)]
    data = np.round(rng.uniform(0, 100, size=(n_rows, n_cols)), 1)

    max_idx = np.unravel_index(int(np.argmax(data)), data.shape)
    max_row, max_col = row_labels[max_idx[0]], col_labels[max_idx[1]]
    max_val = float(data[max_idx])

    r_idx = int(rng.integers(0, n_rows))
    c_idx = int(rng.integers(0, n_cols))
    retrieval_row, retrieval_col = row_labels[r_idx], col_labels[c_idx]
    retrieval_val = float(data[r_idx, c_idx])

    r1, c1 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    r2, c2 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    while r1 == r2 and c1 == c2:
        r2, c2 = int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))
    cell_a = (row_labels[r1], col_labels[c1])
    cell_b = (row_labels[r2], col_labels[c2])
    val_a, val_b = float(data[r1, c1]), float(data[r2, c2])
    higher_cell = cell_a if val_a >= val_b else cell_b

    # --- 3D surface rendering ---
    fig = plt.figure(figsize=(9, 7))
    ax = _setup_3d_axes(fig, elev=30, azim=-45)

    X, Y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    surf = ax.plot_surface(
        X, Y, data,
        cmap="YlOrRd",
        alpha=0.85,
        edgecolor="gray",
        linewidth=0.3,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="Value")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_zlabel("Value", fontsize=9)
    ax.set_xlabel("Column", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)
    ax.set_title("3D Surface Plot", fontsize=12, pad=15)

    gt = ChartGroundTruth(
        chart_type="heatmap_3d",
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

    image_path = save_chart_with_metadata(fig, chart_id, "heatmap_3d", gt, output_dir)

    questions = [
        {"task_type": "max_value_cell", "question": "In this 3D surface plot, which position (row, column) has the highest peak, and what is its value?"},
        {"task_type": "value_retrieval", "question": f"In this 3D surface plot, what is the approximate value at position ({retrieval_row}, {retrieval_col})?"},
        {
            "task_type": "comparison",
            "question": (
                f"In this 3D surface plot, which position has a higher value: ({cell_a[0]}, {cell_a[1]}) "
                f"or ({cell_b[0]}, {cell_b[1]})?"
            ),
        },
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="heatmap_3d",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Generator: 3D Area chart
# ---------------------------------------------------------------------------

def generate_area_chart_3d(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a 3D area chart with filled ribbon polygons."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    # --- Data generation (identical to 2D) ---
    n_series = int(rng.integers(1, 3))
    n_points = int(rng.integers(10, 16))
    x = np.arange(n_points)
    x_labels = _MONTH_LABELS[:n_points]

    series_data: dict[str, list[float]] = {}
    palette = sns.color_palette("pastel", n_series)

    fig = plt.figure(figsize=(10, 6))
    ax = _setup_3d_axes(fig, elev=25, azim=-55, focal_length=0.25)

    for i in range(n_series):
        name = _SERIES_NAMES[i]
        base = rng.uniform(15, 50)
        trend = rng.uniform(-1.0, 1.5)
        noise = rng.normal(0, 3, size=n_points)
        y = np.round(base + trend * x + np.cumsum(noise * 0.4), 1)
        y = np.clip(y, 0, None)
        series_data[name] = y.tolist()

        y_depth = i * 3.0

        # Draw the line
        ax.plot(
            x, np.full(n_points, y_depth), y,
            color=palette[i], linewidth=1.5, label=name,
        )

        # Fill area with polygons
        verts = []
        for j in range(n_points - 1):
            verts.append([
                (x[j], y_depth, 0),
                (x[j], y_depth, float(y[j])),
                (x[j + 1], y_depth, float(y[j + 1])),
                (x[j + 1], y_depth, 0),
            ])
        poly = Poly3DCollection(verts, alpha=0.35, facecolor=palette[i], edgecolor="none")
        ax.add_collection3d(poly)

    ax.set_xticks(x[::max(1, n_points // 6)])
    ax.set_xticklabels(
        [x_labels[j] for j in range(0, n_points, max(1, n_points // 6))],
        fontsize=7,
    )
    if n_series > 1:
        ax.set_yticks([i * 3.0 for i in range(n_series)])
        ax.set_yticklabels([_SERIES_NAMES[i] for i in range(n_series)], fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_zlabel("Value", fontsize=9)
    ax.set_title("3D Area Chart", fontsize=12, pad=15)
    if n_series > 1:
        ax.legend(fontsize=8, loc="upper left")

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
        chart_type="area_3d",
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

    image_path = save_chart_with_metadata(fig, chart_id, "area_3d", gt, output_dir)

    questions = [
        {"task_type": "trend_identification", "question": f"In this 3D area chart, what is the overall trend of '{primary}'?"},
        {"task_type": "max_value", "question": f"In this 3D area chart, what is the maximum value of '{primary}' and when does it occur?"},
        {"task_type": "magnitude_comparison", "question": "In this 3D area chart, which series has the larger overall magnitude?"},
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="area_3d",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Generator: 3D Stacked Bar chart
# ---------------------------------------------------------------------------

def generate_stacked_bar_3d(
    chart_id: str,
    output_dir: str | Path,
    seed: int = 0,
) -> BenchmarkItem:
    """Generate a 3D stacked bar chart with segments stacked along z-axis."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    # --- Data generation (identical to 2D) ---
    n_cats = int(rng.integers(4, 7))
    n_segments = int(rng.integers(2, 4))
    categories = _CATEGORY_NAMES[:n_cats]
    segment_names = [f"Segment {chr(65 + i)}" for i in range(n_segments)]

    data = np.round(rng.uniform(5, 40, size=(n_segments, n_cats)), 1)
    totals = data.sum(axis=0)

    max_total_idx = int(np.argmax(totals))
    max_total_cat = categories[max_total_idx]
    max_total_val = float(totals[max_total_idx])

    seg_idx = int(rng.integers(0, n_segments))
    cat_idx = int(rng.integers(0, n_cats))
    part_val = float(data[seg_idx, cat_idx])
    whole_val = float(totals[cat_idx])
    fraction = round(part_val / whole_val, 3) if whole_val > 0 else 0.0

    # --- 3D rendering ---
    palette = sns.color_palette("Set2", n_segments)
    fig = plt.figure(figsize=(9, 6))
    ax = _setup_3d_axes(fig, elev=20, azim=-50)

    x_pos = np.arange(n_cats)
    dx, dy = 0.6, 0.4
    bottom = np.zeros(n_cats)

    for s in range(n_segments):
        for i in range(n_cats):
            ax.bar3d(
                x_pos[i], 0, bottom[i], dx, dy, float(data[s, i]),
                color=palette[s], alpha=0.85, edgecolor="black", linewidth=0.3,
            )
        bottom += data[s]

    # Legend patches
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=palette[s], label=segment_names[s]) for s in range(n_segments)]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right")

    ax.set_xticks(x_pos + dx / 2)
    ax.set_xticklabels(categories, fontsize=7, rotation=15)
    ax.set_yticks([])
    ax.set_zlabel("Value", fontsize=9)
    ax.set_title("3D Stacked Bar Chart", fontsize=12, pad=15)
    ax.set_zlim(0, max(totals) * 1.15)

    segment_data = {segment_names[s]: data[s].tolist() for s in range(n_segments)}

    gt = ChartGroundTruth(
        chart_type="stacked_bar_3d",
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

    image_path = save_chart_with_metadata(fig, chart_id, "stacked_bar_3d", gt, output_dir)

    questions = [
        {"task_type": "total_comparison", "question": "In this 3D stacked bar chart, which category has the largest total across all segments?"},
        {
            "task_type": "part_to_whole",
            "question": (
                f"In this 3D stacked bar chart, what fraction of the total for '{categories[cat_idx]}' "
                f"does '{segment_names[seg_idx]}' represent?"
            ),
        },
    ]

    return BenchmarkItem(
        chart_id=chart_id,
        chart_type="stacked_bar_3d",
        image_path=str(image_path),
        ground_truth=gt,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CHART_GENERATORS_3D: dict[str, Callable[..., BenchmarkItem]] = {
    "bar_3d": generate_bar_chart_3d,
    "line_3d": generate_line_chart_3d,
    "scatter_3d": generate_scatter_chart_3d,
    "heatmap_3d": generate_heatmap_3d,
    "area_3d": generate_area_chart_3d,
    "stacked_bar_3d": generate_stacked_bar_3d,
}


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_benchmark_dataset_3d(
    chart_types: list[str] | None = None,
    n_per_type: int = 10,
    output_base_dir: str | Path = "benchmark_output_3d",
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Generate a complete 3D benchmark dataset.

    Parameters
    ----------
    chart_types:
        Which 3D chart types to generate. ``None`` means all available types.
    n_per_type:
        Number of charts per type.
    output_base_dir:
        Root directory for output.
    seed:
        Base random seed.
    """
    if chart_types is None:
        chart_types = list(CHART_GENERATORS_3D.keys())

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    all_items: list[BenchmarkItem] = []

    rng = np.random.default_rng(seed)

    for chart_type in chart_types:
        if chart_type not in CHART_GENERATORS_3D:
            raise ValueError(
                f"Unknown 3D chart type '{chart_type}'. "
                f"Available: {list(CHART_GENERATORS_3D.keys())}"
            )

        generator = CHART_GENERATORS_3D[chart_type]
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
        "condition": "3d",
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
