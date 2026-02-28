"""
Unity 3D chart stimulus generator for the VLM evaluation pipeline.

Generates immersive 3D chart images via Unity batch rendering. Uses the same
data generation seeds as the matplotlib 2D/3D generators so ground truth is
identical, but the visual rendering is done in Unity with real 3D lighting,
materials, and perspective.

CS 692 Course Project
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vlm_eval.stimuli.chart_generator import (
    BenchmarkItem,
    ChartGroundTruth,
    _CATEGORY_NAMES,
    _SERIES_NAMES,
    _MONTH_LABELS,
)

# ---------------------------------------------------------------------------
# Unity paths
# ---------------------------------------------------------------------------

UNITY_EDITOR = "/Applications/Unity/Hub/Editor/6000.3.10f1/Unity.app/Contents/MacOS/Unity"
UNITY_PROJECT = str(
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "vlm-chart-renderer"
)


# ---------------------------------------------------------------------------
# Config generators: produce data + ground truth (same as matplotlib versions)
# ---------------------------------------------------------------------------

def _gen_bar_config(chart_id: str, seed: int) -> tuple[dict, ChartGroundTruth, list[dict]]:
    """Generate bar chart data config and ground truth."""
    rng = np.random.default_rng(seed)
    n_cats = int(rng.integers(5, 9))
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

    config = {
        "chart_id": chart_id,
        "chart_type": "bar_unity",
        "title": "3D Bar Chart",
        "labels": categories,
        "values": values.tolist(),
    }

    gt = ChartGroundTruth(
        chart_type="bar_unity",
        data_values={"categories": categories, "values": values.tolist()},
        ground_truth_answers={
            "extremum_detection": {"answer": tallest_cat, "value": tallest_val},
            "value_retrieval": {"category": retrieval_cat, "answer": retrieval_val},
            "comparison": {
                "answer": larger_cat,
                "options": [categories[cmp_a], categories[cmp_b]],
            },
        },
        metadata={"n_categories": n_cats, "seed": seed},
    )

    questions = [
        {"task_type": "extremum_detection", "question": "In this 3D bar chart, which category has the tallest bar?"},
        {"task_type": "value_retrieval", "question": f"In this 3D bar chart, what is the approximate value for '{retrieval_cat}'?"},
        {"task_type": "comparison", "question": f"In this 3D bar chart, which has a larger value: '{categories[cmp_a]}' or '{categories[cmp_b]}'?"},
    ]

    return config, gt, questions


def _gen_line_config(chart_id: str, seed: int) -> tuple[dict, ChartGroundTruth, list[dict]]:
    """Generate line chart data config and ground truth."""
    rng = np.random.default_rng(seed)
    n_series = int(rng.integers(2, 4))
    n_points = int(rng.integers(8, 13))
    x_labels = _MONTH_LABELS[:n_points]

    series_data = {}
    all_flat = []
    for i in range(n_series):
        name = _SERIES_NAMES[i]
        base = rng.uniform(20, 60)
        trend = rng.uniform(-1.5, 2.0)
        noise = rng.normal(0, 4, size=n_points)
        y = np.round(base + trend * np.arange(n_points) + np.cumsum(noise * 0.5), 1)
        y = np.clip(y, 0, None)
        series_data[name] = y.tolist()
        all_flat.extend(y.tolist())

    primary = _SERIES_NAMES[0]
    primary_vals = np.array(series_data[primary])
    slope = np.polyfit(np.arange(n_points), primary_vals, 1)[0]
    trend_dir = "increasing" if slope > 0.5 else ("decreasing" if slope < -0.5 else "stable")

    max_idx = int(np.argmax(primary_vals))
    max_val = float(primary_vals[max_idx])
    max_time = x_labels[max_idx]

    config = {
        "chart_id": chart_id,
        "chart_type": "line_unity",
        "title": "3D Line Chart",
        "labels": x_labels,
        "n_categories": n_points,
        "n_series": n_series,
        "stacked_values_flat": all_flat,
    }

    gt = ChartGroundTruth(
        chart_type="line_unity",
        data_values={"x_labels": x_labels, "series": series_data},
        ground_truth_answers={
            "trend_identification": {"series": primary, "answer": trend_dir, "slope": round(float(slope), 3)},
            "max_value": {"series": primary, "answer": max_val, "time_point": max_time},
            "value_comparison": {
                "answer": _SERIES_NAMES[0] if np.mean(list(series_data.values())[0]) >= np.mean(list(series_data.values())[1]) else _SERIES_NAMES[1],
            },
        },
        metadata={"n_series": n_series, "n_points": n_points, "seed": seed},
    )

    questions = [
        {"task_type": "trend_identification", "question": f"In this 3D line chart, what is the overall trend of '{primary}'?"},
        {"task_type": "max_value", "question": f"In this 3D line chart, what is the maximum value of '{primary}'?"},
        {"task_type": "value_comparison", "question": "In this 3D line chart, which series has the higher average value?"},
    ]

    return config, gt, questions


def _gen_scatter_config(chart_id: str, seed: int) -> tuple[dict, ChartGroundTruth, list[dict]]:
    """Generate scatter plot data config and ground truth."""
    rng = np.random.default_rng(seed)
    n_clusters = int(rng.integers(2, 5))
    points_per = int(rng.integers(15, 30))
    n_total = n_clusters * points_per

    centers = rng.uniform(-5, 5, size=(n_clusters, 3))
    scatter_x, scatter_y, scatter_z, labels = [], [], [], []

    for c in range(n_clusters):
        cx, cy, cz = centers[c]
        spread = rng.uniform(0.5, 1.5)
        pts = rng.normal(0, spread, size=(points_per, 3))
        for p in pts:
            scatter_x.append(float(cx + p[0]))
            scatter_y.append(float(cy + p[1]))
            scatter_z.append(float(cz + p[2]))
            labels.append(c)

    all_x = np.array(scatter_x)
    all_y = np.array(scatter_y)
    all_z = np.array(scatter_z)

    # Correlation (x vs y)
    corr = float(np.corrcoef(all_x, all_y)[0, 1])
    corr_dir = "positive" if corr > 0.3 else ("negative" if corr < -0.3 else "none")

    # Outlier detection
    dists = np.sqrt(all_x**2 + all_y**2 + all_z**2)
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    has_outlier = bool(np.any(dists > mean_dist + 3 * std_dist))

    config = {
        "chart_id": chart_id,
        "chart_type": "scatter_unity",
        "title": "3D Scatter Plot",
        "labels": [],
        "values": [],
        "scatter_x": scatter_x,
        "scatter_y": scatter_y,
        "scatter_z": scatter_z,
        "scatter_labels": labels,
        "n_clusters": n_clusters,
    }

    gt = ChartGroundTruth(
        chart_type="scatter_unity",
        data_values={"x": scatter_x, "y": scatter_y, "z": scatter_z, "labels": labels},
        ground_truth_answers={
            "cluster_count": {"answer": n_clusters},
            "correlation_direction": {"answer": corr_dir, "correlation": round(corr, 3)},
            "outlier_presence": {"answer": "yes" if has_outlier else "no"},
        },
        metadata={"n_clusters": n_clusters, "n_points": n_total, "seed": seed},
    )

    questions = [
        {"task_type": "cluster_count", "question": "In this 3D scatter plot, how many distinct clusters can you identify?"},
        {"task_type": "correlation_direction", "question": "In this 3D scatter plot, what is the overall correlation direction between x and y?"},
        {"task_type": "outlier_presence", "question": "In this 3D scatter plot, are there any clear outlier points?"},
    ]

    return config, gt, questions


def _gen_heatmap_config(chart_id: str, seed: int) -> tuple[dict, ChartGroundTruth, list[dict]]:
    """Generate heatmap surface data config and ground truth."""
    rng = np.random.default_rng(seed)
    n_rows = int(rng.integers(5, 9))
    n_cols = int(rng.integers(5, 9))
    row_labels = [f"R{i}" for i in range(n_rows)]
    col_labels = [f"C{i}" for i in range(n_cols)]

    data = np.round(rng.uniform(0, 100, size=(n_rows, n_cols)), 1)

    max_idx = np.unravel_index(np.argmax(data), data.shape)
    max_val = float(data[max_idx])

    retrieval_r = int(rng.integers(0, n_rows))
    retrieval_c = int(rng.integers(0, n_cols))
    retrieval_val = float(data[retrieval_r, retrieval_c])

    cell_a = (int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols)))
    cell_b = (int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols)))
    while cell_a == cell_b:
        cell_b = (int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols)))
    larger_cell = cell_a if data[cell_a] >= data[cell_b] else cell_b

    config = {
        "chart_id": chart_id,
        "chart_type": "heatmap_unity",
        "title": "3D Heatmap Surface",
        "labels": [],
        "values": [],
        "heatmap_flat": data.flatten().tolist(),
        "heatmap_rows": n_rows,
        "heatmap_cols": n_cols,
        "row_labels": row_labels,
        "col_labels": col_labels,
    }

    gt = ChartGroundTruth(
        chart_type="heatmap_unity",
        data_values={"data": data.tolist(), "row_labels": row_labels, "col_labels": col_labels},
        ground_truth_answers={
            "max_value_cell": {"answer": f"({row_labels[max_idx[0]]}, {col_labels[max_idx[1]]})", "value": max_val},
            "value_retrieval": {"answer": retrieval_val, "position": f"({row_labels[retrieval_r]}, {col_labels[retrieval_c]})"},
            "comparison": {
                "answer": f"({row_labels[larger_cell[0]]}, {col_labels[larger_cell[1]]})",
                "options": [
                    f"({row_labels[cell_a[0]]}, {col_labels[cell_a[1]]})",
                    f"({row_labels[cell_b[0]]}, {col_labels[cell_b[1]]})",
                ],
            },
        },
        metadata={"n_rows": n_rows, "n_cols": n_cols, "seed": seed},
    )

    questions = [
        {"task_type": "max_value_cell", "question": "In this 3D heatmap surface, which cell has the highest peak?"},
        {"task_type": "value_retrieval", "question": f"In this 3D heatmap surface, what is the approximate value at ({row_labels[retrieval_r]}, {col_labels[retrieval_c]})?"},
        {"task_type": "comparison", "question": f"In this 3D heatmap surface, which cell has a higher value: ({row_labels[cell_a[0]]}, {col_labels[cell_a[1]]}) or ({row_labels[cell_b[0]]}, {col_labels[cell_b[1]]})?"},
    ]

    return config, gt, questions


def _gen_area_config(chart_id: str, seed: int) -> tuple[dict, ChartGroundTruth, list[dict]]:
    """Generate area chart data config and ground truth."""
    rng = np.random.default_rng(seed)
    n_series = int(rng.integers(1, 3))
    n_points = int(rng.integers(10, 16))
    x_labels = _MONTH_LABELS[:n_points]

    series_data = {}
    all_flat = []
    for i in range(n_series):
        name = _SERIES_NAMES[i]
        base = rng.uniform(15, 50)
        trend = rng.uniform(-1.0, 1.5)
        noise = rng.normal(0, 3, size=n_points)
        y = np.round(base + trend * np.arange(n_points) + np.cumsum(noise * 0.4), 1)
        y = np.clip(y, 0, None)
        series_data[name] = y.tolist()
        all_flat.extend(y.tolist())

    primary = _SERIES_NAMES[0]
    primary_vals = np.array(series_data[primary])
    slope = np.polyfit(np.arange(n_points), primary_vals, 1)[0]
    trend_dir = "increasing" if slope > 0.5 else ("decreasing" if slope < -0.5 else "stable")

    max_idx = int(np.argmax(primary_vals))
    max_val = float(primary_vals[max_idx])
    max_time = x_labels[max_idx]

    if n_series == 2:
        secondary = _SERIES_NAMES[1]
        sec_vals = np.array(series_data[secondary])
        larger_series = primary if np.mean(primary_vals) >= np.mean(sec_vals) else secondary
    else:
        larger_series = primary

    config = {
        "chart_id": chart_id,
        "chart_type": "area_unity",
        "title": "3D Area Chart",
        "labels": x_labels,
        "n_categories": n_points,
        "n_series": n_series,
        "stacked_values_flat": all_flat,
    }

    gt = ChartGroundTruth(
        chart_type="area_unity",
        data_values={"x_labels": x_labels, "series": series_data},
        ground_truth_answers={
            "trend_identification": {"series": primary, "answer": trend_dir, "slope": round(float(slope), 3)},
            "max_value": {"series": primary, "answer": max_val, "time_point": max_time},
            "magnitude_comparison": {"answer": larger_series},
        },
        metadata={"n_series": n_series, "n_points": n_points, "seed": seed},
    )

    questions = [
        {"task_type": "trend_identification", "question": f"In this 3D area chart, what is the overall trend of '{primary}'?"},
        {"task_type": "max_value", "question": f"In this 3D area chart, what is the maximum value of '{primary}'?"},
        {"task_type": "magnitude_comparison", "question": "In this 3D area chart, which series has the larger overall magnitude?"},
    ]

    return config, gt, questions


def _gen_stacked_bar_config(chart_id: str, seed: int) -> tuple[dict, ChartGroundTruth, list[dict]]:
    """Generate stacked bar chart data config and ground truth."""
    rng = np.random.default_rng(seed)
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

    config = {
        "chart_id": chart_id,
        "chart_type": "stacked_bar_unity",
        "title": "3D Stacked Bar Chart",
        "labels": categories,
        "values": [],
        "stacked_values_flat": data.flatten().tolist(),
        "stacked_n_cats": n_cats,
        "stacked_n_series": n_segments,
        "stacked_series_names": segment_names,
    }

    gt = ChartGroundTruth(
        chart_type="stacked_bar_unity",
        data_values={
            "categories": categories,
            "segments": {segment_names[s]: data[s].tolist() for s in range(n_segments)},
            "totals": totals.tolist(),
        },
        ground_truth_answers={
            "total_comparison": {"answer": max_total_cat, "total_value": max_total_val},
            "part_to_whole": {
                "segment": segment_names[seg_idx],
                "category": categories[cat_idx],
                "part_value": part_val,
                "whole_value": whole_val,
                "answer": fraction,
            },
        },
        metadata={"n_categories": n_cats, "n_segments": n_segments, "seed": seed},
    )

    questions = [
        {"task_type": "total_comparison", "question": "In this 3D stacked bar chart, which category has the largest total?"},
        {"task_type": "part_to_whole", "question": f"In this 3D stacked bar chart, what fraction of '{categories[cat_idx]}' does '{segment_names[seg_idx]}' represent?"},
    ]

    return config, gt, questions


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONFIG_GENERATORS = {
    "bar_unity": _gen_bar_config,
    "line_unity": _gen_line_config,
    "scatter_unity": _gen_scatter_config,
    "heatmap_unity": _gen_heatmap_config,
    "area_unity": _gen_area_config,
    "stacked_bar_unity": _gen_stacked_bar_config,
}


# ---------------------------------------------------------------------------
# Save sidecar JSON (matching matplotlib format)
# ---------------------------------------------------------------------------

def _save_sidecar(chart_id: str, chart_type: str, image_path: Path, gt: ChartGroundTruth) -> None:
    """Write a JSON sidecar alongside the PNG (same format as matplotlib generator)."""
    json_path = image_path.with_suffix(".json")
    sidecar = {
        "chart_id": chart_id,
        "chart_type": chart_type,
        "image_file": image_path.name,
        "ground_truth": asdict(gt),
    }
    json_path.write_text(json.dumps(sidecar, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main: generate configs, call Unity, collect results
# ---------------------------------------------------------------------------

def generate_benchmark_dataset_unity(
    chart_types: list[str] | None = None,
    n_per_type: int = 10,
    output_base_dir: str | Path = "benchmark_output_unity",
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Generate a complete Unity 3D benchmark dataset.

    1. Generate chart data configs + ground truth (Python side)
    2. Write configs to JSON
    3. Call Unity in batch mode to render PNGs
    4. Write JSON sidecars and manifest
    """
    if chart_types is None:
        chart_types = list(CONFIG_GENERATORS.keys())

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    all_configs = []
    all_gt = {}  # chart_id -> (gt, questions, chart_type)

    print(f"Generating configs for {len(chart_types)} chart types x {n_per_type} each...")

    for chart_type in chart_types:
        if chart_type not in CONFIG_GENERATORS:
            raise ValueError(f"Unknown chart type '{chart_type}'. Available: {list(CONFIG_GENERATORS.keys())}")

        generator = CONFIG_GENERATORS[chart_type]
        type_dir = output_base_dir / chart_type
        type_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_per_type):
            chart_seed = int(rng.integers(0, 2**31))
            chart_id = f"{chart_type}_{i:04d}"
            config, gt, questions = generator(chart_id, chart_seed)
            all_configs.append(config)
            all_gt[chart_id] = (gt, questions, chart_type)

    # Write Unity config JSON
    unity_config = {
        "charts": all_configs,
        "imageWidth": 800,
        "imageHeight": 600,
    }
    config_path = output_base_dir / "_unity_configs.json"
    config_path.write_text(json.dumps(unity_config, indent=2, default=str))
    print(f"Wrote {len(all_configs)} chart configs to {config_path}")

    # Call Unity batch renderer
    print(f"Launching Unity batch renderer...")
    print(f"  Unity: {UNITY_EDITOR}")
    print(f"  Project: {UNITY_PROJECT}")

    log_path = output_base_dir / "unity_render.log"
    cmd = [
        UNITY_EDITOR,
        "-batchmode",
        "-projectPath", UNITY_PROJECT,
        "-executeMethod", "ChartRenderer.GenerateAllCharts",
        "-configPath", str(config_path.resolve()),
        "-outputDir", str(output_base_dir.resolve()),
        "-logFile", str(log_path.resolve()),
        "-quit",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(f"Unity exited with code {result.returncode}")

    if result.returncode != 0:
        print(f"  Check log: {log_path}")
        # Don't fail — some charts may have rendered

    # Collect results and write sidecars
    all_items: list[BenchmarkItem] = []

    for chart_id, (gt, questions, chart_type) in all_gt.items():
        type_dir = output_base_dir / chart_type
        image_path = type_dir / f"{chart_id}.png"

        if image_path.exists():
            # Write JSON sidecar
            _save_sidecar(chart_id, chart_type, image_path, gt)

            item = BenchmarkItem(
                chart_id=chart_id,
                chart_type=chart_type,
                image_path=str(image_path),
                ground_truth=gt,
                questions=questions,
            )
            all_items.append(item)
        else:
            print(f"  WARNING: Missing render for {chart_id}")

    # Write manifest
    manifest = {
        "n_items": len(all_items),
        "chart_types": chart_types,
        "n_per_type": n_per_type,
        "seed": seed,
        "condition": "unity",
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

    print(f"\nGenerated {len(all_items)}/{len(all_gt)} charts successfully")
    print(f"Output: {output_base_dir}")

    return all_items
