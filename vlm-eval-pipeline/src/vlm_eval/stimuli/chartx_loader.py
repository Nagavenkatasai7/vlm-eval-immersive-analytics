"""Loader for the ChartX dataset (InternScience/ChartX on HuggingFace).

Filters to chart types usable by this pipeline (bar, line, area, heatmap),
parses the embedded CSV strings into structured dicts, and returns records
ready for our chart generators.
"""

from __future__ import annotations

import io
import csv
import logging
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)

# Mapping from ChartX type names to our pipeline type names
CHARTX_TYPE_MAP: dict[str, str] = {
    "bar_chart": "bar",
    "line_chart": "line",
    "area_chart": "area",
    "heatmap": "heatmap",
}

SUPPORTED_CHARTX_TYPES = set(CHARTX_TYPE_MAP.keys())


def _parse_chartx_csv(raw_csv: str) -> list[dict[str, str]]:
    """Parse a ChartX CSV string into a list of row dicts.

    ChartX stores CSV data with literal ``\\t`` and ``\\n`` as delimiters
    (i.e. the two-character sequences, not actual tab/newline).
    """
    # Replace literal escape sequences with real delimiters
    text = raw_csv.replace("\\n", "\n").replace("\\t", "\t").strip()
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows: list[dict[str, str]] = []
    for row in reader:
        # Strip whitespace from keys and values, skip None values
        cleaned = {
            k.strip(): (v.strip() if v is not None else "")
            for k, v in row.items()
            if k is not None
        }
        if cleaned:
            rows.append(cleaned)
    return rows


def _safe_float(val: str) -> float | None:
    """Try to parse a string as a float, stripping common decorations."""
    cleaned = val.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_bar_data(rows: list[dict[str, str]], title: str) -> dict[str, Any] | None:
    """Parse bar chart CSV into {categories, series_names, series_data}."""
    if not rows:
        return None
    headers = list(rows[0].keys())
    if len(headers) < 2:
        return None

    cat_col = headers[0]
    value_cols = headers[1:]
    categories = [r[cat_col] for r in rows]

    series_data: dict[str, list[float]] = {}
    for col in value_cols:
        vals = []
        for r in rows:
            v = _safe_float(r.get(col, ""))
            if v is None:
                return None  # skip records with unparseable values
            vals.append(v)
        series_data[col] = vals

    return {
        "categories": categories,
        "series_names": value_cols,
        "series_data": series_data,
        "title": title,
    }


def _parse_line_data(rows: list[dict[str, str]], title: str) -> dict[str, Any] | None:
    """Parse line chart CSV into {x_labels, series_names, series_data}."""
    if not rows:
        return None
    headers = list(rows[0].keys())
    if len(headers) < 2:
        return None

    x_col = headers[0]
    series_cols = headers[1:]
    x_labels = [r[x_col] for r in rows]

    series_data: dict[str, list[float]] = {}
    for col in series_cols:
        vals = []
        for r in rows:
            v = _safe_float(r.get(col, ""))
            if v is None:
                return None
            vals.append(v)
        series_data[col] = vals

    return {
        "x_labels": x_labels,
        "series_names": series_cols,
        "series_data": series_data,
        "title": title,
    }


def _parse_area_data(rows: list[dict[str, str]], title: str) -> dict[str, Any] | None:
    """Parse area chart CSV — same structure as line."""
    return _parse_line_data(rows, title)


def _parse_heatmap_data(rows: list[dict[str, str]], title: str) -> dict[str, Any] | None:
    """Parse heatmap CSV into {row_labels, col_labels, values}."""
    if not rows:
        return None
    headers = list(rows[0].keys())
    if len(headers) < 2:
        return None

    row_col = headers[0]
    col_labels = headers[1:]
    row_labels = [r[row_col] for r in rows]

    values: list[list[float]] = []
    for r in rows:
        row_vals = []
        for col in col_labels:
            v = _safe_float(r.get(col, ""))
            if v is None:
                return None
            row_vals.append(v)
        values.append(row_vals)

    return {
        "row_labels": row_labels,
        "col_labels": col_labels,
        "values": values,
        "title": title,
    }


_PARSER_MAP = {
    "bar_chart": _parse_bar_data,
    "line_chart": _parse_line_data,
    "area_chart": _parse_area_data,
    "heatmap": _parse_heatmap_data,
}


def load_chartx(
    n_per_type: int = 50,
    split: str = "validation",
) -> dict[str, list[dict[str, Any]]]:
    """Load and parse ChartX records grouped by our chart type names.

    Returns
    -------
    dict mapping our type name (``"bar"``, ``"line"``, etc.) to a list
    of parsed data dicts ready for chart generation.  Each list has at
    most *n_per_type* entries.
    """
    logger.info("Loading ChartX dataset (split=%s)...", split)
    ds = load_dataset("InternScience/ChartX", split=split)

    result: dict[str, list[dict[str, Any]]] = {
        our_type: [] for our_type in CHARTX_TYPE_MAP.values()
    }

    for record in ds:
        cx_type = record["chart_type"]
        if cx_type not in SUPPORTED_CHARTX_TYPES:
            continue

        our_type = CHARTX_TYPE_MAP[cx_type]
        if len(result[our_type]) >= n_per_type:
            continue

        parser = _PARSER_MAP[cx_type]
        rows = _parse_chartx_csv(record["csv"])
        parsed = parser(rows, record.get("title", "").strip())
        if parsed is None:
            continue

        parsed["chartx_topic"] = record.get("topic", "")
        parsed["chartx_imgname"] = record.get("imgname", "")
        result[our_type].append(parsed)

    for our_type, items in result.items():
        logger.info("ChartX loaded %d records for type '%s'", len(items), our_type)

    return result
