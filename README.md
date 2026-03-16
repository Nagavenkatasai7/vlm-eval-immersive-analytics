# An Empirical Evaluation of Multi-LLM and VLM Capabilities for Visualization Literacy in Immersive Analytics

**CS 692 — Mobile Immersive Computing, Spring 2026, George Mason University**

**Authors:** Naga Venkata Sai Chennu, Hemanjali Buchireddy

**Instructor:** Dr. Bo Han

---

## Overview

This project systematically evaluates how well Vision Language Models (VLMs) can interpret data visualizations across different rendering conditions — from standard 2D charts to immersive 3D renderings. We test whether current VLMs can reliably serve as AI assistants in immersive analytics environments where visualizations are inherently three-dimensional.

### Research Questions

1. How does immersion (2D vs. 3D) affect VLM visualization literacy?
2. How much does the rendering method (matplotlib 3D vs. Unity 3D) impact model accuracy?
3. Which chart types and task types are most robust or fragile under 3D conditions?
4. What are the cost-performance tradeoffs across different VLM providers?

### Key Findings

| Model | 2D Accuracy | Matplotlib 3D | Unity 3D |
|-------|-------------|---------------|----------|
| Claude 3.5 Sonnet | 83.9% | — (API limits) | 27.3% |
| Gemini 2.5 Flash | 83.6% | 56.8% | 45.6% |
| GPT-5.2 (ChartX) | 82.0% | 31.1% | — |

- **Progressive accuracy degradation** from 2D to matplotlib-3D to Unity 3D
- **51 percentage point drop** from 2D to 3D on real-world ChartX data
- Bar charts worst affected: 100% (2D) to 25.3% (3D)
- Scatter plots most robust to 3D rendering
- Gemini 2.5 Flash is 24x cheaper than Claude at comparable 3D accuracy

---

## Repository Structure

```
.
├── README.md                       # This file
├── vlm-eval-pipeline/             # Python evaluation pipeline
│   ├── configs/
│   │   └── default.yaml           # Model and pipeline configuration
│   ├── src/vlm_eval/
│   │   ├── __main__.py            # CLI entry point (generate / evaluate / report)
│   │   ├── config.py              # Configuration management
│   │   ├── pipeline.py            # Async evaluation orchestrator
│   │   ├── models/
│   │   │   ├── base.py            # Abstract VisionModel class
│   │   │   └── clients.py         # OpenAI, OpenRouter, Anthropic, Gemini clients
│   │   ├── stimuli/
│   │   │   ├── chart_generator.py         # 2D synthetic chart generation
│   │   │   ├── chart_generator_3d.py      # 3D matplotlib chart generation
│   │   │   ├── chart_generator_unity.py   # Unity 3D batch rendering
│   │   │   ├── chart_generator_chartx.py  # ChartX real-world 2D charts
│   │   │   ├── chart_generator_chartx_3d.py  # ChartX real-world 3D charts
│   │   │   └── chartx_loader.py           # HuggingFace ChartX dataset loader
│   │   ├── evaluation/
│   │   │   ├── scorer.py          # Response parsing and scoring
│   │   │   └── metrics.py         # Accuracy, cost, and latency metrics
│   │   ├── storage/
│   │   │   └── store.py           # Result caching and CSV storage
│   │   └── visualization.py       # Publication-quality figure generation
│   ├── generate_report_pdf.py     # PDF report generator (ReportLab)
│   ├── data/
│   │   ├── charts/                # Generated 2D charts
│   │   ├── charts_3d/             # Generated 3D matplotlib charts
│   │   ├── charts_unity/          # Generated Unity 3D charts
│   │   ├── charts_chartx/         # ChartX real-world 2D charts
│   │   └── charts_chartx_3d/      # ChartX real-world 3D charts
│   ├── results/
│   │   ├── responses/             # Cached VLM response JSONs
│   │   ├── scores/                # Evaluation result CSVs
│   │   └── figures/               # Generated analysis figures
│   └── pyproject.toml             # Dependencies and build config
│
└── vlm-chart-renderer/            # Unity 3D chart rendering project
    ├── Assets/
    │   ├── Editor/
    │   │   └── ChartRenderer.cs   # Batch rendering entry point
    │   └── Scripts/
    │       ├── ChartData.cs       # JSON-serializable chart configs
    │       ├── ChartUtils.cs      # Shared utilities (materials, colors, grids)
    │       └── Charts/
    │           ├── BarChart3D.cs
    │           ├── LineChart3D.cs
    │           ├── ScatterPlot3D.cs
    │           ├── HeatmapSurface3D.cs
    │           ├── AreaChart3D.cs
    │           └── StackedBarChart3D.cs
    ├── Packages/
    └── ProjectSettings/
```

---

## Setup

### Prerequisites

- **Python** 3.12 or 3.13
- **uv** (Python package manager) — [install guide](https://docs.astral.sh/uv/)
- **Unity 6 LTS** (6000.3.10f1) — only needed for Unity 3D chart generation

### Installation

```bash
cd vlm-eval-pipeline

# Install dependencies
uv sync

# Copy and fill in API keys
cp .env.example .env
# Edit .env with your keys:
#   ANTHROPIC_API_KEY=...
#   GOOGLE_API_KEY=...
#   OPENAI_API_KEY=... (or OPENROUTER_API_KEY for GPT-5.2)
```

### API Keys Required

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| Anthropic | `ANTHROPIC_API_KEY` | Claude 3.5 Sonnet |
| Google | `GOOGLE_API_KEY` | Gemini 2.0 Flash, Gemini 2.5 Flash |
| OpenAI | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini |
| OpenRouter | `OPENROUTER_API_KEY` | GPT-5.2 (via `openai/gpt-5.2-chat`) |

---

## Usage

All commands are run from the `vlm-eval-pipeline/` directory:

```bash
cd vlm-eval-pipeline
```

### 1. Generate Chart Stimuli

```bash
# 2D synthetic charts (default: 50 per type, 6 types = 300 charts)
PYTHONPATH=src uv run python -m vlm_eval generate

# 3D matplotlib charts
PYTHONPATH=src uv run python -m vlm_eval generate --condition 3d

# Unity 3D charts (requires Unity installed)
PYTHONPATH=src uv run python -m vlm_eval generate --condition unity

# ChartX real-world data (2D)
PYTHONPATH=src uv run python -m vlm_eval generate --source chartx

# ChartX real-world data (3D)
PYTHONPATH=src uv run python -m vlm_eval generate --source chartx --condition 3d

# Custom: specific types, count, and seed
PYTHONPATH=src uv run python -m vlm_eval generate --types bar,scatter --n 100 --seed 42
```

### 2. Run VLM Evaluation

```bash
# Evaluate on 2D charts (uses models from configs/default.yaml)
PYTHONPATH=src uv run python -m vlm_eval evaluate

# Evaluate on 3D charts
PYTHONPATH=src uv run python -m vlm_eval evaluate --condition 3d

# Evaluate on Unity 3D charts
PYTHONPATH=src uv run python -m vlm_eval evaluate --condition unity

# Evaluate ChartX data
PYTHONPATH=src uv run python -m vlm_eval evaluate --source chartx
PYTHONPATH=src uv run python -m vlm_eval evaluate --source chartx --condition 3d

# Evaluate specific models only
PYTHONPATH=src uv run python -m vlm_eval evaluate --models gpt-5.2,gemini-2.5-flash
```

### 3. Generate Reports

```bash
# Generate analysis figures
PYTHONPATH=src uv run python -m vlm_eval report

# Generate PDF status report
PYTHONPATH=src uv run python generate_report_pdf.py
```

---

## Chart Types

Six chart types are supported across all conditions:

| Chart Type | 2D | Matplotlib 3D | Unity 3D |
|------------|-----|---------------|----------|
| Bar | `bar` | `bar_3d` | `bar_unity` |
| Line | `line` | `line_3d` | `line_unity` |
| Scatter | `scatter` | `scatter_3d` | `scatter_unity` |
| Heatmap | `heatmap` | `heatmap_3d` | `heatmap_unity` |
| Area | `area` | `area_3d` | `area_unity` |
| Stacked Bar | `stacked_bar` | `stacked_bar_3d` | `stacked_bar_unity` |

### Data Sources

- **Synthetic**: Randomly generated data with controlled parameters (categories, series, value ranges)
- **ChartX** (InternScience/ChartX): Real-world chart data from HuggingFace. Bar, line, area, and heatmap use real CSV data; scatter and stacked bar use synthetic data.

---

## Evaluation Task Types

Each chart is evaluated with 2-3 task types depending on the chart:

| Task Type | Description | Charts |
|-----------|-------------|--------|
| `extremum_detection` | Identify max/min values | Bar |
| `value_retrieval` | Read specific values | Bar, Stacked Bar |
| `value_comparison` | Compare two values | Bar, Line |
| `trend_identification` | Identify trend direction | Line, Area |
| `max_value` | Find peak value | Line |
| `cluster_count` | Count clusters | Scatter |
| `outlier_presence` | Detect outliers | Scatter |
| `correlation_direction` | Identify correlation | Scatter, Heatmap |
| `max_value_cell` | Find highest cell | Heatmap |
| `part_to_whole` | Compute ratios | Heatmap, Stacked Bar |
| `magnitude_comparison` | Compare magnitudes | Stacked Bar |
| `total_comparison` | Compare totals | Area |

### Scoring Methods

- **Relaxed accuracy**: Numeric values within 5% relative tolerance
- **Keyword match**: Text answers with synonym expansion (e.g., "increasing" = "upward" = "rising")
- **Exact match**: For count-type tasks

---

## Unity 3D Renderer

The `vlm-chart-renderer/` project renders 3D charts in Unity for the most realistic immersive condition.

### How It Works

1. Python generates JSON chart configs with data + metadata
2. Unity is invoked in batch mode via `ChartRenderer.GenerateAllCharts()`
3. Charts are rendered with professional 3-point lighting, 4x MSAA antialiasing
4. Output: 800x600 PNG images per chart

### Running Manually

```bash
/Applications/Unity/Hub/Editor/6000.3.10f1/Unity.app/Contents/MacOS/Unity \
  -batchmode \
  -projectPath ./vlm-chart-renderer \
  -executeMethod ChartRenderer.GenerateAllCharts \
  -configPath /path/to/configs.json \
  -outputDir /path/to/output \
  -quit
```

### Rendering Features

- Camera positions optimized per chart type
- 3-point lighting: warm key light, cool fill light, rim/back light
- 10-color analytics palette across all chart types
- Grid floor for spatial reference
- Dark background (0.08, 0.08, 0.12) for contrast

---

## Configuration

Edit `vlm-eval-pipeline/configs/default.yaml` to configure models and parameters:

```yaml
models:
- name: gpt-5.2
  provider: openrouter
  model_id: openai/gpt-5.2-chat
  temperature: 1
  max_tokens: 4096

chart_types: [bar, line, scatter, heatmap, area, stacked_bar]
n_per_type: 50        # Charts per type
n_trials: 1           # Evaluation repetitions
concurrency_limit: 5  # Parallel API calls

paths:
  charts_dir: data/charts
  charts_3d_dir: data/charts_3d
  charts_unity_dir: data/charts_unity
  charts_chartx_dir: data/charts_chartx
  charts_chartx_3d_dir: data/charts_chartx_3d
  results_dir: results
  figures_dir: results/figures
```

### Adding a Model

Add an entry under `models:` in `default.yaml`:

```yaml
- name: claude-3.5-sonnet
  provider: anthropic
  model_id: claude-sonnet-4-20250514
  temperature: 0
  max_tokens: 1024
```

Supported providers: `openai`, `anthropic`, `google`, `openrouter`

---

## Results

Results are saved to `vlm-eval-pipeline/results/`:

```
results/
├── responses/           # Individual VLM response JSONs (cached)
│   ├── gpt-5.2/
│   ├── claude-3.5-sonnet/
│   └── gemini-2.5-flash/
├── scores/              # Aggregated result CSVs
│   ├── all_results.csv          # 2D synthetic
│   ├── all_results_3d.csv       # 3D matplotlib
│   ├── all_results_unity.csv    # Unity 3D
│   ├── all_results_chartx_2d.csv
│   └── all_results_chartx_3d.csv
└── figures/             # Generated analysis plots
```

Each response JSON contains: model name, chart ID, question, expected answer, raw response, parsed answer, correctness, latency, token counts, and cost.

---

## Regenerating Data

Chart images and evaluation results are **not** stored in the repository (they total ~180 MB of reproducible data). To regenerate:

```bash
cd vlm-eval-pipeline

# Generate all chart conditions
PYTHONPATH=src uv run python -m vlm_eval generate                              # 2D
PYTHONPATH=src uv run python -m vlm_eval generate --condition 3d               # 3D
PYTHONPATH=src uv run python -m vlm_eval generate --condition unity             # Unity
PYTHONPATH=src uv run python -m vlm_eval generate --source chartx               # ChartX 2D
PYTHONPATH=src uv run python -m vlm_eval generate --source chartx --condition 3d # ChartX 3D

# Run evaluations (requires API keys in .env)
PYTHONPATH=src uv run python -m vlm_eval evaluate
PYTHONPATH=src uv run python -m vlm_eval evaluate --condition 3d
# ... etc.
```

All generation uses seed 42 by default for reproducibility.

---

## Dependencies

**Python** (see `pyproject.toml`):

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API client |
| `openai` | OpenAI / OpenRouter API client |
| `google-genai` | Gemini API client |
| `matplotlib`, `seaborn` | Chart generation and figures |
| `pandas`, `numpy` | Data processing |
| `datasets`, `huggingface-hub` | ChartX dataset loading |
| `pillow` | Image handling |
| `tqdm` | Progress bars |
| `pyyaml`, `python-dotenv` | Configuration |
| `reportlab` | PDF report generation |

**Unity** (see `vlm-chart-renderer/`):

| Requirement | Version |
|-------------|---------|
| Unity Editor | 6000.3.10f1 (Unity 6 LTS) |
| Platform | macOS (batch mode rendering) |

---

## License

This project is part of academic coursework at George Mason University.
