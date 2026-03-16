# An Empirical Evaluation of Multi-LLM and VLM Capabilities for Visualization Literacy in Immersive Analytics

**CS 692 — Mobile Immersive Computing, Spring 2026, George Mason University**

**Authors:** Naga Venkata Sai Chennu & Hemanjali Buchireddy

**Advisor:** Dr. Bo Han | **TA:** Fahim Arsad Nafis

---

## Overview

This project systematically evaluates how well Vision-Language Models (VLMs) can interpret data visualizations when those visualizations move from traditional flat 2D charts into 3D environments. We built a fully automated evaluation pipeline that generates charts from real-world data, sends them to commercial VLM APIs along with questions about the displayed data, and scores the model responses against known ground-truth answers.

We evaluate **GPT-5.2** (OpenAI's latest flagship multimodal model, accessed via OpenRouter) using data from the **ChartX** benchmark dataset — a collection of real-world data tables extracted from academic publications across 22 topics.

### Experimental Conditions

| Condition | Description | GPT-5.2 Accuracy | Cost |
|-----------|-------------|:-----------------:|:----:|
| **2D Baseline** | Standard matplotlib charts | **82.0%** | $2.72 |
| **3D matplotlib** | mplot3d 3D projections | **31.1%** | $1.47 |

The evaluation encompasses 1,700 total question-answer pairs (850 per condition), covering 6 chart types with 50 instances each. Total API cost: **$4.19**.

### Key Findings

1. **Dramatic 3D Accuracy Degradation**: GPT-5.2 drops from 82.0% to 31.1% when the same real-world data is rendered in 3D — a **50.9 percentage point loss**
2. **Bar Charts: Perfect 2D, Broken 3D**: Bar charts achieve 100% in 2D but collapse to 25.3% in 3D — the largest drop (-74.7pp) in our evaluation
3. **Scatter Plots Are Most Resilient**: Scatter plots show the smallest 2D-to-3D drop (-34.7pp vs -74.7pp for bars), as their tasks depend on spatial patterns that survive 3D transformation
4. **Task-Type Sensitivity**: Value retrieval tasks are devastated by 3D rendering because they require precise mapping from visual marks to numerical axes
5. **Economic Impact**: Cost per correct answer increases 1.4x on 3D charts ($0.0039 to $0.0056)
6. **Immersive Analytics Gap**: With accuracy at ~31% on 3D charts, current VLMs are not yet reliable for interpreting 3D data visualizations

### Accuracy by Chart Type

| Chart Type | 2D Accuracy | 3D Accuracy | Drop (pp) | Data Source |
|------------|:-----------:|:-----------:|:---------:|:-----------:|
| Bar | 100.0% | 25.3% | -74.7 | ChartX |
| Line | 82.0% | 27.3% | -54.7 | ChartX |
| Scatter | 79.3% | 44.7% | -34.7 | Generated |
| Heatmap | 82.7% | 25.3% | -57.3 | ChartX |
| Area | 60.0% | 31.3% | -28.7 | ChartX |
| Stacked Bar | 91.0% | 33.0% | -58.0 | Generated |
| **Overall** | **82.0%** | **31.1%** | **-50.9** | **Hybrid** |

### Why VLMs Struggle with 3D

1. **Perspective Foreshortening** — Objects further from the camera appear smaller; a tall bar in the back can appear the same pixel height as a shorter bar in front
2. **Occlusion** — Front elements partially or fully hide elements behind them; the VLM cannot rotate the view
3. **Axis Readability** — 3D charts render axes at oblique angles, making tick marks and labels harder to read
4. **Depth Ambiguity** — Two points at different depths may appear at the same (x, y) position in the image

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
│   │   ├── charts/                # Generated 2D synthetic charts
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

## Methodology

### The ChartX Dataset

We use real-world data from the [ChartX](https://huggingface.co/datasets/InternScience/ChartX) dataset (Apache 2.0 license). ChartX is a comprehensive chart benchmark containing 6,000+ chart images across 18 chart types and 22 academic topics (economics, health, demographics, environment, etc.). We adopt a **hybrid approach**: ChartX provides real-world data tables for 4 chart types (bar, line, area, heatmap), while scatter and stacked bar charts use procedurally generated data with controlled statistical properties.

### Chart Generation Pipeline

All 600 charts (300 per condition) are generated programmatically. For each ChartX record, we parse the embedded CSV data, extract categories, series names, and values, then render the chart through matplotlib. The same data is rendered once in 2D and once in 3D, using deterministic random seeds for full reproducibility. Ground-truth answers are computed at generation time and stored in JSON sidecar files.

### The Six Chart Types

| Chart Type | Data Source | Visual Encoding | What It Tests |
|------------|:-----------:|-----------------|---------------|
| Bar | ChartX | Rectangular bars | Value retrieval, comparison, extremum detection |
| Line | ChartX | Connected data points over time | Trend identification, value reading |
| Scatter | Generated | Points by x/y coordinates | Cluster counting, correlation, outlier detection |
| Heatmap | ChartX | Cells colored by value intensity | Max-cell identification, cross-row comparison |
| Area | ChartX | Lines with filled regions below | Trend identification, value comparison |
| Stacked Bar | Generated | Multiple series stacked within bars | Part-to-whole reasoning, total comparison |

### Scoring and Evaluation Logic

When a VLM receives a chart image and a question, it returns a free-text response. Our automated scoring pipeline processes responses in two stages:

- **Parsing**: For numeric tasks (value retrieval, max value, cluster count, part-to-whole), we extract the last number found in the response using regular expressions. For text tasks (comparison, trend ID, correlation direction), we use the full response text and search for expected keywords.
- **Scoring**: Numeric answers are scored using a 10% tolerance — if the parsed number is within 10% of the ground truth, it is marked correct. Text answers are scored by checking whether the expected keyword appears in the response.

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
| OpenRouter | `OPENROUTER_API_KEY` | GPT-5.2 (via `openai/gpt-5.2-chat`) |
| Anthropic | `ANTHROPIC_API_KEY` | Claude 3.5 Sonnet |
| Google | `GOOGLE_API_KEY` | Gemini 2.0 Flash, Gemini 2.5 Flash |
| OpenAI | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini |

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
PYTHONPATH=src uv run python -m vlm_eval evaluate --models gpt-5.2
```

### 3. Generate Reports

```bash
# Generate analysis figures
PYTHONPATH=src uv run python -m vlm_eval report

# Generate PDF status report
PYTHONPATH=src uv run python generate_report_pdf.py
```

---

## Unity 3D Renderer

The `vlm-chart-renderer/` project renders 3D charts in Unity for a more realistic immersive condition (planned third condition).

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
- Dark background for contrast

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
- name: gemini-2.5-flash
  provider: google
  model_id: gemini-2.5-flash
  temperature: 0
  max_tokens: 1024
```

Supported providers: `openai`, `anthropic`, `google`, `openrouter`

---

## Results Output

Results are saved to `vlm-eval-pipeline/results/`:

```
results/
├── responses/           # Individual VLM response JSONs (cached per model+condition)
│   └── gpt-5.2/
├── scores/              # Aggregated result CSVs
│   ├── all_results_chartx_2d.csv
│   └── all_results_chartx_3d.csv
└── figures/             # Generated analysis plots
```

Each response JSON contains: model name, chart ID, question, expected answer, raw response, parsed answer, correctness, scoring method, latency, token counts, and cost.

---

## Regenerating Data

Chart images and evaluation results are **not** stored in the repository (they total ~180 MB of reproducible data). To regenerate:

```bash
cd vlm-eval-pipeline

# Generate all chart conditions
PYTHONPATH=src uv run python -m vlm_eval generate                              # 2D synthetic
PYTHONPATH=src uv run python -m vlm_eval generate --condition 3d               # 3D matplotlib
PYTHONPATH=src uv run python -m vlm_eval generate --condition unity             # Unity 3D
PYTHONPATH=src uv run python -m vlm_eval generate --source chartx               # ChartX 2D
PYTHONPATH=src uv run python -m vlm_eval generate --source chartx --condition 3d # ChartX 3D

# Run evaluations (requires API keys in .env)
PYTHONPATH=src uv run python -m vlm_eval evaluate --source chartx
PYTHONPATH=src uv run python -m vlm_eval evaluate --source chartx --condition 3d
```

All generation uses seed 42 by default for reproducibility.

---

## Next Steps

- Add Claude 3.5 Sonnet and Gemini 2.5 Flash to the ChartX evaluation for a full multi-model comparison
- Render ChartX data through Unity 3D for a third, more immersive rendering condition
- Manual error analysis: review 50+ failure cases to categorize failure modes
- Statistical significance testing using McNemar's test for paired comparisons
- Prompt engineering experiments: can chain-of-thought improve 3D accuracy?
- Test additional conditions: different camera angles, lighting setups, resolutions
- Write final report (6+ pages, ACM double-column format)
- Final presentation: April 17, 2026

---

## Dependencies

**Python** (see `pyproject.toml`):

| Package | Purpose |
|---------|---------|
| `openai` | OpenAI / OpenRouter API client |
| `anthropic` | Anthropic API client |
| `google-genai` | Gemini API client |
| `matplotlib`, `seaborn` | Chart generation and analysis figures |
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
