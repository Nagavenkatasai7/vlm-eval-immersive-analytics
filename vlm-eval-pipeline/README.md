# VLM Evaluation Pipeline

Python evaluation pipeline for benchmarking Vision Language Models on visualization literacy tasks across 2D, 3D (matplotlib), and 3D (Unity) rendering conditions.

See the [root README](../README.md) for full project documentation, setup instructions, and usage examples.

## Quick Start

```bash
# Install
uv sync
cp .env.example .env   # Add your API keys

# Generate charts
PYTHONPATH=src uv run python -m vlm_eval generate
PYTHONPATH=src uv run python -m vlm_eval generate --condition 3d
PYTHONPATH=src uv run python -m vlm_eval generate --source chartx

# Evaluate
PYTHONPATH=src uv run python -m vlm_eval evaluate
PYTHONPATH=src uv run python -m vlm_eval evaluate --condition 3d

# Report
PYTHONPATH=src uv run python -m vlm_eval report
```

## Source Layout

```
src/vlm_eval/
├── __main__.py          # CLI: generate / evaluate / report
├── config.py            # YAML config loader
├── pipeline.py          # Async evaluation orchestrator
├── models/
│   ├── base.py          # Abstract VisionModel + retry logic
│   └── clients.py       # OpenAI, OpenRouter, Anthropic, Gemini
├── stimuli/
│   ├── chart_generator.py          # 2D synthetic
│   ├── chart_generator_3d.py       # 3D matplotlib
│   ├── chart_generator_unity.py    # Unity batch rendering
│   ├── chart_generator_chartx.py   # ChartX 2D (real-world)
│   ├── chart_generator_chartx_3d.py # ChartX 3D (real-world)
│   └── chartx_loader.py           # HuggingFace dataset loader
├── evaluation/
│   ├── scorer.py        # Response parsing + scoring
│   └── metrics.py       # Accuracy, cost, latency metrics
├── storage/
│   └── store.py         # Caching + CSV storage
└── visualization.py     # Figure generation
```
