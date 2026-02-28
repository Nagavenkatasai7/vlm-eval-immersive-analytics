"""CLI entry point for the VLM evaluation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def cmd_generate(args):
    """Generate benchmark chart stimuli."""
    from vlm_eval.config import load_config

    config = load_config(Path(args.config) if args.config else None)
    n = args.n or config.n_per_type

    if args.condition == "unity":
        from vlm_eval.stimuli.chart_generator_unity import generate_benchmark_dataset_unity
        chart_types = args.types.split(",") if args.types else [
            "bar_unity", "line_unity", "scatter_unity",
            "heatmap_unity", "area_unity", "stacked_bar_unity",
        ]
        print(f"Generating {n} Unity 3D charts each for: {chart_types}")
        items = generate_benchmark_dataset_unity(
            chart_types=chart_types,
            n_per_type=n,
            output_base_dir=config.charts_unity_dir,
            seed=args.seed,
        )
        print(f"Generated {len(items)} Unity 3D benchmark items")
        print(f"Charts saved to: {config.charts_unity_dir}")
    elif args.condition == "3d":
        from vlm_eval.stimuli.chart_generator_3d import generate_benchmark_dataset_3d
        chart_types = args.types.split(",") if args.types else list(config.__class__.__dataclass_fields__.keys()) and [
            "bar_3d", "line_3d", "scatter_3d", "heatmap_3d", "area_3d", "stacked_bar_3d"
        ]
        # Check if config has chart_types_3d from yaml
        raw_config = {}
        config_path = Path(args.config) if args.config else Path("configs/default.yaml")
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                raw_config = yaml.safe_load(f) or {}
        if args.types:
            chart_types = args.types.split(",")
        elif "chart_types_3d" in raw_config:
            chart_types = raw_config["chart_types_3d"]

        print(f"Generating {n} 3D charts each for: {chart_types}")
        items = generate_benchmark_dataset_3d(
            chart_types=chart_types,
            n_per_type=n,
            output_base_dir=config.charts_3d_dir,
            seed=args.seed,
        )
        print(f"Generated {len(items)} 3D benchmark items")
        print(f"Charts saved to: {config.charts_3d_dir}")
    else:
        from vlm_eval.stimuli.chart_generator import generate_benchmark_dataset
        chart_types = args.types.split(",") if args.types else config.chart_types

        print(f"Generating {n} charts each for: {chart_types}")
        items = generate_benchmark_dataset(
            chart_types=chart_types,
            n_per_type=n,
            output_base_dir=config.charts_dir,
            seed=args.seed,
        )
        print(f"Generated {len(items)} benchmark items")
        print(f"Charts saved to: {config.charts_dir}")


def cmd_evaluate(args):
    """Run VLM evaluation pipeline."""
    from vlm_eval.config import load_config
    from vlm_eval.pipeline import EvalPipeline

    config = load_config(Path(args.config) if args.config else None)

    # Switch chart directory based on condition
    if args.condition == "unity":
        config.charts_dir = config.charts_unity_dir
        config.condition = "unity"
    elif args.condition == "3d":
        config.charts_dir = config.charts_3d_dir
        config.condition = "3d"

    # Filter models if specified
    if args.models:
        model_names = set(args.models.split(","))
        config.models = [m for m in config.models if m.name in model_names]

    if not config.models:
        print("Error: No models configured. Check configs/default.yaml or --models flag.")
        sys.exit(1)

    print(f"Running evaluation ({config.condition} condition) with models: {[m.name for m in config.models]}")
    pipeline = EvalPipeline(config)
    asyncio.run(pipeline.run())


def cmd_report(args):
    """Generate result figures and tables."""
    from vlm_eval.config import load_config
    from vlm_eval.visualization import generate_all_figures

    config = load_config(Path(args.config) if args.config else None)
    generate_all_figures(config)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="vlm-eval",
        description="VLM Evaluation Pipeline for Visualization Literacy",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate benchmark charts")
    gen_parser.add_argument("--types", help="Comma-separated chart types (e.g. bar,line,scatter)")
    gen_parser.add_argument("--n", type=int, help="Number of charts per type")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.add_argument("--config", help="Path to config YAML")
    gen_parser.add_argument("--condition", choices=["2d", "3d", "unity"], default="2d",
                            help="Generate 2D or 3D charts")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run VLM evaluation")
    eval_parser.add_argument("--models", help="Comma-separated model names to evaluate")
    eval_parser.add_argument("--config", help="Path to config YAML")
    eval_parser.add_argument("--condition", choices=["2d", "3d", "unity"], default="2d",
                             help="Evaluate on 2D or 3D charts")

    # Report
    report_parser = subparsers.add_parser("report", help="Generate figures and tables")
    report_parser.add_argument("--config", help="Path to config YAML")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
