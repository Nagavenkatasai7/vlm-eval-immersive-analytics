"""Main evaluation pipeline orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from tqdm import tqdm

from vlm_eval.config import PipelineConfig
from vlm_eval.evaluation.scorer import parse_response, score_item
from vlm_eval.models import VisionResponse, get_model
from vlm_eval.storage.store import ResultStore

logger = logging.getLogger(__name__)


class EvalPipeline:
    """Orchestrates the full evaluation across models and benchmark items."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.store = ResultStore(config.results_dir)

    async def _evaluate_item(
        self,
        model_name: str,
        provider: str,
        model_id: str,
        chart_id: str,
        image_path: str,
        task_type: str,
        question: str,
        expected_answer: str,
        chart_type: str,
        trial: int = 0,
        semaphore: asyncio.Semaphore | None = None,
    ) -> dict:
        """Evaluate a single item: query VLM, parse, score, store."""
        cache_key = f"{chart_id}_{task_type}"
        condition = getattr(self.config, "condition", "2d")

        # Check cache
        cached = self.store.check_cached(model_name, cache_key, trial, condition)
        if cached:
            return cached

        api_key = self.config.get_api_key(provider)
        model = get_model(
            provider=provider,
            model_id=model_id,
            api_key=api_key,
            temperature=0,
            max_tokens=self.config.models[0].max_tokens if self.config.models else 1024,
        )

        async def _query():
            if semaphore:
                async with semaphore:
                    return await model.query(Path(image_path), question)
            return await model.query(Path(image_path), question)

        try:
            response: VisionResponse = await _query()
        except Exception as e:
            logger.error(f"Error querying {model_name} for {chart_id}: {e}")
            condition = getattr(self.config, "condition", "2d")
            result = {
                "model_name": model_name,
                "chart_id": chart_id,
                "chart_type": chart_type,
                "task_type": task_type,
                "question": question,
                "expected_answer": str(expected_answer),
                "raw_response": f"ERROR: {e}",
                "parsed_answer": None,
                "correct": False,
                "score_method": "error",
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
                "trial": trial,
                "condition": condition,
            }
            self.store.save_response(model_name, cache_key, result, trial, condition)
            return result

        # Determine expected type for parsing
        numeric_tasks = {
            "value_retrieval", "max_value", "max_value_cell",
            "cluster_count", "part_to_whole",
        }
        # Text tasks: the answer is a category name, direction, yes/no, etc.
        # For these, we pass the full response to the scorer as text.
        expected_type = "numeric" if task_type in numeric_tasks else "text"
        parsed = parse_response(response.raw_response, expected_type)

        # Score
        score = score_item(parsed, str(expected_answer), task_type)

        condition = getattr(self.config, "condition", "2d")
        result = {
            "model_name": model_name,
            "chart_id": chart_id,
            "chart_type": chart_type,
            "task_type": task_type,
            "question": question,
            "expected_answer": str(expected_answer),
            "raw_response": response.raw_response,
            "parsed_answer": parsed,
            "correct": score["correct"],
            "score_method": score["method"],
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "trial": trial,
            "condition": condition,
        }

        self.store.save_response(model_name, cache_key, result, trial, condition)
        return result

    async def _run_model(
        self,
        model_cfg,
        benchmark_items: list[dict],
    ) -> list[dict]:
        """Run a single model on all benchmark items."""
        semaphore = asyncio.Semaphore(self.config.concurrency_limit)
        results = []

        tasks = []
        for item in benchmark_items:
            for trial in range(self.config.n_trials):
                tasks.append(
                    self._evaluate_item(
                        model_name=model_cfg.name,
                        provider=model_cfg.provider,
                        model_id=model_cfg.model_id,
                        chart_id=item["chart_id"],
                        image_path=item["image_path"],
                        task_type=item["task_type"],
                        question=item["question"],
                        expected_answer=item["expected_answer"],
                        chart_type=item["chart_type"],
                        trial=trial,
                        semaphore=semaphore,
                    )
                )

        # Run with progress bar
        pbar = tqdm(total=len(tasks), desc=f"  {model_cfg.name}", leave=True)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
        pbar.close()

        return results

    def _load_benchmark_items(self) -> list[dict]:
        """Load benchmark items from manifest + sidecar JSONs."""
        manifest_path = self.config.charts_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest found at {manifest_path}. Run 'generate' first."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        entries = manifest.get("items", manifest) if isinstance(manifest, dict) else manifest

        # Flatten: one entry per (chart, task_type) pair
        items = []
        for entry in entries:
            image_path = entry["image_path"]
            chart_type = entry["chart_type"]

            # Load ground truth from sidecar JSON
            sidecar_path = Path(image_path).with_suffix(".json")
            gt_answers = {}
            if sidecar_path.exists():
                with open(sidecar_path) as f:
                    sidecar = json.load(f)
                gt = sidecar.get("ground_truth", {})
                gt_answers = gt.get("ground_truth_answers", {})

            questions = entry.get("questions", [])
            q_map = {q["task_type"]: q["question"] for q in questions}

            for task_type, answer_data in gt_answers.items():
                # Extract the actual answer value from structured ground truth
                if isinstance(answer_data, dict):
                    expected = answer_data.get("answer", str(answer_data))
                else:
                    expected = answer_data

                question = q_map.get(task_type, f"Answer this about the chart: {task_type}")
                items.append({
                    "chart_id": entry["chart_id"],
                    "chart_type": chart_type,
                    "image_path": image_path,
                    "task_type": task_type,
                    "question": question,
                    "expected_answer": str(expected),
                })

        return items

    async def run(self) -> None:
        """Run the full evaluation pipeline."""
        import pandas as pd

        from vlm_eval.evaluation.metrics import (
            compute_accuracy_by_group,
            compute_cost_metrics,
            generate_summary_table,
        )

        logger.info("Loading benchmark items...")
        benchmark_items = self._load_benchmark_items()
        logger.info(f"Loaded {len(benchmark_items)} evaluation items")

        all_results = []
        for model_cfg in self.config.models:
            logger.info(f"Evaluating {model_cfg.name}...")
            results = await self._run_model(model_cfg, benchmark_items)
            all_results.extend(results)

        # Save all results (separate CSV per condition)
        df = pd.DataFrame(all_results)
        condition = getattr(self.config, "condition", "2d")
        condition_filenames = {
            "3d": "all_results_3d.csv",
            "unity": "all_results_unity.csv",
            "chartx_2d": "all_results_chartx_2d.csv",
            "chartx_3d": "all_results_chartx_3d.csv",
        }
        filename = condition_filenames.get(condition, "all_results.csv")
        self.store.save_results_df(df, filename=filename)
        logger.info(f"Saved {len(df)} results ({condition}) to CSV")

        # Compute and display metrics
        if not df.empty and "correct" in df.columns:
            print("\n=== Results Summary ===\n")

            summary = generate_summary_table(df)
            print(summary.to_string(index=False))
            condition_suffix = {"3d": "_3d", "unity": "_unity"}.get(condition, "")
            summary_name = f"summary{condition_suffix}.csv"
            summary.to_csv(self.config.results_dir / "scores" / summary_name, index=False)

            print("\n--- Accuracy by Model ---")
            acc_model = compute_accuracy_by_group(df, "model_name")
            print(acc_model.to_string(index=False))

            print("\n--- Accuracy by Chart Type ---")
            acc_chart = compute_accuracy_by_group(df, "chart_type")
            print(acc_chart.to_string(index=False))

            print("\n--- Cost Metrics ---")
            cost = compute_cost_metrics(df)
            print(cost.to_string(index=False))
