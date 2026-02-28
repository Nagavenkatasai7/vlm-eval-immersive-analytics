# Status Report: An Empirical Evaluation of Multi-LLM and VLM Capabilities for Visualization Literacy in Immersive Analytics

**Team**: Naga Venkata Sai Chennu (G01514409), Hemanjali Buchireddy (G01520809)
**Course**: CS 692 — Mobile Immersive Computing, Spring 2026
**Instructor**: Dr. Bo Han | **Date**: March 6, 2026

---

## Project Overview

We are building an automated evaluation pipeline to benchmark how well Vision-Language Models (VLMs) can interpret data visualizations — a critical capability for AI-assisted immersive analytics. Our pipeline generates chart stimuli with ground-truth answers, queries VLMs via API, parses responses, and scores them against established visualization literacy task taxonomies.

## Progress Summary

Since our proposal presentation on February 6, we have completed the following:

**1. Evaluation Pipeline (Complete)**
We implemented a modular Python pipeline (`vlm-eval`) with five major components: (a) a chart stimulus generator producing 6 chart types (bar, line, scatter, heatmap, area, stacked bar) with auto-generated ground truth; (b) an async VLM client layer supporting OpenAI, Anthropic, and Google APIs with rate limiting; (c) a response parser handling numeric, categorical, and boolean outputs; (d) a scoring engine with three methods — relaxed numeric accuracy (±5% tolerance), keyword matching with synonym expansion, and exact match; and (e) a metrics module computing accuracy by group, cost metrics, and publication-quality figures.

**2. Benchmark Dataset (Complete)**
We generated 300 chart images (50 per type) with 850 total evaluation items spanning 13 task types derived from IVLAT (value retrieval, extremum detection, trend identification, comparison, cluster counting, correlation direction, outlier detection, part-to-whole estimation, magnitude comparison, and max-value identification).

**3. Two-Model Evaluation (Complete)**
We evaluated Claude 3.5 Sonnet and Gemini 2.5 Flash on all 850 items each (1,700 total). Key results:

| Chart Type | Claude 3.5 Sonnet | Gemini 2.5 Flash |
|---|---|---|
| Bar | 100.0% | 100.0% |
| Heatmap | 100.0% | 100.0% |
| Line | 90.0% | 90.7% |
| Scatter | 68.7% | 90.0% |
| Stacked Bar | 73.0% | 50.0% |
| Area | 68.0% | 60.0% |
| **Overall** | **83.9%** | **83.6%** |

| Task Category | Claude 3.5 Sonnet | Gemini 2.5 Flash |
|---|---|---|
| Value Retrieval | 98.7% | 98.7% |
| Comparisons (value, total) | 100.0% | 100.0% |
| Extremum Detection | 93.0% | 100.0% |
| Cluster Counting | 42.0% | 98.0% |
| Trend Identification | 85.0% | 68.0% |
| Part-to-Whole | 46.0% | 0.0% |

**Cost**: Claude $3.52 ($0.004/task); Gemini $0.11 ($0.0001/task) — 31x cheaper at equivalent accuracy. Mean latency: Claude 3.8s, Gemini 4.5s per task.

**Key Findings**: (1) Overall accuracy is nearly identical (83.9% vs 83.6%), but the models exhibit complementary strengths — Gemini excels at spatial tasks like cluster counting (98% vs 42%) and scatter plots (90% vs 69%), while Claude is stronger at compositional tasks like stacked bar interpretation (73% vs 50%) and trend identification (85% vs 68%). (2) Gemini achieves parity at 31x lower cost, suggesting lightweight VLMs may be sufficient for many visualization literacy tasks. (3) Both models achieve perfect accuracy on structured chart types (bar, heatmap) but diverge on charts requiring holistic visual reasoning.

## Remaining Work

| Week | Deliverable |
|---|---|
| Mar 7–14 | Add GPT-4o for 3-model comparison |
| Mar 15–28 | Immersive condition: Unity 3D chart renders + VR frame captures (with Fahim) |
| Mar 29–Apr 10 | System state metadata experiments (RQ2), consistency analysis |
| Apr 11–17 | Final analysis, figures, presentation preparation |

## Challenges

The primary challenge was building a robust parsing pipeline — VLM responses are verbose and unstructured, requiring careful extraction of numeric values, category names, and directional keywords. We solved this with a multi-strategy parser that prefers decimal numbers over step numbering and uses synonym groups for trend/direction matching. API key management across providers also required careful environment configuration.
