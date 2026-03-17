# Presentation Script: Dataset & Pipeline

## Part 1 — The Dataset

### Opening

For this project, we needed a dataset that gives us real-world chart data — not just chart images, but the actual underlying numbers. That way we can re-render the same data in different visual conditions and have ground-truth answers to score the model against. The dataset we chose is called ChartX.

### What Is ChartX?

ChartX is a large-scale chart understanding benchmark published by InternScience and hosted on HuggingFace. It contains over 6,000 chart images spanning 18 different chart types and 22 academic topics — things like economics, healthcare, demographics, environment, and education. These are real charts pulled from actual academic publications and reports.

What makes ChartX especially useful for our work is that it doesn't just give us images. Each record in the dataset includes the underlying CSV data table — the actual rows and columns of numbers that produced the chart. This is critical because it means we can take that raw data and re-render it ourselves in whatever visual format we want: a flat 2D matplotlib chart, a 3D matplotlib projection, or even a Unity 3D scene. The data stays the same; only the visual representation changes. That's how we isolate the effect of 3D rendering on model accuracy.

### Our Hybrid Approach

ChartX supports four of our six chart types natively: bar charts, line charts, area charts, and heatmaps. For these, we pull real-world CSV data directly from the dataset. The data covers diverse topics — healthcare statistics, artwork pricing, nonprofit performance metrics, industry distributions — so we get a wide variety of real-world distributions and scales.

For the other two chart types — scatter plots and stacked bar charts — ChartX does not provide suitable data. Scatter plots require explicit x/y coordinate pairs with cluster structure, and stacked bar charts need multi-series breakdowns that ChartX records don't encode. So for these two types, we generate synthetic data with controlled statistical properties: known cluster counts, known correlations, known segment ratios. This hybrid approach gives us the ecological validity of real-world data where possible, and the control of synthetic data where necessary.

### How We Load and Parse ChartX Data

Our pipeline has a dedicated ChartX loader module. It connects to HuggingFace, downloads the dataset, and filters records to our four supported types. The tricky part is parsing: ChartX stores its CSV data as a single string field with literal backslash-n and backslash-t characters as delimiters — not actual newlines and tabs. Our parser converts these escaped sequences into real delimiters, then uses Python's csv module to extract a clean list of row dictionaries.

We then run type-specific parsers. For bar charts, we extract category names from the first column and numeric values from the remaining columns. For line charts, we extract x-axis labels and multiple series. For heatmaps, we extract row labels, column labels, and a 2D matrix of values. Each parser validates that all values are actually numeric — if a record has unparseable data, we skip it and move to the next one. We sample 50 charts per type, yielding 300 charts per condition.

### Ground Truth Generation

This is important: we don't use any pre-existing questions from ChartX. We generate our own questions and ground-truth answers at chart-generation time. For each chart, we compute 2–3 task-specific questions based on the actual data:

- For a bar chart, we might ask: "Which category has the highest value?" and the ground truth is computed by finding the argmax of the values array.
- For a line chart: "What is the overall trend?" — computed by fitting the direction of the series.
- For a scatter plot: "How many distinct clusters are visible?" — we know the answer because we generated the clusters.
- For a heatmap: "Which cell has the highest value?" — computed directly from the data matrix.

Every question has a programmatically verified answer. There is no human annotation and no ambiguity. This is what lets us do fully automated scoring at scale.

---

## Part 2 — The Evaluation Pipeline

### Pipeline Overview

Our evaluation pipeline is a fully automated system built in Python. It has three stages: chart generation, VLM evaluation, and scoring with reporting. The entire pipeline runs from the command line with a single command per stage.

### Stage 1: Chart Generation

The first stage takes data — either from ChartX or synthetic generation — and produces chart images with paired metadata.

For 2D charts, we render using standard matplotlib with default styling: flat 2D axes, solid color fills, clear axis tick marks, a white background, and a standard legend. These charts look exactly like the ones you would find in a textbook or a research paper — the kind of images that VLMs have seen millions of times during pre-training.

For 3D charts, we take the exact same data and render it using matplotlib's mplot3d extension. This adds a third axis, projects the data into 3D space, and renders it in perspective. Bars become 3D rectangular blocks, scatter points gain a z-coordinate, and axes are rendered at oblique angles. The viewing angle is set to 20 degrees elevation and negative 50 degrees azimuth with an immersive focal length. This condition isolates the effect of 3D geometry on VLM accuracy.

For Unity 3D — which is our planned third condition — the pipeline writes JSON configuration files and invokes Unity in batch mode. Unity renders the charts as real 3D scenes with proper lighting, materials, and camera perspectives. This gives us the most realistic immersive condition.

Every chart image is saved as a PNG alongside a JSON sidecar file. The sidecar contains the chart type, the ground-truth answers for each task type, the questions, and the full data used to generate the chart. We also generate a manifest file — a single JSON listing every chart and its associated questions — which the evaluation stage reads.

### Stage 2: VLM Evaluation

The second stage sends each chart image to a Vision-Language Model and collects its response. Currently, we have evaluated GPT-5.2 from OpenAI, accessed through OpenRouter's API. The pipeline supports multiple providers — OpenAI direct, Anthropic for Claude, Google for Gemini, and OpenRouter — through a common abstract interface.

Here is how a single evaluation works:

1. The pipeline loads a chart image and a question from the manifest.
2. It base64-encodes the image and constructs an API request with the image and the question text.
3. It sends the request asynchronously to the VLM API with a concurrency limiter — we run up to 5 parallel requests to balance speed with rate limits.
4. When the response comes back, we record the raw text, the latency in milliseconds, the input and output token counts, and the computed cost in USD based on the provider's pricing table.
5. The result is cached as a JSON file keyed by model name, chart ID, and condition. If we re-run the evaluation, cached results are loaded instantly without making another API call.

For our ChartX evaluation, we had 850 question-answer pairs per condition — 6 chart types times 50 charts times roughly 2.8 questions per chart. Across both 2D and 3D conditions, that is 1,700 total API calls. The entire evaluation cost $4.19.

### Stage 3: Scoring and Reporting

The third stage parses the VLM's free-text responses and scores them against our ground truth. This is not trivial because VLMs don't return clean answers — they return full sentences, explanations, and sometimes hedging language.

For numeric tasks like value retrieval, max value, or cluster count, we extract numbers from the response using regular expressions. We look for the last number in the response, since VLMs often restate the question before giving the answer. We then compare this extracted number to the ground truth using a 10% relative tolerance. If the model says 87 and the ground truth is 85, that is within tolerance and counts as correct.

For text-based tasks like trend identification, comparison, or correlation direction, we search the full response for expected keywords. We also use synonym expansion — if the ground truth is "increasing" but the model says "upward" or "rising," that still counts as correct. We maintain synonym groups: increasing maps to upward, rising, goes up, and uptrend. Decreasing maps to downward, falling, and so on.

After scoring, the pipeline aggregates results into a pandas DataFrame and computes metrics: accuracy by model, accuracy by chart type, accuracy by task type, mean and p95 latency, total cost, and cost per correct answer. Results are saved as condition-specific CSV files and summary tables.

### The Caching System

One practical detail worth mentioning: every VLM response is cached to disk as a JSON file in a directory structure organized by model name and condition. The directory name encodes both — for example, `gpt-5.2_chartx_3d` — so evaluations across different conditions never collide. If a chart has already been evaluated, the pipeline loads the cached result and skips the API call. This means we can re-run the scoring logic with different tolerance thresholds or synonym groups without re-querying the API.

### End-to-End Flow

To summarize the full flow:

1. **Generate**: `python -m vlm_eval generate --source chartx` — loads ChartX data, renders 300 charts in 2D with ground truth
2. **Generate 3D**: `python -m vlm_eval generate --source chartx --condition 3d` — renders the same 300 data points in 3D
3. **Evaluate 2D**: `python -m vlm_eval evaluate --source chartx` — sends 850 questions to GPT-5.2, caches responses, scores them → 82.0% accuracy
4. **Evaluate 3D**: `python -m vlm_eval evaluate --source chartx --condition 3d` — same 850 questions on 3D charts → 31.1% accuracy
5. **Report**: `python -m vlm_eval report` — generates comparison figures, heatmaps, and summary tables

The same data, the same questions, the same model — the only thing that changed was the visual rendering. That 50.9 percentage point drop from 82% to 31% is entirely caused by the shift from 2D to 3D.

---

## Closing

Our pipeline is designed to be extensible. Adding a new model is a single entry in the YAML config file. Adding a new rendering condition means writing a new chart generator module. Adding a new chart type means implementing the generation function and defining the task types with their ground-truth computation. Everything is modular, reproducible, and fully automated — no manual annotation, no subjective scoring, no human-in-the-loop at evaluation time.
