# Presentation Script: Dataset & Evaluation Pipeline

---

## Part 1 — The Dataset

### Why We Needed a Specific Kind of Dataset

Before we talk about ChartX specifically, it is worth understanding what we were looking for and why most existing chart datasets would not work for this project.

Our central research question is: does the same VLM give different answers when it looks at a 2D chart versus a 3D chart that shows the exact same data? To answer this, we need strict experimental control. The data behind the chart must be identical across conditions — the only variable that changes is the visual rendering. Most chart understanding datasets, like ChartQA or PlotQA, provide chart images with pre-written questions, but they do not provide the raw data tables that produced those images. Without the raw data, we cannot re-render the chart in a different style. We would be stuck comparing across different datasets, different questions, and different data, which introduces confounds we cannot control for.

We also need to compute ground-truth answers programmatically. If we ask "what is the tallest bar?", we need to know the exact answer from the data, not from a human annotator who might round differently or phrase the answer ambiguously. Human-annotated ground truth introduces noise, and at the scale of 1,700 evaluations, even small inconsistencies would distort our accuracy measurements.

So our requirements were: a dataset with real-world data in structured tabular form, spanning multiple chart types and diverse topics, with enough records to sample from. ChartX met all of these.

### What Is ChartX?

ChartX is a large-scale chart understanding benchmark created by InternScience and publicly hosted on HuggingFace under an Apache 2.0 license. It is one of the most comprehensive chart benchmarks available today. The dataset contains over 6,000 chart records spanning 18 distinct chart types — bar charts, line charts, pie charts, radar charts, area charts, heatmaps, and more — drawn from 22 academic subject areas including economics, healthcare, demographics, environmental science, education, technology, and the arts.

What makes ChartX fundamentally different from datasets like ChartQA is that every record includes not just a chart image and questions, but the underlying CSV data table. This is the actual tabular data — the rows and columns of numbers — that was used to create the original chart. For example, a bar chart record from ChartX might contain a CSV with rows like "North: 80, South: 93, East: 85, West: 100" alongside the rendered bar chart image. This dual representation — image plus data — is exactly what enables our controlled experiment. We take the CSV data, discard the original image entirely, and re-render the data from scratch using our own chart generators. This guarantees that the only difference between our 2D and 3D versions of a chart is the rendering method, not the data or the visual style of the original.

The subject diversity is important too. Because ChartX draws from 22 different academic topics, the data has natural variation in scale, distribution shape, number of categories, and label complexity. A healthcare bar chart might have 4 regions with values between 80 and 100. An economics line chart might have 12 months of data across 4 series with values in the thousands. A nonprofit heatmap might have 6 rows and 6 columns with values spanning two orders of magnitude. This diversity means our evaluation is not biased toward any particular data profile — it reflects the kind of charts that VLMs would encounter in real-world usage.

### Our Hybrid Data Approach — and Why It Matters

Our evaluation covers six chart types: bar, line, scatter, heatmap, area, and stacked bar. We chose these six because they represent a broad spectrum of visual encoding strategies — position encoding (bar, scatter), connection encoding (line), area encoding (area, stacked bar), and color encoding (heatmap) — and because they are among the most commonly used chart types in data analysis and immersive analytics tools like DxR.

ChartX natively supports four of these six types. For bar charts, line charts, area charts, and heatmaps, we pull real-world data directly from ChartX. Each record gives us category labels, series names, and the numeric values. The data comes from actual published sources, so it has authentic distributions — not perfectly uniform random numbers, but the kind of messy, uneven distributions you see in real research data. Some bar charts have one bar dramatically taller than the others. Some line charts have crossing series. Some heatmaps have sparse regions and dense clusters. This ecological validity makes our evaluation more meaningful than one conducted entirely on synthetic data.

For scatter plots and stacked bar charts, however, ChartX does not provide suitable data. Scatter plots in our pipeline require explicit x and y coordinates with known cluster assignments — we need to ask questions like "how many clusters are visible?" and know the exact answer. ChartX's scatter data does not encode cluster structure in a way we can parse reliably. Similarly, stacked bar charts require multi-series segment breakdowns where we know the exact ratio of each segment, which ChartX records do not provide.

So for these two types, we generate synthetic data with carefully controlled statistical properties. Our scatter generator creates 3 to 5 Gaussian clusters with specified means and variances, plus optional outlier points. Our stacked bar generator creates 5 to 8 categories with 2 to 3 stacked segments whose ratios we know precisely. By controlling the generation, we can ask questions with unambiguous answers: the cluster count is exactly 4, the correlation direction is positive, the largest segment is exactly "Segment A."

This hybrid strategy — real data where available, controlled synthetic data where necessary — gives us the best of both worlds. Four of our six types use data from actual academic publications. Two use data we control precisely. In either case, every ground-truth answer is computed directly from the numbers, not estimated or annotated by hand.

### Parsing ChartX: The Technical Details

Loading data from ChartX is not as simple as calling a HuggingFace API and reading a DataFrame. The CSV data in ChartX is stored as a single string field within each record, and the delimiters are encoded as literal two-character sequences: the string `\n` (backslash followed by n) instead of an actual newline character, and `\t` (backslash followed by t) instead of an actual tab. This means the entire data table arrives as one long line of text.

Our loader module handles this in several steps. First, we replace the escaped sequences with their real counterparts. Then we pass the cleaned string through Python's `csv.DictReader` with tab as the delimiter, which gives us a list of dictionaries — one per row — where each key is a column header and each value is a string. We strip whitespace from all keys and values, and we skip any rows where the key itself is None, which can happen with malformed records.

Next, we run type-specific parsers. Each parser knows the expected structure for its chart type:

- The **bar chart parser** treats the first column as category names and all subsequent columns as numeric value series. It extracts the first series for a simple bar chart. If any value fails to parse as a float — after stripping commas, dollar signs, and percent signs — we reject the entire record and move to the next one.
- The **line chart parser** treats the first column as x-axis labels (typically time periods) and remaining columns as named series. It validates that every series has the same number of data points.
- The **area chart parser** uses the same logic as the line parser, since area charts are structurally identical to line charts but rendered with filled regions below the line.
- The **heatmap parser** treats the first column as row labels, the remaining column headers as column labels, and the cell values as the 2D data matrix. It constructs a list of lists representing the full grid.

We sample up to 50 records per chart type from the ChartX validation split. After filtering out records with unparseable data, this yields 300 charts per rendering condition — 50 per type times 6 types. Each chart gets a unique ID (like `bar_0001`, `heatmap_0027`), a PNG image, and a JSON sidecar file.

### How Ground-Truth Questions and Answers Are Generated

This is one of the most important design decisions in our pipeline, so it deserves a thorough explanation.

We do not use any pre-existing questions from ChartX or any other source. Every question in our evaluation is generated programmatically at chart-creation time, and every answer is computed directly from the data array. This means there is zero ambiguity in what the correct answer is, and the entire scoring process can be fully automated with no human judgment involved.

Each chart type has 2 to 3 associated task types, drawn from the visualization literacy framework proposed by Lee et al. (2017) and the VLAT assessment by Boy et al. (2014). Here is exactly what we test for each chart type:

**Bar charts** test three tasks:
- *Extremum detection*: "Which category has the highest/lowest value?" — we compute `argmax` or `argmin` on the values array and the answer is the corresponding category label.
- *Value retrieval*: "What is the value of [specific category]?" — we look up the exact value in the data.
- *Value comparison*: "Which has a higher value, [A] or [B]?" — we compare two randomly selected categories.

**Line charts** test three tasks:
- *Trend identification*: "What is the overall trend of [series name]?" — we compute the slope of the series and classify it as increasing, decreasing, or stable.
- *Max value*: "What is the peak value across all series?" — we compute the global maximum.
- *Value comparison*: "Which series has a higher value at [time point]?" — we compare two series at a randomly selected x position.

**Scatter plots** test three tasks:
- *Cluster count*: "How many distinct clusters are visible?" — we know the answer because we specified the number of clusters during generation.
- *Outlier presence*: "Are there any outliers?" — we know because we optionally injected outlier points.
- *Correlation direction*: "What is the overall correlation direction?" — we computed the Pearson correlation of the pooled x/y data.

**Heatmaps** test three tasks:
- *Max value cell*: "Which cell has the highest value?" — we find the argmax across the entire 2D matrix and return the row/column label pair.
- *Correlation direction*: "Is there a row-wise trend?" — we compute the direction of values across columns.
- *Part-to-whole / Comparison*: "Which cell is higher, (row A, col X) or (row B, col Y)?" — we directly compare two cell values.

**Area charts** test three tasks:
- *Trend identification*: same logic as line charts.
- *Value comparison*: comparing two series at a specific point.
- *Total comparison*: "Which time point has the highest total across all series?" — we sum series values at each x position.

**Stacked bar charts** test three tasks:
- *Part-to-whole*: "What percentage of the total does [segment] represent in [category]?" — computed from the known segment values.
- *Value retrieval*: "What is the value of [segment] in [category]?" — direct lookup.
- *Magnitude comparison*: "Which category has the largest total?" — sum all segments per category.

The key insight is that every one of these answers is deterministic. There is no judgment call, no "reasonable interpretation." The answer to "which bar is tallest?" is whichever label corresponds to the maximum value in the array. This makes automated scoring reliable and removes the need for human evaluation, which would be impractical at our scale of 1,700 questions.

---

## Part 2 — The Evaluation Pipeline

### What the Pipeline Does — The Big Picture

At the highest level, our pipeline does three things: it creates chart images from data, it asks a VLM questions about those images, and it checks whether the VLM answered correctly. But each of these steps involves significant engineering to make the results reliable, reproducible, and scalable.

The pipeline is written entirely in Python, uses asynchronous programming for API calls, and is orchestrated through a command-line interface with three subcommands: `generate`, `evaluate`, and `report`. The entire system is configured through a single YAML file and reads API keys from a `.env` file that is never committed to the repository.

### Stage 1: Chart Generation — Creating the Visual Stimuli

The chart generation stage is where we transform structured data into visual images. The critical design principle here is that we render every chart in at least two conditions — 2D and 3D — using the exact same underlying data. This is what makes our experiment a controlled comparison rather than a loose correlation.

**The 2D Baseline Condition:**

For the 2D condition, we use matplotlib with its default styling. Charts are rendered with flat 2D axes, solid color fills from the Seaborn Set2 palette, clear axis tick marks with numeric labels, a white background, and a standard legend when multiple series are present. The figure size is 9 by 6 inches at 150 DPI, producing clean, readable images.

The reason we chose standard matplotlib styling — rather than something custom or artistic — is deliberate. These charts look exactly like the ones found in textbooks, research papers, Jupyter notebooks, and data science blog posts. VLMs like GPT-5.2 have been trained on billions of images, and standard 2D charts are among the most common structured images in their training data. So the 2D condition represents the best-case scenario for VLM performance — the kind of chart the model has seen millions of times.

For each chart, we save two files: a PNG image and a JSON sidecar. The JSON sidecar stores everything we need to evaluate the model later: the chart type, the chart ID, the raw data values, the generated questions with their task types, and the ground-truth answers. This separation means we can regenerate images without losing evaluation data, and we can re-score existing results without re-rendering charts.

We also generate a manifest file — `manifest.json` — which is a single JSON file listing every chart in the dataset along with its image path, chart type, and associated questions. The evaluation stage reads this manifest to know which images to send to the VLM and which questions to ask.

**The 3D Matplotlib Condition:**

For the 3D condition, we take the identical data and render it using matplotlib's `mplot3d` extension. This is the same matplotlib library, but the charts are now projected into three-dimensional space with perspective rendering.

The transformation is dramatic. Bar charts become 3D rectangular blocks with depth, width, and height, rendered using `ax.bar3d()`. The perspective projection causes bars further from the camera to appear smaller — a phenomenon called foreshortening — which fundamentally changes the visual relationship between bar height and data value. Line charts gain a z-axis, with multiple series placed at different depths. Scatter plots are rendered as point clouds in 3D space. Heatmaps become surfaces or 3D bar grids colored by value. Area charts become ribbon surfaces. Stacked bars become towers of stacked cubes.

We set the camera angle to 20 degrees elevation and negative 50 degrees azimuth, with an immersive focal length of 0.2, and transparent panes with light gridlines. These settings produce a viewing angle that is visually compelling but introduces all the challenges of 3D visualization: foreshortening, occlusion (front elements hiding back elements), oblique axis labels, and depth ambiguity.

The crucial point is that the ground-truth answers remain identical. The tallest bar is still the tallest bar. The trend is still increasing. The cluster count is still 4. The only thing that changed is how the data is visually presented. So any difference in VLM accuracy between the 2D and 3D conditions is caused entirely by the visual rendering — not by the data, the questions, or the expected answers.

**The Unity 3D Condition (Planned):**

Our pipeline also supports a third condition: rendering charts through Unity, a professional 3D game engine. This is handled by a separate Unity project — `vlm-chart-renderer` — that receives JSON configuration files from Python and renders charts as actual 3D scenes with realistic materials, three-point studio lighting, antialiased rendering, and configurable camera positions.

The Unity condition represents the most realistic approximation of what a user would see in an immersive analytics environment — a VR or AR headset running a 3D visualization toolkit like DxR. The Python side generates the same data configurations as the matplotlib conditions, writes them to a JSON file, and invokes Unity in headless batch mode. Unity instantiates 3D GameObjects (cubes for bars, spheres for scatter points, mesh ribbons for area charts), positions the camera, renders the scene to a RenderTexture, and exports a PNG. The chart types — `bar_unity`, `scatter_unity`, `heatmap_unity`, and so on — produce images that look fundamentally different from matplotlib's pseudo-3D: they have real depth, real shadows, real material reflections.

**Deterministic Reproducibility:**

All chart generation is seeded. The default random seed is 42, and it is passed through to NumPy's random number generator. This means that running `generate` twice with the same parameters produces byte-identical charts. The synthetic scatter clusters will have the same center points, the same spread, and the same outlier positions. The ChartX records will be sampled in the same order. This reproducibility is essential for scientific validity — anyone with access to the code and ChartX dataset can regenerate our exact stimuli.

### Stage 2: VLM Evaluation — Querying the Models

The evaluation stage is the heart of the pipeline. This is where we take our chart images, send them to a commercial VLM API along with a question, and collect the model's response.

**The Provider Abstraction:**

We support four VLM providers through a common abstract interface. Every provider implements a single method — `query(image_path, prompt)` — that takes a chart image and a question string, and returns a standardized `VisionResponse` object containing the raw text response, latency in milliseconds, input and output token counts, and computed cost in USD.

The four providers are:
- **OpenAI** (direct API): for GPT-4o and GPT-4o-mini.
- **OpenRouter**: a proxy service that gives access to models like GPT-5.2 through an OpenAI-compatible API. This is what we currently use for our primary model.
- **Anthropic**: for Claude 3.5 Sonnet and other Claude models, using Anthropic's native message format with base64-encoded images.
- **Google Gemini**: for Gemini 2.0 Flash and Gemini 2.5 Flash, using Google's `genai` client with PIL Image objects.

Each provider has its own pricing table hardcoded in the client module. When a response comes back, we compute the cost by multiplying the input token count by the per-million-token input price and the output token count by the per-million-token output price. This gives us precise per-query cost tracking, which is how we know our entire evaluation cost $4.19.

**How a Single Query Works:**

When the pipeline evaluates a single chart question, the following happens:

1. It loads the chart image from disk and base64-encodes it. The encoding preserves the full image quality — we do not resize or compress.
2. It constructs an API message containing two content blocks: a text block with the question ("Which bar has the highest value?") and an image block with the base64 data and the correct MIME type (image/png).
3. Before sending, it checks the cache. Every prior response is stored as a JSON file in a directory named `{model}_{condition}` — for example, `gpt-5.2_chartx_3d`. If a cached file exists for this chart ID, task type, and trial number, the pipeline loads it directly and skips the API call entirely.
4. If no cache exists, it acquires a slot from an asyncio semaphore (concurrency limit of 5) and sends the request. The semaphore prevents us from overwhelming the API with too many simultaneous requests.
5. The API call is wrapped in a retry-with-exponential-backoff function. If the call fails due to a rate limit (HTTP 429) or server error (HTTP 500/529), it waits an exponentially increasing delay — 1 second, 2 seconds, 4 seconds — before retrying, up to 3 attempts.
6. When the response arrives, we create a result dictionary containing: model name, chart ID, chart type, task type, question, expected answer, raw response, latency, tokens, cost, trial number, and condition.
7. We write this result to the cache directory as a JSON file, so future runs will find it instantly.

**Asynchronous Execution and Concurrency:**

The pipeline uses Python's `asyncio` for concurrent API calls. For each model, all chart-question pairs are bundled into a list of coroutines, and they are dispatched through `asyncio.as_completed()` with a progress bar from tqdm. The semaphore limits concurrency to 5 simultaneous in-flight requests.

This matters for practical reasons. Our evaluation involves 850 questions per condition. If each API call takes about 3 seconds (typical for GPT-5.2 with an image), sequential execution would take over 42 minutes. With 5-way concurrency, it completes in about 8-9 minutes. But we cannot go higher than 5 without hitting rate limits, especially on OpenRouter, which aggregates traffic from many users.

**What Gets Recorded:**

Every response JSON captures comprehensive metadata beyond just the answer. The `latency_ms` field records wall-clock time from request send to response receive. The `input_tokens` and `output_tokens` fields come from the API response headers. The `cost_usd` is computed from our pricing tables. The `raw_response` preserves the complete unedited model output, which is essential for debugging and error analysis — we can go back and read exactly what the model said about any chart.

### Stage 3: Scoring and Reporting — Measuring Accuracy

Once we have all the VLM responses, we need to determine whether each one is correct. This is more complex than it sounds, because VLMs do not return tidy one-word answers. They return natural language paragraphs with explanations, qualifications, and sometimes contradictory statements.

**Response Parsing — Extracting the Answer:**

The first step is parsing: extracting the model's actual answer from its verbose response. We have two parsing strategies depending on the task type.

For numeric tasks — value retrieval, max value, max cell value, cluster count, and part-to-whole ratios — we need to extract a number. We use a regular expression that finds all numeric patterns in the response (integers, decimals, negative numbers), strips common decorations like dollar signs, percent symbols, and commas, and returns the last number found. We use the last number because VLMs typically restate the question first ("The chart shows values of 80, 93, 85, and 100...") and give their answer at the end ("...so the highest value is 100"). Taking the last number gives us the model's intended answer in most cases.

For text tasks — comparison, trend identification, extremum detection, correlation direction, outlier presence, magnitude comparison, total comparison, and value comparison — we keep the full response text, lowercase it, and collapse whitespace. We then check whether the expected keyword appears anywhere in the response.

**Scoring — Determining Correctness:**

After parsing, we score each response using one of three strategies:

*Relaxed numeric accuracy* is used for numeric tasks. We compare the parsed number to the ground truth with a 10% relative tolerance. If the ground truth is 100, any answer between 90 and 110 counts as correct. This tolerance accounts for the fact that reading exact values from a chart is difficult — especially in 3D where perspective distortion makes axis alignment imprecise. There are two special cases: if both values are zero, it is always correct; and if both values look like years (between 1900 and 2100), we require an exact match because a 10% tolerance on years would be meaninglessly wide.

*Keyword matching with synonym expansion* is used for text tasks. We check whether the expected answer appears in the model's response as a substring. But we also expand the expected answer through synonym groups. Our scorer maintains five synonym groups:
- "increasing" = "upward" = "rising" = "goes up" = "uptrend"
- "decreasing" = "downward" = "falling" = "goes down" = "downtrend"
- "stable" = "flat" = "constant" = "no change" = "unchanged"
- "positive" = "direct"
- "negative" = "inverse"

So if the ground truth is "increasing" and the model says "the trend is clearly upward over the time period," the synonym expansion matches "upward" to the "increasing" group, and the response is scored as correct. This prevents false negatives from legitimate paraphrasing.

*Exact match* is used for count tasks where the answer must be a precise integer. "How many clusters?" has an exact answer; 3 is not the same as 4.

**Aggregation and Metrics:**

After scoring all 850 items per condition, the pipeline constructs a pandas DataFrame with every result and computes several aggregate metrics:

- *Accuracy by model*: overall percentage of correct answers per model.
- *Accuracy by chart type*: how well the model performs on bars versus lines versus scatter plots, etc.
- *Accuracy by task type*: which cognitive tasks (value retrieval, trend identification, cluster counting) the model handles best and worst.
- *Mean and p95 latency*: how fast the model responds, and the tail latency.
- *Total cost and cost per correct answer*: both the raw expenditure and the economic efficiency — how much it costs to get one right answer.

These metrics are saved as CSV files with condition-specific names: `all_results_chartx_2d.csv`, `all_results_chartx_3d.csv`, and so on. The pipeline also generates a summary table and can produce publication-quality matplotlib figures (bar charts comparing 2D vs 3D accuracy, heatmaps of accuracy by chart type and condition, degradation waterfall charts, cost analysis plots).

### The Caching System — Why It Matters

The caching layer deserves emphasis because it is central to how we work iteratively with the pipeline.

Every VLM response is persisted as an individual JSON file in a directory tree structured as `results/responses/{model}_{condition}/{chart_id}_{task_type}_trial{n}.json`. The condition is encoded directly in the directory name — `gpt-5.2_chartx_2d` versus `gpt-5.2_chartx_3d` — so there is no risk of cross-condition collisions.

Before making any API call, the pipeline checks for a cached file. If one exists, it loads the result dictionary and returns it immediately. This means:
- We never pay for the same query twice.
- We can re-run the scoring pipeline with different tolerance settings or synonym groups without making any API calls.
- We can add new charts to the benchmark without re-evaluating existing ones.
- If an evaluation run is interrupted (network error, rate limit, laptop going to sleep), we can simply re-run the command and it will resume from where it left off, skipping all previously completed items.

This is particularly important for our project because API costs are real money. Our GPT-5.2 evaluation cost $4.19 for 1,700 queries. Without caching, every debugging run or scoring tweak would cost another $4.19.

### End-to-End Flow — Putting It All Together

Here is the complete workflow for reproducing our evaluation:

**Step 1 — Generate 2D ChartX charts:**
```
python -m vlm_eval generate --source chartx
```
This loads 50 records per chart type from ChartX, generates synthetic data for scatter and stacked bar, renders 300 charts in 2D matplotlib, computes ground-truth answers, and saves PNG images with JSON sidecars and a manifest file.

**Step 2 — Generate 3D ChartX charts:**
```
python -m vlm_eval generate --source chartx --condition 3d
```
This takes the exact same data (same seed, same records) and re-renders all 300 charts in 3D matplotlib. The ground-truth answers are identical.

**Step 3 — Evaluate on 2D charts:**
```
python -m vlm_eval evaluate --source chartx
```
This reads the 2D manifest, sends 850 question-image pairs to GPT-5.2, caches each response, parses and scores them, and writes `all_results_chartx_2d.csv`. Result: **82.0% accuracy**, $2.72 total cost.

**Step 4 — Evaluate on 3D charts:**
```
python -m vlm_eval evaluate --source chartx --condition 3d
```
Same process, same model, same questions, but now on 3D-rendered images. Result: **31.1% accuracy**, $1.47 total cost.

**Step 5 — Generate report:**
```
python -m vlm_eval report
```
This produces comparison figures, summary tables, and optionally a full PDF status report.

The same data, the same questions, the same model, the same scoring logic. The only variable that changed between Steps 3 and 4 is the visual rendering of the chart images. That 50.9 percentage point accuracy drop — from 82.0% to 31.1% — is entirely attributable to the shift from 2D to 3D visualization.

---

## Closing — Extensibility and Reproducibility

Our pipeline is designed from the ground up to be modular and extensible.

**Adding a new model** requires a single entry in `configs/default.yaml` — a name, provider, model ID, and optional temperature and token limit. The factory function in `clients.py` instantiates the right client class based on the provider string.

**Adding a new rendering condition** means writing a new chart generator module that takes the same data inputs and produces PNG images with the same JSON sidecar format. The evaluation and scoring stages do not need to change at all.

**Adding a new chart type** means implementing the generation function (how to render it) and defining the task types with their ground-truth computation (what questions to ask and how to compute the answer).

**Changing the scoring logic** — for example, adjusting the numeric tolerance from 10% to 5%, or adding new synonym groups — only requires modifying the scorer module. Because all responses are cached, no API calls are needed to re-score.

Everything is reproducible through deterministic seeding, everything is cached to avoid redundant API costs, and every result can be traced back to its source data, chart image, model response, and scoring decision. There is no manual annotation, no subjective judgment, and no human-in-the-loop at any stage of the evaluation.
