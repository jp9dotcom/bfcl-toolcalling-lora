# BFCL Tool-Calling with LoRA

A notebook-based research project exploring reliable LLM tool calling with the **Berkeley Function-Calling Leaderboard (BFCL)**. The experiments focus on three behaviours: selecting and calling an available tool, asking for missing required information, and abstaining when no supplied tool can handle a request.

The repository includes BFCL baseline workflows, a data-preparation pipeline using xLAM and ToolACE, Qwen tool-calling experiments, and a 45-case diagnostic probe suite.

> **Important status note:** The repository is an experimental notebook project. The checked-in `qwen-lora.ipynb` currently demonstrates 4-bit Qwen model loading and tool-call inference; it does **not** contain a complete LoRA training loop. The data-preparation notebook does implement dataset processing and JSONL export. This README distinguishes implemented workflows from intended next steps.

---

## Contents

- [Motivation](#motivation)
- [Project goals](#project-goals)
- [Repository structure](#repository-structure)
- [Workflow](#workflow)
- [Data preparation](#data-preparation)
- [Prompt and output format](#prompt-and-output-format)
- [Models and inference](#models-and-inference)
- [LoRA and QLoRA: technical foundations](#lora-and-qlora-technical-foundations)
- [BFCL baseline evaluation](#bfcl-baseline-evaluation)
- [Diagnostic probe evaluation](#diagnostic-probe-evaluation)
- [Setup](#setup)
- [Running the notebooks](#running-the-notebooks)
- [Interpreting probe results](#interpreting-probe-results)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

## Motivation

An effective tool-using language model must do more than generate valid-looking JSON. It must determine whether a tool is appropriate, select the correct function, extract arguments from the user's request, and avoid inventing information.

Three distinct decisions are particularly important:

1. **Call:** A supplied tool supports the request and the required arguments are available.
2. **Ask:** A relevant tool exists, but one or more required arguments are missing. The model should ask the user instead of guessing or silently inserting defaults.
3. **Abstain:** None of the supplied tools can perform the requested operation. The model should communicate this instead of forcing an unrelated call.

A further failure mode is **fabrication**: the model responds as though it retrieved external information even though it did not call the tool. This project investigates these behaviours through BFCL workflows and targeted probes.

## Project goals

- Establish a reproducible base-model evaluation workflow using BFCL.
- Prepare function-calling examples from public datasets.
- Normalize tool schemas and convert different call representations into a common format.
- Filter structurally invalid and potentially ungrounded calls.
- Reduce direct and near-duplicate overlap with BFCL evaluation queries.
- Explore Qwen models and quantized inference on Kaggle GPUs.
- Inspect call, ask, abstain, and fabricated-answer failure modes.
- Provide a foundation for a future, fully implemented LoRA fine-tuning and adapter-vs-baseline comparison.

## Repository structure

| File | Purpose |
|---|---|
| `bfcl-baseline.ipynb` | Installs BFCL, starts a local Qwen inference endpoint, and contains BFCL generation/evaluation workflows. |
| `bfcl-baseline-2` | Additional exported notebook artifact with a baseline evaluation workflow and recorded environment setup. |
| `training-data.ipynb` | Downloads xLAM and ToolACE, parses examples, normalizes schemas, validates calls, decontaminates queries, and exports JSONL splits. |
| `qwen-lora.ipynb` | Demonstrates 4-bit Qwen loading and simple tool-calling inference. The checked-in version does not implement a full LoRA trainer. |
| `probe-eval.ipynb` | Defines eight tools and a 45-case call/abstain/ask diagnostic suite for two Qwen variants. |
| `probe_results_v2.json` | Saved raw responses and per-case results from the diagnostic probe runs. |
| `LICENSE` | MIT License. |

The generated train/validation JSONL files and audit outputs are written to Kaggle's `/kaggle/working/` directory when the relevant data-preparation cells are run; they are not included as ordinary checked-in files in the current repository listing.

## Workflow

The intended experiment flow is:

```text
xLAM Function Calling 60k       ToolACE
          \                       /
           \                     /
            v                   v
              Parse and normalize
                      |
                      v
             Structural validation
                      |
                      v
           BFCL query decontamination
                      |
                      v
          BFCL-compatible prompt rendering
                      |
                      v
             Train / validation JSONL
                      |
            Fine-tuning (planned /
              not yet in notebook)
                      |
          +-----------+-----------+
          |                       |
          v                       v
    BFCL evaluation         Diagnostic probes
```

The baseline evaluation and probe evaluation are separate experiments. Probe metrics are not official BFCL scores.

## Data preparation

The data pipeline is implemented in `training-data.ipynb` and uses two source datasets.

### xLAM Function Calling 60k

Dataset: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)

The notebook loads the `train` split (60,000 records) and uses the `id`, `query`, `answers`, and `tools` fields. Records provide a user request, tool definitions, and target function calls.

The notebook notes that the dataset is gated. Accept the access terms on its Hugging Face page before attempting to download it. The notebook identifies its license as CC-BY-4.0; verify the current dataset card and attribution conditions before redistributing data or derived artifacts.

xLAM supplies positive tool-calling examples. It does not, by itself, provide the full range of “no tool fits” and “required parameter missing” cases needed for the project's call/ask/abstain objective.

### ToolACE

Dataset: [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)

The notebook downloads `data.json` from the dataset repository. ToolACE conversations include function calls as well as natural-language assistant turns for cases where a function is unsuitable or more information is required.

The notebook identifies ToolACE as Apache-2.0. Check the dataset card for current license and redistribution terms before using its contents outside the experiment.

### Parsing and schema normalization

The data notebook converts both sources into a common internal example representation.

For xLAM, it:

- Parses JSON-encoded tool definitions and answers.
- Maps Python-style parameter types into BFCL-style names, such as `str` → `string`, `int` → `integer`, `list` → `array`, and `object` → `dict`.
- Builds a parameter schema with properties and required-field information.
- Converts answer records into named calls with argument dictionaries.

For ToolACE, it:

- Extracts tool definitions from the conversation's system content.
- Parses bracket-style calls such as `[function_name(arg="value"), other_function(x=1)]`.
- Uses Python AST parsing for keyword arguments and literal values.
- Rejects unsupported positional-argument syntax instead of guessing which parameter each value represents.
- Retains non-call natural-language responses and heuristically labels them as ask or abstain/other.

### Structural validation and grounding

For call targets, the notebook checks:

- The target function exists in the supplied tool list.
- Required arguments are present.
- Argument names are valid for the tool schema.
- Required string arguments are grounded in the user query, except where an enum is defined.

The notebook sets `DROP_UNGROUNDED = True`, so ungrounded required string values are rejected in that pipeline version.

These checks are heuristic data-quality controls. They do not establish semantic correctness and do not replace manual review or execution-based validation.

### BFCL decontamination

The notebook extracts user queries from BFCL v4 data and compares source-dataset queries against them. It uses:

- Normalized exact-text matching.
- TF-IDF with unigram and bigram features.
- Cosine similarity.
- A near-duplicate threshold of **0.80**.

One recorded run reports 4,826 unique BFCL/additional blocked queries, 6 exact matches removed, 193 near matches removed, and 58,824 examples remaining at that stage.

This is lexical decontamination, not semantic proof of independence. It may miss paraphrases and may flag unrelated queries that share vocabulary.

### Sampling, length filtering, and splits

The more developed data-preparation path:

- Samples up to 4,000 xLAM examples.
- Combines them with accepted ToolACE examples.
- Renders prompt and completion text in the target format.
- Uses the Qwen3-1.7B tokenizer to measure total prompt-plus-completion length.
- Applies a 2,048-token maximum.
- Targets a 5% validation split with random seed 0.
- Groups examples by normalized query plus tool context so matching contexts remain in the same split.
- Exports `train.jsonl`, `val.jsonl`, and `dataset_manifest.json`.

A recorded later run reports **13,090 training rows**, **689 validation rows**, and zero train/validation group overlap. Earlier cells contain an older split implementation with different counts (3,800 training and 200 validation rows). These represent different notebook states, not a single consistent output. For new experiments, restart the notebook kernel and run the chosen pipeline from the beginning, then record the resulting manifest.

The later export includes `id`, `source`, `kind`, `prompt`, `completion`, and `n_tokens`. The notebook also creates audit artifacts under `/kaggle/working/data_audit/` in its later pipeline version.

Expected generated files:

```text
/kaggle/working/data/
├── train.jsonl
├── val.jsonl
└── dataset_manifest.json

/kaggle/working/data_audit/
├── structural_rejections.jsonl
└── ungrounded_review.jsonl
```

Actual filenames and counts should be taken from the output of the run being reported.

## Prompt and output format

Training examples are rendered to resemble the Qwen format expected by BFCL's Qwen handler. The prompt includes tool definitions inside a `<tools>...</tools>` block, a user message, and an assistant generation prefix. A function-call completion is encoded inside a `<tool_call>...</tool_call>` block.

Simplified example:

```text
<tool_call>
{"name": "get_weather", "arguments": {"city": "Mumbai"}}
</tool_call>
```

For multiple calls, the completion contains multiple tool-call blocks. Ask and abstain targets are natural-language completions rather than tool-call blocks.

Prompt consistency matters: a model trained on one tool-schema representation and evaluated with a materially different representation may fail to produce outputs the evaluator can parse. The notebook includes a comparison of its prompt renderer against BFCL's handler for benchmark entries.

## Models and inference

### Qwen3-1.7B baseline

The baseline notebook loads `Qwen/Qwen3-1.7B` in FP16, places it on GPU 0, and uses SDPA attention. It adds a no-thinking prefix for this hybrid Qwen3 model.

The notebook wraps generation in a small FastAPI service implementing:

- `GET /v1/models`
- `POST /v1/completions`

This supplies BFCL with model completions. The service does **not** execute external tools; it returns model-generated text for the BFCL evaluation harness to process.

### Qwen3-4B-Instruct-2507

The probe notebook evaluates `Qwen/Qwen3-4B-Instruct-2507` as a second model configuration.

### 4-bit quantized loading

The `qwen-lora.ipynb` notebook demonstrates bitsandbytes 4-bit loading with:

- NF4 quantization.
- Double quantization enabled.
- FP16 compute dtype.
- GPU 0 device map.
- SDPA attention.

A recorded Kaggle session used two NVIDIA T4 GPUs but explicitly placed the model on GPU 0. The notebook recorded approximately 2.68 GB of GPU memory allocated after loading Qwen3-4B-Instruct-2507 in 4-bit mode in that session.

**Scope clarification:** this notebook currently demonstrates quantized loading and inference examples. It does not include the adapter configuration, trainer, training loop, or adapter-save steps required to reproduce LoRA fine-tuning.

## LoRA and QLoRA: technical foundations

This section explains the parameter-efficient fine-tuning method that motivates the repository. The equations below describe the standard LoRA/QLoRA formulation; they are **methodological background**, not a claim that the current notebook has trained an adapter. The checked-in `qwen-lora.ipynb` currently performs quantized model loading and inference only.

### 1. Why parameter-efficient fine-tuning?

A transformer contains large weight matrices learned during pretraining. Full fine-tuning updates every parameter, which can require substantial GPU memory for:

- The base model weights.
- Gradients for trainable parameters.
- Optimizer states (for example, first and second moments in AdamW).
- Activations retained for backpropagation.

LoRA (Low-Rank Adaptation) reduces the number of trainable parameters by freezing the pretrained weights and learning a low-rank update to selected linear layers.

Let a pretrained linear layer have weight matrix:

`W_0 \in \mathbb{R}^{d_{out}\times d_{in}}`.

For an input activation `x`, the original layer computes:

`h = W_0 x`.

Full fine-tuning learns a dense update `\Delta W`:

`h = (W_0 + \Delta W)x`.

LoRA instead constrains the update to a low-rank factorization:

`\Delta W = BA`

where:

- `A \in \mathbb{R}^{r\times d_{in}}`
- `B \in \mathbb{R}^{d_{out}\times r}`
- `r` is the adapter rank, typically much smaller than `d_{in}` and `d_{out}`.

The forward pass becomes:

`h = W_0x + BAx`.

The base matrix `W_0` is frozen. Only `A` and `B` are optimized. A common LoRA implementation scales the adapter update by `\alpha/r`:

`h = W_0x + \frac{\alpha}{r}BAx`.

Here, `\alpha` controls the contribution of the adapter and `r` controls the rank/capacity of the update. Some implementations use alternative rank scaling; the exact convention should be recorded with the experiment.

### 2. Why does low rank save parameters?

A dense update to a layer with dimensions `d_{out}\times d_{in}` requires:

`N_{dense}=d_{out}d_{in}`

trainable parameters.

The LoRA factors require:

`N_{LoRA}=rd_{in}+d_{out}r=r(d_{in}+d_{out})`.

For an illustrative square projection of size `4096\times4096` with rank `r=16`:

| Approach | Trainable parameters |
|---|---:|
| Dense weight update | 16,777,216 |
| LoRA factors | 131,072 |

That is approximately **0.78%** as many trainable parameters for this one projection. This is an illustrative calculation, not a measurement of the exact Qwen layers or adapter configuration in this repository.

The savings become substantial when adapters are applied across many transformer projections. The trade-off is that a low-rank update has less freedom than an unrestricted dense update. A larger rank increases adapter capacity and trainable parameters; a smaller rank is cheaper but may underfit the task.

### 3. Where are LoRA adapters applied?

In transformer language models, LoRA is commonly attached to selected linear projections inside attention and/or the feed-forward network.

For an attention block, the projections typically include:

- Query projection (`W_q`)
- Key projection (`W_k`)
- Value projection (`W_v`)
- Attention output projection (`W_o`)

Some recipes also target MLP projections. The choice of target modules changes both the number of trainable parameters and the adaptation capacity.

For tool calling, adapting attention projections can help the model attend to the user request, available tool descriptions, parameter schemas, and relevant context when producing a structured call. This is a motivation, not a guarantee: the model still needs representative training examples and a consistent output format.

**Important:** the current repository does not record a verified LoRA target-module list, rank, alpha, dropout, or trained adapter checkpoint. Those values must be selected, implemented, and reported when the training stage is added.

### 4. What is QLoRA?

QLoRA combines a frozen, quantized base model with trainable LoRA adapters. Instead of storing the full base weights in 16-bit precision, the base weights are stored in a lower-bit representation and dequantized as needed for computation.

Conceptually, the layer is:

`h = \operatorname{dequant}(Q(W_0))x + \frac{\alpha}{r}BAx`.

Where:

- `Q(W_0)` is the quantized representation of the frozen pretrained weights.
- `\operatorname{dequant}` maps the quantized weights to the compute representation for the forward operation.
- `A` and `B` remain trainable adapter matrices.

The common QLoRA recipe uses:

- **4-bit NormalFloat (NF4):** a 4-bit data type designed for representing normally distributed neural-network weights.
- **Double quantization:** quantizes the quantization constants as well, reducing additional memory overhead.
- **Paged optimizers:** help manage optimizer memory spikes in supported training setups.
- **Low-rank adapters:** only adapter parameters receive gradient updates; the quantized base weights remain frozen.

Quantization reduces base-weight memory, but it does not eliminate activation memory, optimizer memory for adapters, or all runtime overhead. Actual memory use depends on model size, context length, batch size, attention implementation, and training configuration.

### 5. What the current notebook actually configures

The checked-in `qwen-lora.ipynb` demonstrates the following 4-bit loading configuration for `Qwen/Qwen3-4B-Instruct-2507`:

| Setting | Notebook configuration |
|---|---|
| Quantization | 4-bit |
| Quantization type | NF4 |
| Double quantization | Enabled |
| Compute dtype | FP16 |
| Device placement | GPU 0 |
| Attention implementation | SDPA |

This is the quantized **inference/model-loading** portion of a potential QLoRA workflow. It is not evidence of adapter training. In particular, the current notebook does not provide a complete `LoraConfig`, optimizer, trainer, loss computation, training loop, or adapter export.

### 6. Supervised fine-tuning objective for tool calling

Once a LoRA trainer is implemented, the prepared prompt/completion records can be used for supervised fine-tuning (SFT).

Let:

- `x` be the prompt containing the user request and available tool schemas.
- `y=(y_1,\ldots,y_T)` be the target assistant completion.
- `\theta` be the frozen base-model parameters.
- `\phi` be the trainable LoRA parameters.

The model learns to maximize the likelihood of the target completion conditioned on the prompt:

`P_{\theta,\phi}(y\mid x)=\prod_{t=1}^{T}P_{\theta,\phi}(y_t\mid x,y_{<t})`.

The standard token-level negative log-likelihood objective is:

`\mathcal{L}_{SFT}(\phi)=-\sum_{t=1}^{T}\log P_{\theta,\phi}(y_t\mid x,y_{<t})`.

Across a dataset of examples, the training objective averages or sums this loss over the target tokens and examples. If the implementation uses completion-only loss, prompt tokens are masked from the loss; whether that masking is applied must be explicitly verified in the trainer configuration.

For this project, target completions should represent all three decision types:

1. **Call:** a correctly formatted `<tool_call>` with the expected function name and arguments.
2. **Ask:** a natural-language clarification request when required arguments are absent.
3. **Abstain:** a natural-language response when no available tool can perform the task.

This is important because training only on successful tool calls can teach the model to call too aggressively. The ask and abstain examples teach the decision boundary around tool use.

### 7. How LoRA parameters are optimized

During backpropagation, gradients are computed for the adapter factors while the pretrained model weights remain frozen:

`\nabla_{W_0}\mathcal{L}=0,\qquad \nabla_A\mathcal{L}\ne0,\qquad \nabla_B\mathcal{L}\ne0`.

An optimizer such as AdamW updates the adapter parameters. The exact optimizer, learning rate, schedule, batch size, gradient accumulation, warmup, epochs, and checkpoint-selection criterion should be included in a future run manifest. They are not specified as completed training settings in the current repository.

### 8. How to evaluate whether the adapter helped

A valid before/after comparison should keep the benchmark version, test categories, prompt template, decoding parameters, and evaluation implementation fixed.

At minimum, compare:

- BFCL overall and category-level accuracy.
- Tool/function selection correctness.
- Argument-name and argument-value correctness.
- Multi-call and parallel-call behaviour.
- Irrelevance / unsupported-request handling.
- Missing-argument clarification behaviour.
- Invalid or fabricated tool-call frequency.

The 45-case probe suite is useful for targeted call/ask/abstain diagnostics, but it is not a substitute for BFCL's official scoring. Results should be reported separately.

---

## BFCL baseline evaluation

The baseline notebook creates a Python 3.11 virtual environment and installs `bfcl-eval==2026.3.23`. It starts a local FastAPI completion endpoint and invokes BFCL generation and scoring.

The recorded full-category configuration includes:

- `simple_python`
- `multiple`
- `parallel`
- `parallel_multiple`
- `irrelevance`
- `live_simple`
- `live_multiple`
- `live_parallel`
- `live_parallel_multiple`
- `live_irrelevance`
- `live_relevance`

The notebook reports **3,491 expected cases** across these categories for the installed BFCL package version.

Recorded baseline settings:

| Setting | Value |
|---|---|
| Base weights | `Qwen/Qwen3-1.7B` |
| BFCL model identifier | `Qwen/Qwen3-1.7B-FC` |
| BFCL package | `bfcl-eval==2026.3.23` |
| Precision | FP16 |
| Attention | SDPA |
| Decoding | Greedy when temperature ≤ 0.01 |
| Maximum new-token cap | 768 |
| Thinking prefix | Disabled |
| Local endpoint | `127.0.0.1:1053` |

There is also a focused irrelevance-family run using the `irrelevance`, `live_irrelevance`, and `live_relevance` categories. That run injects a careful system instruction: call only when a supplied tool supports the request and required arguments are present; ask for missing information; abstain if no tool fits; and never claim a tool result without actually calling a tool.

The notebook contains the generation and evaluation code, but this README does not claim an overall BFCL score. A complete, verified set of final score CSVs for every relevant configuration was not established from the repository artifacts reviewed.

## Diagnostic probe evaluation

The probe notebook defines eight toy tools:

- `get_weather`
- `send_email`
- `create_calendar_event`
- `search_web`
- `get_stock_price`
- `set_reminder`
- `convert_currency`
- `read_file`

It tests **45 hand-authored cases**:

| Category | Cases | Expected behavior |
|---|---:|---|
| Call | 15 | Emit the expected tool call(s) and arguments |
| Abstain | 15 | Make no tool call and communicate that the request is unsupported |
| Ask | 15 | Make no tool call and request missing required information |

The suite includes near-miss examples, such as asking a read-only stock-price tool to sell shares or asking a read-only file tool to write a file.

### Probe scoring

The notebook uses custom rule-based checks:

- **Call pass:** expected calls are present, the call count matches, function names are available, and the arguments satisfy required/allowed schema fields.
- **Strict abstain pass:** no tool call is emitted and the response contains an inability/refusal signal.
- **Ask pass:** no tool call is emitted, the response is phrased as a question/request, and it mentions expected missing-information keywords.
- **Answered without call:** a diagnostic flag when the model gives a direct answer without a call, question, or refusal signal.

These are lightweight diagnostic heuristics, not the official BFCL evaluator. Their keyword-based nature makes them sensitive to phrasing and possible false positives/negatives.

### Saved probe results

`probe_results_v2.json` contains results for four configurations: Qwen3-4B-Instruct-2507 and Qwen3-1.7B, each evaluated with no system prompt and with the careful system prompt.

The notebook's saved summary is:

| Metric | Qwen3-4B no system | Qwen3-4B careful | Qwen3-1.7B no system | Qwen3-1.7B careful |
|---|---:|---:|---:|---:|
| Call pass | 1.00 | 1.00 | 0.87 | 0.93 |
| Abstain: no call | 1.00 | 1.00 | 1.00 | 1.00 |
| Abstain: strict | 1.00 | 1.00 | 0.93 | 1.00 |
| Abstain: near-miss no call | 1.00 | 1.00 | 1.00 | 1.00 |
| Ask pass | 0.80 | 0.80 | 0.33 | 0.33 |
| Ask cases that produced a call | 0.20 | 0.20 | 0.60 | 0.53 |
| Direct answer without call (call/ask cases) | 0.00 | 0.00 | 0.10 | 0.07 |

Values are proportions rounded to two decimal places from this project's 45-case diagnostic suite. **They are not official BFCL scores and should not be compared directly with leaderboard results.**

The probe suggests that missing-argument handling is a weakness for the tested 1.7B configuration: it sometimes inserts plausible defaults or fabricated values instead of asking. The careful instruction improved the 1.7B call and strict-abstain figures in this small test but did not improve ask pass. The 4B configuration performed better on this particular hand-authored suite, but the sample is small and does not establish general benchmark superiority.

## Setup

The experiments were developed primarily in Kaggle notebooks. A GPU-enabled notebook runtime is recommended for model inference.

### Environment used by recorded runs

- Python 3.11 for the BFCL CLI environment.
- CUDA-enabled PyTorch for local GPU inference.
- NVIDIA T4 GPU(s) in the recorded Qwen experiments.
- Internet access to download datasets, model weights, and packages.
- Hugging Face account/token for gated dataset access and higher Hub rate limits.

The notebooks install packages with upgrade flags and do not provide a fully pinned environment lockfile. Versions printed in notebook outputs are snapshots of individual sessions, not a guaranteed reproducible dependency set.

### Recommended: Kaggle

1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and create a notebook.
2. Select a GPU accelerator. The Qwen notebook was documented for T4 x2, although the model is explicitly loaded onto GPU 0.
3. Enable internet access.
4. Add a Kaggle secret called `HF_TOKEN` if authenticated Hugging Face access is required.
5. Clone the repository or upload its notebooks.
6. Run cells in order and inspect outputs before proceeding.

For xLAM, first open [its dataset page](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) and accept the access conditions if prompted.

### Clone locally

```bash
git clone https://github.com/jp9dotcom/bfcl-toolcalling-lora.git
cd bfcl-toolcalling-lora
```

The notebooks can be opened in Jupyter or uploaded to Kaggle. Local execution requires adapting Kaggle-specific paths, secret access, and GPU configuration.

## Running the notebooks

Suggested exploration order:

1. **`qwen-lora.ipynb`** — inspect the 4-bit model load and basic call / abstain / ask inference examples.
2. **`training-data.ipynb`** — run the dataset download, parsing, validation, decontamination, prompt rendering, and JSONL export.
3. **`bfcl-baseline.ipynb`** — install BFCL, start the local model endpoint, and run the baseline generation/scoring cells.
4. **`probe-eval.ipynb`** — run the 45 diagnostic cases and compare model/prompt configurations.
5. **Future fine-tuning stage** — implement and validate a LoRA/QLoRA trainer that consumes the generated JSONL files before claiming adapter-training results.

### BFCL environment commands

The baseline notebook uses the following setup commands in a Linux/Kaggle environment:

```bash
pip install -q uv fastapi uvicorn
uv venv --python 3.11 /kaggle/working/bfclenv
uv pip install --python /kaggle/working/bfclenv/bin/python bfcl-eval==2026.3.23
uv pip install --python /kaggle/working/bfclenv/bin/python soundfile
```

The notebook then creates a BFCL project root, configures the local endpoint environment variables, and starts the FastAPI server. The actual CLI calls follow this pattern once the local server and paths are configured:

```bash
bfcl generate \
  --model Qwen/Qwen3-1.7B-FC \
  --skip-server-setup \
  --test-category simple_python,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance \
  --result-dir <RESULT_DIRECTORY> \
  --num-threads 4

bfcl evaluate \
  --model Qwen/Qwen3-1.7B-FC \
  --test-category simple_python,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance \
  --result-dir <RESULT_DIRECTORY> \
  --score-dir <SCORE_DIRECTORY>
```

Replace placeholders with the directories configured in the notebook. These commands assume that the endpoint is running and BFCL environment variables have been set as shown in the notebook.

For each run, preserve the result directory, score directory, model ID, base-weight ID, BFCL version, generation parameters, and timestamp.

## Interpreting probe results

The JSON object is keyed by model and prompt condition:

```text
Qwen3-4B-Instruct-2507 | no_system
Qwen3-4B-Instruct-2507 | careful
Qwen3-1.7B | no_system
Qwen3-1.7B | careful
```

Each case record includes fields such as:

- `id`: probe case ID.
- `kind`: expected behavior category.
- `n_calls`: number of parsed tool calls.
- `pass`: whether the case passed the notebook's custom checks.
- `raw`: raw model response.
- `near_miss`: whether the case is a near-miss example.

When analyzing failures, inspect the raw response and identify whether the model selected an unrelated tool, omitted a required argument, added an ungrounded argument, invented a default, fabricated a result without a call, or refused a supported request. Aggregate metrics alone can hide these distinct errors.

## Reproducibility and limitations

1. **LoRA training is not implemented in the checked-in Qwen notebook.** There is no verified adapter configuration, trainer, training loop, or adapter-save procedure in the current `qwen-lora.ipynb`.
2. **The data notebook contains multiple pipeline states.** An earlier simple split and a later grouped split have different outputs. Restart the kernel and rerun the intended complete path before using dataset counts in a report.
3. **Recorded dataset outputs differ.** An earlier recorded cell reports only xLAM examples, while a later pipeline version includes ToolACE-derived ask/abstain records. The later grouped-split output is the more developed path.
4. **Probe scores are not BFCL scores.** The 45 cases are hand-authored, and the scoring logic relies partly on keywords and regular expressions.
5. **No final aggregate BFCL score is claimed here.** The repository includes BFCL generation/evaluation code, but a complete verified set of final score artifacts for every configuration was not established during this documentation review.
6. **Dependencies are not fully pinned.** Package upgrades may resolve to different versions over time.
7. **Kaggle paths and secrets are environment-specific.** Local use requires path and secret-management changes.
8. **External licenses remain separate.** The repository's MIT license does not supersede dataset, model-weight, or package terms. Check source terms before redistributing derived artifacts.

## Roadmap

- [ ] Implement an end-to-end LoRA/QLoRA training notebook using the exported JSONL files.
- [ ] Add assertions that the final train/validation splits contain call, ask, and abstain examples.
- [ ] Record exact LoRA rank, alpha, target modules, optimizer, learning rate, epochs, batch size, gradient accumulation, warmup, scheduler, seed, and checkpoint selection.
- [ ] Add a pinned environment file and reproducible run manifest.
- [ ] Evaluate base and fine-tuned models on the same BFCL version, categories, and inference settings.
- [ ] Publish official BFCL overall, live, non-live, and category-level results with error analysis.
- [ ] Expand diagnostic probes with paraphrases, nested arguments, ambiguous requests, malformed schemas, and multi-tool cases.
- [ ] Add stricter argument-grounding checks and execution-based validation.
- [ ] Publish adapter weights and model card only after the training and licensing details are verified.

## References

- [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [BFCL / Gorilla repository](https://github.com/ShishirPatil/gorilla)
- [xLAM Function Calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

## License

This repository is released under the [MIT License](LICENSE).

The repository license applies to this repository's code and documentation. It does not automatically grant rights to redistribute external datasets, model weights, or third-party dependencies. Review their respective licenses and terms before use.
