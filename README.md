# bfcl-toolcalling-lora

LoRA fine-tuning of **Qwen3-4B-Instruct-2507** to improve function / tool calling, evaluated on the **Berkeley Function Calling Leaderboard (BFCL)**. Everything runs on a free Kaggle GPU.

> **Status:** research / learning project. The goal is to understand the full loop (baseline → data → fine-tune → eval) and report what actually happened, good or bad.

---

## Table of Contents

1. [Overview](#overview)
2. [Why this project](#why-this-project)
3. [Repository Structure](#repository-structure)
4. [Pipeline](#pipeline)
5. [Setup](#setup)
6. [How to Run](#how-to-run)
7. [Results](#results)
8. [Methodology Notes](#methodology-notes)
9. [Limitations](#limitations)
10. [Roadmap](#roadmap)
11. [Acknowledgements](#acknowledgements)
12. [License](#license)

---

## Overview

Tool calling is the skill that lets an LLM turn a request like *"What's the weather in Pune tomorrow?"* into a structured call like `get_weather(city="Pune", date="tomorrow")`. Small open models are often weak at it: wrong argument names, wrong types, hallucinated functions, or calling a tool when it shouldn't.

This repo tries to close part of that gap cheaply:

- Take a small open model that BFCL supports out of the box.
- Measure how it does **before** any training (baseline).
- Build a training set of function-calling examples.
- Fine-tune with **LoRA** (parameter-efficient, fits on Kaggle).
- Re-run the same evaluation and compare.

| Item | Choice |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (BFCL self-hosted function-calling entry) |
| Method | LoRA (adapter-only fine-tuning) |
| Benchmark | Berkeley Function Calling Leaderboard (BFCL) |
| Compute | Kaggle GPU notebooks |
| License | MIT |

> Qwen2.5 isn't in BFCL's supported-model list, which is why this project uses Qwen3-4B-Instruct-2507 instead.

---

## Why this project

- **Practical skill:** tool calling is the backbone of agents (MCP, LangGraph, function-calling APIs).
- **Measurable:** BFCL gives a standard, comparable score instead of vibes.
- **Cheap:** LoRA + a 4B model keeps the whole thing within free-tier GPU limits.
- **Honest:** baseline and post-training numbers use the same evaluation, so the delta means something.

---

## Repository Structure

```
bfcl-toolcalling-lora/
├── bfcl-baseline.ipynb      # Load base model and run BFCL baseline evaluation
├── bfcl-baseline-2          # Second baseline run / outputs
├── training-data.ipynb      # Builds the fine-tuning dataset
├── training-data            # Generated training data
├── qwen-lora.ipynb          # LoRA fine-tuning of Qwen3-4B-Instruct-2507
├── probe-eval.ipynb         # Quick probe evaluation of the model
├── probe_results_v2.json    # Saved probe results (v2)
├── LICENSE                  # MIT
└── README.md
```

| File | Purpose |
|---|---|
| `bfcl-baseline.ipynb` | Loads the base model on Kaggle and gets the pre-training BFCL numbers. |
| `bfcl-baseline-2` | Follow-up baseline run (re-run / refined setup). |
| `training-data.ipynb` | Assembles and formats function-calling examples into the training set. |
| `training-data` | The produced dataset artifacts. |
| `qwen-lora.ipynb` | LoRA training notebook: model load, adapter config, training loop, adapter save. |
| `probe-eval.ipynb` | Lightweight eval to check behaviour on targeted probes without a full BFCL run. |
| `probe_results_v2.json` | Output of the probe evaluation. |

---

## Pipeline

```
 ┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌─────────────┐
 │  Baseline    │ →  │ Training data │ →  │  LoRA train  │ →  │  Evaluate   │
 │  (BFCL)      │    │ construction  │    │  (Kaggle)    │    │ BFCL + probe│
 └──────────────┘    └───────────────┘    └──────────────┘    └─────────────┘
```

1. **Baseline** – run the untouched model through BFCL so there's a reference score.
2. **Training data** – build function-calling examples (prompt + function schemas → correct call) in the model's chat/tool format.
3. **LoRA training** – train low-rank adapters on top of the frozen base model.
4. **Evaluation** – re-run BFCL and the probe set with the adapter and compare to baseline.

---

## Setup

### Requirements

- A Kaggle account with GPU enabled (Settings → Accelerator → GPU)
- Python 3.10+
- Typical stack for this kind of project: `transformers`, `peft`, `datasets`, `accelerate`, `trl`, `bitsandbytes` (if using quantized loading), and the BFCL evaluation package (`bfcl-eval`)

> Check the first cell of each notebook for the exact packages and versions used. Pin them in a `requirements.txt` if you want reproducible runs.

### Clone

```bash
git clone https://github.com/jp9dotcom/bfcl-toolcalling-lora.git
cd bfcl-toolcalling-lora
```

---

## How to Run

Run the notebooks in this order, on Kaggle with a GPU accelerator:

1. **`bfcl-baseline.ipynb`** – loads `Qwen/Qwen3-4B-Instruct-2507` and records baseline BFCL results.
2. **`training-data.ipynb`** – builds the training set (outputs go to `training-data`).
3. **`qwen-lora.ipynb`** – trains the LoRA adapter and saves it.
4. **`probe-eval.ipynb`** – evaluates the tuned model on probes; writes `probe_results_v2.json`.
5. Re-run the BFCL evaluation with the adapter loaded and compare against step 1.

Tips for Kaggle:

- Save adapters and results to `/kaggle/working/` and download them (or save as a Kaggle Dataset) before the session ends.
- Keep batch size and max sequence length modest; a 4B model plus long tool schemas can run out of memory fast.

---

## Results

> **TODO:** fill this in with your real numbers. Don't leave placeholders in the final version.

| Category | Baseline | LoRA | Δ |
|---|---|---|---|
| Simple | TBD | TBD | TBD |
| Multiple | TBD | TBD | TBD |
| Parallel | TBD | TBD | TBD |
| Parallel multiple | TBD | TBD | TBD |
| Irrelevance detection | TBD | TBD | TBD |
| **Overall** | TBD | TBD | TBD |

Probe results live in `probe_results_v2.json`.

Suggested things to report honestly:

- Where fine-tuning helped, where it did nothing, and where it **hurt** (regressions, e.g. on irrelevance detection, are common after tool-calling SFT).
- Whether gains hold on categories not represented in the training data.

---

## Methodology Notes

- **Same eval before and after.** The baseline and fine-tuned runs use the same BFCL setup so the comparison is fair.
- **LoRA, not full fine-tuning.** Only small adapter matrices are trained; the base weights stay frozen. This keeps memory and time low and makes the adapter small and shareable.
- **Training-data hygiene.** If any training examples overlap with BFCL test prompts, scores will be inflated. Keep the training source separate from BFCL's test set and note which sources you used.
- **Probe eval is not BFCL.** `probe-eval.ipynb` is a fast sanity check; BFCL remains the headline metric.

---

## Limitations

- Single base model, single training recipe: no hyperparameter sweep yet.
- Free-tier GPU limits restrict context length, batch size, and run time.
- Gains on BFCL may not transfer to real-world tool use with messy schemas.
- Results can vary run to run; report seeds and the number of runs if you claim a delta.

---

## Roadmap

- [ ] Add `requirements.txt` with pinned versions
- [ ] Add final BFCL results table and error analysis
- [ ] Ablations: LoRA rank / alpha, learning rate, dataset size
- [ ] Publish the adapter on Hugging Face
- [ ] Multi-turn and agentic evaluation
- [ ] Compare against another base model

---

## Acknowledgements

- [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) – benchmark and evaluation harness
- [Qwen team](https://huggingface.co/Qwen) – base model
- [Hugging Face PEFT](https://github.com/huggingface/peft) – LoRA implementation
- Kaggle – free GPU compute

---

## License

Released under the [MIT License](LICENSE). © 2026 jp9dotcom
