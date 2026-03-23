---
title: "Fine-Tune Gemma 3 270M on Apple Silicon with MLX-LM and Python"
date: 2026-03-22
draft: false
tags: ["small-models", "slm", "edge-ai", "gemma3", "mlx", "lora", "apple-silicon", "on-device-ai"]
categories: ["small-ai-models"]
description: "A practical, step-by-step guide to fine-tuning Google's Gemma 3 270M on your Mac using Apple's MLX framework, LoRA, and Python."
---

[![M2 MacBook Air Midnight — Apple Silicon fine-tuning rig](https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/M2_Macbook_Air_Midnight_model_-_1.jpg/1200px-M2_Macbook_Air_Midnight_model_-_1.jpg)](https://commons.wikimedia.org/wiki/File:M2_Macbook_Air_Midnight_model_-_1.jpg)
*M2 MacBook Air. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:M2_Macbook_Air_Midnight_model_-_1.jpg) · CC BY-SA 4.0*

Your MacBook is already a fine-tuning machine. You just haven't told it yet.

If you've been staring at cloud GPU bills, waiting in Colab queues, or assuming that model fine-tuning is reserved for people with data centre access — this post is going to change your workflow. Google's Gemma 3 270M is a surprisingly capable small language model, and Apple's MLX framework turns your M-series Mac into a first-class local training environment. Together, they let you go from raw dataset to a domain-specialized model without leaving your desk.

This is a practical walkthrough — no hype, no hand-waving. By the end you'll have a fine-tuned adapter, know how to fuse it into a deployable model, and understand the real trade-offs along the way.

---

## Why Gemma 3 270M Is Worth Your Attention

Before diving into commands, it's worth understanding what you're actually working with.

Gemma 3 270M is Google DeepMind's smallest model in the Gemma 3 family, trained on 6 trillion tokens of web text, code, mathematics, and images. The model is designed for text generation tasks including question answering, summarisation, and reasoning — and its relatively compact size makes it practical for deployment on consumer hardware like laptops and desktops.

The model supports a 32K token context window and has an instruction-tuned (`-it`) checkpoint ready to use out of the box. On benchmarks, the instruction-tuned 270M scores 51.2 on IFEval — noteworthy for a model of this size. For practical on-device tasks — text classification, structured extraction, domain Q&A, style transfer — it punches well above what its parameter count would suggest.

The killer feature for fine-tuning purposes: it's small enough to fit comfortably in the unified memory of any M-series Mac, leaving plenty of headroom for your data pipeline and OS even on an 8GB machine.

[![A coloured neural network diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/400px-Colored_neural_network.svg.png)](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg)
*LoRA fine-tuning inserts small trainable rank-decomposed matrices into a frozen network like this — updating only those, not the full weights. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg) · CC BY-SA 3.0*

---

## The Stack: MLX-LM Explained

Apple's [MLX framework](https://github.com/ml-explore/mlx-lm) is a NumPy-like array framework designed specifically for Apple Silicon's unified memory architecture. Unlike PyTorch or TensorFlow, MLX doesn't shuttle data between CPU and GPU — the CPU, GPU, and Neural Engine all share the same memory pool. For fine-tuning, this matters enormously: you're not paying the memory bandwidth tax of discrete GPU setups.

`mlx-lm` is the high-level package built on top of MLX that handles LLM inference and fine-tuning. It integrates directly with Hugging Face Hub, supports LoRA and QLoRA out of the box, and ships with a clean CLI so you can get a training run going in a single command.

Key things `mlx-lm` supports for fine-tuning:

- **LoRA** (Low-Rank Adaptation) — the default, trains small rank-decomposed matrices instead of full weights
- **DoRA** — a variation of LoRA, also supported via `--fine-tune-type dora`
- **Full fine-tuning** — trains all weights, more memory-hungry but can produce more dramatic style shifts
- **QLoRA** — triggered automatically when you point `--model` at a quantized checkpoint
- Logging to Weights & Biases (`--report-to wandb`) or SwanLab (`--report-to swanlab`)
- Resumable training with `--resume-adapter-file`
- Prompt masking so the loss is computed only on completions, not prompts

The `mlx-lm` repository also has an official Jupyter notebook by core contributor Awni Hannun ([@awni](https://gist.github.com/awni/773e2a12079da40a1cbc566686c84c8f)) that demonstrates the Python API for LoRA fine-tuning end-to-end — worth bookmarking as a reference even if you prefer the CLI path.

For those who prefer a more opinionated, higher-level interface, [mlx-tune](https://github.com/ARahim3/mlx-tune) is a community project that aims to bring an Unsloth-like experience to Mac users through MLX, abstracting away some of the configuration boilerplate.

[![A 2021 14-inch MacBook Pro with M1 Pro chip](https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/A_2021_14-inch_Silver_MacBook_Pro.jpg/1200px-A_2021_14-inch_Silver_MacBook_Pro.jpg)](https://commons.wikimedia.org/wiki/File:A_2021_14-inch_Silver_MacBook_Pro.jpg)
*A 2021 14-inch MacBook Pro with M1 Pro. The unified memory architecture shared by CPU, GPU, and Neural Engine is why MLX can train without expensive CPU↔GPU data transfers. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:A_2021_14-inch_Silver_MacBook_Pro.jpg) · CC BY-SA 4.0*

---

## Prerequisites

- Apple Silicon Mac (M1 or later). M2/M3/M4 with 16GB+ unified memory is comfortable; 8GB works for 270M with care.
- Python 3.11 (3.12 works but 3.11 is best-tested with current MLX builds)
- A [Hugging Face account](https://huggingface.co) with an access token — Gemma 3 requires agreeing to Google's license before you can download weights
- `pip` or `conda`

[![Python logo](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png)](https://commons.wikimedia.org/wiki/File:Python-logo-notext.svg)
*Python 3.11 — the recommended runtime for mlx-lm. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Python-logo-notext.svg) · PSF License*

```bash
# Install the training-enabled build
pip install "mlx-lm[train]"

# Authenticate with Hugging Face
huggingface-cli login
```

---

## Step 1: Choose Your Model Checkpoint

The MLX Community on Hugging Face maintains pre-converted MLX-format checkpoints for Gemma 3 270M. You have a few options depending on your RAM and goals:

| Checkpoint | Precision | Notes |
|---|---|---|
| `mlx-community/gemma-3-270m-it-4bit` | INT4 | Smallest, fastest, ~42M effective params loaded |
| `mlx-community/gemma-3-270m-it-bf16` | BF16 | Full precision, best for full fine-tuning |
| `google/gemma-3-270m-it` | BF16 (HF native) | MLX-LM will auto-convert on first load |

For LoRA fine-tuning on a base MacBook Air, `mlx-community/gemma-3-270m-it-bf16` is the recommended starting point — full precision but small enough to fit comfortably. Pointing MLX-LM at a quantized checkpoint automatically switches the training mode to QLoRA.

---

## Step 2: Prepare Your Dataset

MLX-LM's LoRA trainer expects JSONL files in a specific format, placed in a directory with at least a `train.jsonl`. An optional `valid.jsonl` enables periodic validation loss reporting during training. For evaluation, provide `test.jsonl`.

The supported data formats are `chat`, `completions`, `tools`, and raw `text`. For most fine-tuning tasks, `chat` format is the right choice:

```json
{"messages": [{"role": "system", "content": "You are a concise SQL assistant."}, {"role": "user", "content": "List all customers in New York."}, {"role": "assistant", "content": "SELECT * FROM customers WHERE city = 'New York';"}]}
```

Save these as `dataset/train.jsonl` (and optionally `dataset/valid.jsonl`).

A practical tip: use `--mask-prompt` during training if you only want loss computed on the assistant's response, not the system/user turns. This is supported for `chat` and `completion` datasets and usually produces cleaner fine-tuning results for instruction-following tasks.

---

## Step 3: LoRA Fine-Tuning via CLI

The main command is `mlx_lm.lora`. Here's a solid baseline invocation:

```bash
mlx_lm.lora \
    --model mlx-community/gemma-3-270m-it-bf16 \
    --train \
    --data dataset/ \
    --iters 600 \
    --adapter-path adapters/my-task \
    --mask-prompt
```

What's happening here:

- `--model` points to the pre-converted MLX checkpoint on Hugging Face Hub (downloaded and cached locally on first run)
- `--train` triggers training mode
- `--data` points to the folder with your JSONL files
- `--iters 600` runs 600 gradient steps — a reasonable starting point for small datasets
- `--adapter-path` specifies where to save the trained LoRA adapter weights (defaults to `adapters/`)
- `--mask-prompt` ignores loss on the prompt tokens

By default, the fine-tuning type is LoRA. To use DoRA or full fine-tuning:

```bash
# Full fine-tuning (trains all layers)
mlx_lm.lora \
    --model mlx-community/gemma-3-270m-it-bf16 \
    --train \
    --data dataset/ \
    --iters 100 \
    --fine-tune-type full \
    --num-layers -1 \
    --adapter-path adapters/my-task-full
```

The `--num-layers -1` flag tells MLX-LM to apply fine-tuning to all transformer layers. This is more memory-intensive but can produce more dramatic style adaptation — particularly for dialect, strong output-format conditioning, or persona tasks where full training consistently outperforms LoRA at this model scale.

For machines with limited RAM, add `--batch-size 1` and `--grad-checkpoint` to reduce memory usage at the cost of some speed.

---

## Step 4: Running Inference with Your Adapter

Once training completes, you can test the adapter immediately without fusing:

```bash
mlx_lm.generate \
    --model mlx-community/gemma-3-270m-it-bf16 \
    --adapter-path adapters/my-task \
    --prompt "List all customers in New York."
```

Or use the Python API for programmatic access:

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "mlx-community/gemma-3-270m-it-bf16",
    adapter_path="adapters/my-task"
)

messages = [{"role": "user", "content": "List all customers in New York."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

---

## Step 5: Evaluating Performance

To measure test-set perplexity against a held-out `test.jsonl`:

```bash
mlx_lm.lora \
    --model mlx-community/gemma-3-270m-it-bf16 \
    --adapter-path adapters/my-task \
    --data dataset/ \
    --test
```

A lower perplexity means the model assigns higher probability to the correct completions in your test set. For domain fine-tuning, comparing the base model's perplexity to the adapter's is the quickest sanity check that training did something useful.

---

## Step 6: Fusing and Sharing the Model

Adapters are lightweight (a few MB of safetensors files), but for deployment you'll often want a single fused model. MLX-LM provides `mlx_lm.fuse` for this:

```bash
mlx_lm.fuse \
    --model mlx-community/gemma-3-270m-it-bf16 \
    --adapter-path adapters/my-task \
    --save-path fused-model/
```

The fused model lands in `fused-model/` ready to use like any other MLX checkpoint. You can also upload it directly to Hugging Face:

```bash
mlx_lm.fuse \
    --model mlx-community/gemma-3-270m-it-bf16 \
    --adapter-path adapters/my-task \
    --upload-repo your-hf-username/gemma-3-270m-my-task \
    --hf-path mlx-community/gemma-3-270m-it-bf16
```

For maximum portability, `mlx_lm.fuse` also supports GGUF export — though note GGUF export is currently limited to Llama, Mistral, and Mixtral-style architectures, so this path isn't available for Gemma at the moment.

---

## A Note on QLoRA for Tighter Hardware

If you're working on an 8GB machine or want to minimize the memory footprint during training, point `--model` at the 4-bit quantized checkpoint instead:

```bash
mlx_lm.lora \
    --model mlx-community/gemma-3-270m-it-4bit \
    --train \
    --data dataset/ \
    --iters 600 \
    --adapter-path adapters/my-task-qlora
```

When MLX-LM detects a quantized model, it automatically runs QLoRA — updating only the LoRA adapter weights in float, while keeping the base model quantized. This is the most memory-efficient path and works well for 270M.

---

## Practical Tips and Gotchas

**Dataset size matters more than iterations.** 600 iterations on a 50-example dataset will overfit. 600 iterations on 2,000+ examples will generalize better. Watch the validation loss curve if you've set up `valid.jsonl`.

**Prompt masking is almost always the right call.** Unless you explicitly want the model to learn to reproduce system prompts, use `--mask-prompt`. It focuses the gradient signal on the actual completions.

**The `tokenizer` argument gotcha.** If you're using the Python API via the `train()` function from `mlx_lm.tuner.trainer` directly, note that recent versions removed `tokenizer` as a direct argument. Use `CacheDataset` wrappers for your train and validation datasets — this is a known upstream API change documented in community gist comments.

**Full fine-tuning vs LoRA.** For 270M, full fine-tuning is feasible on most modern Apple Silicon Macs since the model is small. If you need strong style adaptation (persona, dialect, output format), full fine-tuning with `--fine-tune-type full` tends to produce more consistent results than LoRA at this scale.

**Resume broken runs.** Long training sessions can be interrupted. Resume with `--resume-adapter-file adapters/my-task/adapters.safetensors`.

---

## Why This Matters in 2026

The combination of sub-300M models and on-device training frameworks continues a clear trend: the floor for "useful AI" keeps dropping in both compute cost and model size. Gemma 3 270M trained on 6 trillion tokens, achieving IFEval scores competitive with much larger models from two years ago, is a direct result of this efficiency research compounding.

For indie hackers and tinkerers, this means a genuinely useful domain model — one that knows your product's terminology, your data's schema, or your users' dialect — now costs less than an afternoon and runs locally with no API bill. The MLX-LM + Apple Silicon stack makes that loop tight enough to actually iterate on.

Fine-tuning isn't magic, and 270M parameters won't replace reasoning-heavy tasks. But for well-defined, narrow jobs like classification, extraction, formatting, or constrained generation, a specialized 270M model often outperforms prompting a much larger general model — and does it faster with zero latency.

---

## Sources

- [ml-explore/mlx-lm — GitHub](https://github.com/ml-explore/mlx-lm)
- [mlx-lm LoRA Fine-Tuning Documentation (LORA.md)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX LM LoRA Fine Tune notebook by @awni — GitHub Gist](https://gist.github.com/awni/773e2a12079da40a1cbc566686c84c8f)
- [ARahim3/mlx-tune — GitHub](https://github.com/ARahim3/mlx-tune)
- [Fine-Tuning LLMs Locally Using MLX-LM — DZone](https://dzone.com/articles/fine-tuning-llms-locally-using-mlx-lm-guide)
- [google/gemma-3-270m — Hugging Face Model Card](https://huggingface.co/google/gemma-3-270m)
- [mlx-community/gemma-3-270m-it-4bit — Hugging Face](https://huggingface.co/mlx-community/gemma-3-270m-it-4bit)