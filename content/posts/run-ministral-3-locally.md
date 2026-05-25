---
title: "How to Run Ministral 3 Locally: Mistral's 3B, 8B, and 14B Vision Models"
date: 2026-05-25
draft: false
tags: ["small-models", "slm", "edge-ai", "mistral", "ministral", "local-inference", "vision"]
categories: ["small-ai-models"]
description: "A practical guide to running Ministral-3 (3B, 8B, 14B) locally with Ollama and LM Studio: compact models with vision, tool use, and 256K context in under 10GB."
slug: "run-ministral-3-locally"
---

Ministral 3 is Mistral AI's compact model family for local and edge deployment, released in December 2025 under Apache 2.0. It covers three sizes: 3B, 8B, and 14B parameters. Every variant ships with a 0.4B vision encoder and native tool use; all share a 256K context window. This guide covers how to run Ministral 3 locally using Ollama and LM Studio, how to pick the right size for your hardware, and what the benchmark numbers actually tell you.

{{< figure src="https://images.pexels.com/photos/34804005/pexels-photo-34804005.jpeg" alt="Laptop screen showing a code and data analysis interface in a dark environment" caption="The Ministral 3 3B runs on a laptop CPU with under 4 GB of RAM for the model, no discrete GPU required. (Photo: Daniil Komov, Pexels)" >}}

## The Ministral 3 model family

Each Ministral 3 model uses the same core architecture scaled up across three sizes. The LM (language model) component is the main reasoning network; a shared 0.4B vision encoder bolts onto every variant.

| Model | LM params | Total (incl. vision) | Ollama size | Min RAM |
|---|---|---|---|---|
| Ministral-3-3B-Instruct-2512 | 3.4B | 3.8B | 3.0 GB | ~6 GB |
| Ministral-3-8B-Instruct-2512 | 8.4B | 8.8B | 6.0 GB | ~10 GB |
| Ministral-3-14B-Instruct-2512 | 13.5B | 13.9B | 9.1 GB | ~14 GB |

Each size ships in three variants: base (for fine-tuning), instruct (chat-ready), and reasoning (chain-of-thought enabled by default). All three sizes support vision, multilingual output across 11 languages, and function calling.

The "-2512" suffix in the model names is Mistral's convention for December 2025 training checkpoints.

## Benchmarks

Mistral publishes two separate result tables for Ministral 3: one for instruct variants and one for reasoning variants. The distinction matters. The AIME and GPQA scores below come from the reasoning variants; MATH Maj@1 is from the instruct variants; MMLU is from the base.

| Benchmark | Ministral 3B | Ministral 8B | Ministral 14B | Variant |
|---|---|---|---|---|
| AIME 2025 | 72.1% | 78.7% | 85.0% | Reasoning |
| AIME 2024 | 77.5% | 86.0% | 89.8% | Reasoning |
| GPQA Diamond | 53.4% | 66.8% | 71.2% | Reasoning |
| LiveCodeBench | 54.8% | 61.6% | 64.6% | Reasoning |
| MATH Maj@1 | 83.0% | 87.6% | 90.4% | Instruct |
| MMLU 5-shot | — | 76.1% | — | Base |

The 8B reasoning variant's 66.8% on GPQA Diamond (a graduate-level science and engineering benchmark) puts it ahead of several models from the previous generation that are two to three times larger. LiveCodeBench scores are notable because the benchmark is constructed from problems released after training cutoff, making it harder to inflate with memorized answers; 61.6% at 8B is genuinely competitive.

For how these numbers sit relative to other small models available in 2026, see our [best small language models comparison](/posts/best-small-language-models-2026/).

## How to run Ministral 3 locally with Ollama

Ollama is the quickest path. It handles GGUF downloads automatically, applies Q4 quantization by default, and exposes an OpenAI-compatible local API at `http://localhost:11434`. If you haven't installed Ollama yet, our [Ollama vs LM Studio vs llama.cpp guide](/posts/ollama-vs-lm-studio-vs-llama-cpp/) covers setup and a runtime comparison.

Pull and run the 8B instruct variant (the best starting point for most hardware):

```bash
ollama run ministral-3:8b
```

For the 3B on a machine with 8 GB of total RAM or no discrete GPU:

```bash
ollama run ministral-3:3b
```

For the 14B on a system with 16 GB+ of unified memory or 12 GB+ of VRAM:

```bash
ollama run ministral-3:14b
```

On an M-series Mac or a Ryzen AI laptop, the 8B instruct variant runs at roughly 20–30 tokens per second on CPU alone. With a dedicated GPU, expect 80–120 tokens per second on the 8B. The 3B runs faster still on CPU-only setups: 40–60 tokens per second on a modern laptop is typical.

Ollama currently carries only the instruct variants. For specific quantizations, you can pull them directly:

```bash
# Standard Q4_K_M (default)
ollama pull ministral-3:8b-instruct-2512-q4_K_M

# Higher precision Q8
ollama pull ministral-3:8b-instruct-2512-q8_0
```

{{< figure src="https://images.pexels.com/photos/3888151/pexels-photo-3888151.jpeg" alt="A laptop with code visible on screen, open on a wooden desk" caption="The Ministral 8B instruct variant fits on any 16 GB machine and handles vision, tool use, and multilingual output. (Photo: Rodrigo Santos, Pexels)" >}}

## How to run Ministral 3 with LM Studio

LM Studio's model browser makes it easy to compare sizes and switch between instruct and reasoning variants without a terminal. Open the Discover tab and search for "ministral-3". The three sizes download as GGUF files: approximately 2.0 GB (3B), 6.5 GB (8B), and 9.5 GB (14B). The instruct and reasoning variants appear as separate entries in the search results.

LM Studio also exposes a local API on port 1234 using the OpenAI-compatible format, which means you can plug Ministral 3 into most existing tools without code changes.

LM Studio is also where you can access the reasoning variants. They appear as separate model entries (for example, "Ministral 3 8B Reasoning" alongside "Ministral 3 8B") and download at the same file sizes as the instruct versions. The reasoning variant outputs its chain-of-thought in `<think>...</think>` tags before the final answer; you can filter these in application code if you only want the result.

## Vision

Every Ministral 3 variant includes the 0.4B vision encoder. In Ollama, passing an image alongside your prompt looks like this:

```bash
ollama run ministral-3:8b "Describe what's in this image" /path/to/image.jpg
```

The model handles scene description and object identification. It can also reason about relationships between visual elements: it will describe the layout of a UI screenshot or explain what a chart is communicating, not just list what it sees.

For structured visual content (charts, tables, scanned documents), the 14B reasoning variant is more reliable. It produces more precise descriptions and is better at reasoning across multiple elements in a single image.

Mistral's model card recommends keeping image aspect ratios close to 1:1. Very wide or tall images should be cropped before being passed to the model.

## Which Ministral 3 model should you use?

**What's the best Ministral 3 model for most people?**

The 8B instruct variant (`ministral-3:8b`). At 6 GB downloaded and roughly 10 GB of RAM to run, it fits on any 16 GB machine. It handles conversation, document Q&A, function calling, code generation, and basic vision without needing the reasoning overhead. Its 76.1% MMLU and 87.6% MATH scores come from the instruct variant without extended thinking enabled.

**When does the 3B make sense?**

When the 8B doesn't fit: 8 GB total RAM machines, a Raspberry Pi 5, or any setup where download size is a practical constraint. The 3B's reasoning variant still hits 72.1% on AIME 2025 and 83.0% MATH, which is strong for a sub-4B model. The 3.0 GB Ollama download is fast on a slow connection.

**Is the 14B worth the extra memory?**

If you have 16 GB+ of RAM or a GPU with 12 GB+ of VRAM, the 14B earns its keep on hard reasoning tasks. The reasoning variant reaches 85.0% on AIME 2025 and 71.2% on GPQA Diamond, which is a meaningful step up from the 8B on problems that require careful multi-step logic. For everyday chat and quick lookups, the gap between 8B and 14B is small enough that most users won't notice.

## Quantization

Ollama's default Q4_K_M is the right starting point for most hardware. If you want to explore the tradeoffs between Q4, Q5, and Q8 in more detail, our [GGUF vs ONNX vs MLX guide](/posts/gguf-vs-onnx-vs-mlx/) covers how quantization levels affect output quality and RAM usage. GGUF files for Ministral 3 at higher quantizations (Q5_K_M, Q8_0) are available on HuggingFace from the `mistralai` organization if you have the RAM headroom.

## Sources

- [Introducing Mistral 3 (Mistral AI)](https://mistral.ai/news/mistral-3)
- [Ministral-3-3B-Instruct-2512 (HuggingFace)](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)
- [Ministral-3-8B-Instruct-2512 (HuggingFace)](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- [Ministral-3-14B-Instruct-2512 (HuggingFace)](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512)
- [Ministral 3 technical report, arXiv:2601.08584](https://arxiv.org/abs/2601.08584)
- [ministral-3 on Ollama Library](https://ollama.com/library/ministral-3)
- [Ministral 3 on LM Studio](https://lmstudio.ai/models/ministral)
