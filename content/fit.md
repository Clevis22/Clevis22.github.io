---
title: "LLM VRAM & RAM Calculator: Which Model Fits Your Hardware?"
description: "Free LLM VRAM and RAM calculator. Enter your GPU VRAM, Apple Silicon unified memory, or system RAM and find out which small language models fit, at which quantization, with estimated memory use."
layout: "fit"
hidemeta: true
disableShare: true
url: "/fit/"
---

The tool above answers the question every local-LLM search eventually lands on: **will this model fit in my memory?** Pick your hardware and the wizard lists every model from this blog's coverage that fits, with a recommended GGUF quantization and an estimated total footprint. Switch to calculator mode to estimate the VRAM or RAM cost of any model at any quantization and context length, including models we haven't covered (enter the parameter count manually).

## How the VRAM and RAM estimates work

Three things consume memory when you run a model locally, and the math for each is simple enough to do on a napkin. The tool just does it faster.

The model weights dominate. A GGUF file's size is roughly the parameter count times the bits per weight, divided by eight. Q4_K_M works out to about 0.6 GB per billion parameters, which is why a 7B model lands near 4.5 GB and a 3B model near 2 GB. Our [Q4 vs Q5 vs Q8 breakdown](/posts/gguf-quantization-levels-q4-q5-q8/) has real measured file sizes if you want to check the approximation against reality (it holds up well).

The KV cache is the part most people forget. It grows with context length: a standard 7B transformer at 32K context adds several extra gigabytes on top of the weights. Hybrid architectures (marked in the tool) like Gated DeltaNet and Mamba blends grow far slower, which is why a 3B hybrid can honestly advertise a 256K window on a laptop. The [RAM requirements guide](/posts/how-much-ram-for-local-llms/) walks through the full formula.

The runtime adds overhead on top: llama.cpp is fairly lean, Ollama holds roughly an extra gigabyte. The tool folds a conservative allowance into every estimate.

One MoE caveat: mixture-of-experts models are sized by their **total** parameter count, not the active count. A 35B-A3B model decodes at 3B-dense speed but still needs all 35B parameters resident in memory.

## What can I run with 8 GB of RAM?

At Q4_K_M, an 8 GB machine comfortably runs models up to about 4B parameters with a usable context window: Phi-4-mini, SmolLM3-3B, Qwen3.5-4B, and Gemma 3 4B all fit with room to spare. A 7B to 8B model technically loads but leaves so little headroom for context and the OS that you should either step down a size or cap the context hard. The wizard above gives you the exact list for your setup; for GPU owners the same logic applies to VRAM, covered in detail in our [low-VRAM Windows GPU guide](/posts/run-local-llms-low-vram-windows-gpu/).

Apple Silicon is the friendliest case because the unified memory pool serves both CPU and GPU. A 16 GB M-series Mac handles 7B and 8B models well; the specifics are in the [Apple Silicon guide](/posts/run-small-llms-apple-silicon/).

## Notes on accuracy

These are estimates, not measurements. Real memory use varies by runtime, attention architecture, KV cache precision, and OS overhead, typically within 10 to 15 percent of the numbers shown. The model list covers models reviewed on this blog (each result links to its hands-on guide), so it grows as coverage does. For anything not listed, calculator mode with a manual parameter count works for any transformer.
