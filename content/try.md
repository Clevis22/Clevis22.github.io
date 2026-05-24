---
title: "Try a Small LLM in Your Browser"
description: "Run Qwen2.5-0.5B locally in your browser via WebGPU. No backend, no API key, no install — a live demo of a small language model executing entirely on your hardware."
layout: "try"
hidemeta: true
disableShare: true
url: "/try/"
---

This page is the thesis of the whole blog made tangible. A 500-million-parameter chat model is being downloaded to your browser, compiled to GPU shaders, and run on your own hardware. There is no server doing the inference. Your prompts never leave this tab.

## What's actually running

The model is **Qwen2.5-0.5B-Instruct**, released by Alibaba's Qwen team in September 2024 under Apache 2.0. At 4-bit quantization (the `q4f16_1` MLC build) the weights are roughly 280 MB. That's the entire model: vocabulary, attention layers, MLP weights, everything. Small enough to fit in a single CDN download and live comfortably inside a browser tab.

It is not a smart model. A 0.5B parameter chat model is not going to beat foundation models at anything. The point is not capability; the point is that *this works at all* — that you can have a conversation with a real instruction-tuned LLM with zero install and zero network calls after the initial weight download.

## How it works

The stack is short:

1. **WebLLM** — MLC AI's in-browser inference runtime. It takes a quantized model in MLC format and runs it through WebGPU, the modern browser API that exposes GPU compute to JavaScript.
2. **WebGPU** — the actual GPU API. Available in Chrome/Edge 113+, Safari 18+, and Firefox with a flag. Falls back to WebAssembly + CPU on unsupported browsers, but it's much slower there.
3. **IndexedDB** — once the weights download, they're cached locally. Refresh this page and the model loads instantly the second time.
4. **The model itself** — Qwen2.5-0.5B-Instruct, pre-compiled by the MLC team into a format WebLLM can stream into GPU memory.

When you hit "send", your prompt goes into the model's tokenizer, the resulting tokens flow through the attention layers on your GPU, and tokens stream back into the output panel — exactly the same loop that runs on a server, just running on the silicon already in front of you.

## Why this matters

Most "AI demos" you see on the web are a thin chat UI wrapped around a hosted API. They cost the operator money per call, they require a network round-trip per token, and they collect every prompt for training.

A model that runs in your browser inverts all of that. It costs the host nothing. It works on a plane. It works in a hospital. It works after the API provider shuts down the endpoint or 10x's the price. The trade-off is capability — a 0.5B model is dramatically less smart than a frontier model — but it's also the trade-off the entire small-models movement is built on. The bet of this blog is that "less smart but runs anywhere" wins for more workloads than people currently realise.

## Limitations of this specific demo

- **First load takes 30–90 seconds** depending on your connection. After that, the model is cached.
- **Needs WebGPU.** On Chrome/Edge 113+, Safari 18+, or Firefox with `dom.webgpu.enabled` set in `about:config`. On unsupported browsers the demo will tell you so.
- **Mobile is hit-or-miss.** Some phones have WebGPU; many don't have enough free memory for the weights.
- **A 0.5B model hallucinates a lot.** Treat outputs as a tech demo, not a reference.

If you want to actually use a small model for real work, the rest of [the blog](/) covers how to run bigger ones — 3B, 4B, 7B — locally via Ollama, llama.cpp, and LM Studio.
