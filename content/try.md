---
title: "Try a Small LLM in Your Browser"
description: "Run a small open language model (Qwen, SmolLM, or Llama) locally in your browser via WebGPU. No backend, no API key, no install. A live demo of an SLM executing entirely on your hardware."
layout: "try"
hidemeta: true
disableShare: true
url: "/try/"
---

This page is the thesis of the whole blog made tangible. A small open language model is being downloaded to your browser, compiled to GPU shaders, and run on your own hardware. There is no server doing the inference. Your prompts never leave this tab.

## What's actually running

You get to pick from four open instruction-tuned models. All four are 4-bit quantized, and each one downloads once then caches in your browser's IndexedDB for instant reload on subsequent visits.

| Model | Params | Download | Notes |
|---|---|---|---|
| SmolLM2-360M-Instruct | 360M | ~200 MB | HuggingFace's tiniest. Fast, often incoherent. Default on phones. |
| Qwen2.5-0.5B-Instruct | 500M | ~280 MB | Default on desktop. Best balance of size and quality at this scale. |
| Llama-3.2-1B-Instruct | 1B | ~670 MB | Meta's smallest Llama 3.2. Noticeably smarter. |
| Qwen2.5-1.5B-Instruct | 1.5B | ~840 MB | Best quality on this page; biggest download. |

Picking from the dropdown above the chat tells the demo which model to load. Typing a message and hitting send before any model is loaded triggers a load of whichever one is currently selected; your message queues and auto-sends as soon as the model is ready. To switch models later, just change the dropdown. The current engine unloads, the new one streams in, and your conversation carries over, since the full history is replayed to whichever model answers next.

Conversations live in the sidebar and are saved to your browser's localStorage, so they survive a page reload but never leave your machine. The `$ cfg` button exposes the system prompt, temperature, and max-tokens settings if you want to poke at generation behaviour directly.

None of these are smart models. A sub-2B parameter chat model is not going to beat foundation models at anything. The point is not capability; the point is that *this works at all*, that you can have a conversation with a real instruction-tuned LLM with zero install and zero network calls after the initial weight download.

## How it works

The stack is short. **WebLLM** is MLC AI's in-browser inference runtime; it takes a quantized model in MLC format and runs it through WebGPU, the browser API that exposes GPU compute to JavaScript. **WebGPU** itself is the GPU API, available in Chrome/Edge 113+, Safari 18+, and Firefox behind a flag. **IndexedDB** is the browser's local key-value store; weights live there after the first download, which is why a second visit to the same model is instant.

When you hit "send", your prompt goes into the model's tokenizer, the resulting tokens flow through the attention layers on your GPU, and tokens stream back into the output panel. Exactly the same loop that runs on a server, just running on the silicon already in front of you.

## Why this matters

Most "AI demos" you see on the web are a thin chat UI wrapped around a hosted API. They cost the operator money per call, they require a network round-trip per token, and they collect every prompt for training.

A model that runs in your browser inverts all of that. It costs the host nothing. It works on a plane. It works in a hospital. It works after the API provider shuts down the endpoint or 10x's the price. The trade-off is capability; a sub-2B model is dramatically less smart than a frontier model. That's also the trade-off the entire small-models movement is built on. The bet of this blog is that "less smart but runs anywhere" wins for more workloads than people currently realise.

## Limitations of this specific demo

- First load takes 15–90 seconds depending on your connection and which model you picked; after that, the chosen model is cached.
- WebGPU is required. Works in Chrome/Edge 113+, Safari 18+, or Firefox with `dom.webgpu.enabled` set in `about:config`. On unsupported browsers the demo will say so.
- Mobile is hit-or-miss. Some phones have WebGPU; many don't have enough free memory for the larger models, especially the 1B and 1.5B options.
- Sub-2B models hallucinate a lot. Treat outputs as a tech demo, not a reference.

If you want to actually use a small model for real work, the rest of [the blog](/) covers how to run bigger ones (3B, 4B, 7B) locally via Ollama, llama.cpp, and LM Studio.
