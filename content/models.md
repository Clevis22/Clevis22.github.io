---
title: "Small Language Models Compared: Params, Context, License & RAM"
description: "A sortable, filterable comparison of every small language model tested on this blog: parameter counts, context windows, licenses, capabilities, and estimated Q4 GGUF size."
layout: "models"
hidemeta: true
disableShare: true
url: "/models/"
---

This is a running comparison of every small language model covered on this blog, in one sortable table. Each row is a model we have actually run and written up, so the parameter counts, context windows, and licenses come from hands-on posts rather than a scraped spec sheet. Sort by any column, filter by use case or capability, or search by name. Every model links to its full guide.

The table is built from the same data file that powers the [hardware-fit calculator](/fit/), so the two stay in sync. Use this page to browse and compare small LLMs; use `/fit` when you want to know exactly what runs in your memory budget.

## What counts as a small language model here?

Everything on this blog tops out around 35B dense parameters, plus a few mixture-of-experts models whose active parameter count stays small even when the total is larger. The `params` column shows total parameters, which is what has to fit in memory. For MoE models a second figure shows the active count, which is what sets decode speed. A 35B-A3B model needs room for all 35B but runs at roughly 3B-dense speed.

## Which small language models support vision?

Filter the table with the `vision` chip to see the multimodal picks. At the small end, Qwen3.5-4B and Gemma 3 4B both take image input while staying under 5 GB at Q4. MedGemma 1.5 is a vision specialist for medical imaging rather than a general chat model, and Reka Edge focuses on image, video, and object detection. For a full walkthrough of running one locally, see our [Reka Edge guide](/posts/run-reka-edge-locally-vision-model-mac/).

## How much RAM do I need to run these models?

The `~Q4` column gives the weights-only GGUF size, which is a good first approximation: a 3B model lands near 2 GB, a 7B near 4.5 GB. Real memory use adds the KV cache (which grows with context length) and about a gigabyte of runtime overhead on top. Hybrid-attention models, marked with the `hybrid` badge, keep the KV cache small even at long context. For a precise per-machine estimate, the [fit calculator](/fit/) does the full math, and the [RAM requirements guide](/posts/how-much-ram-for-local-llms/) explains where each number comes from.

## What is the best small language model to start with?

If you are new to running models locally, start with a well-supported 3B to 4B model and step up only if quality falls short. Our [best small language models of 2026](/posts/best-small-language-models-2026/) pillar ranks the current picks by task, and you can try a sub-1B model in your browser right now on the [/try demo](/try/) without installing anything. The table below is the reference; those two pages are the opinionated starting points.
