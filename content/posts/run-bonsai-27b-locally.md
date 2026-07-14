---
title: "Run Bonsai 27B Locally: A 27B Model in 3.9 GB, Even on a Phone"
date: 2026-07-14
draft: false
tags: ["bitnet", "quantization", "qwen", "edge-ai", "small-models"]
categories: ["small-ai-models"]
description: "How to run Bonsai 27B locally: PrismML's 1-bit and ternary builds of Qwen3.6-27B fit in 3.9 to 7.2 GB, keep 90-95% of FP16 benchmark scores, and reach 11 tok/s on an iPhone 17 Pro Max."
bluesky: "PrismML rebuilt Qwen3.6-27B with binary and ternary weights: 3.9 GB deployed, ~90% of its FP16 benchmark average, 262K context, and ~11 tok/s on an iPhone 17 Pro Max."
slug: "run-bonsai-27b-locally"
hardware: "8 GB laptop and up · iPhone 17 Pro Max"
---

A 27B model normally occupies about 54 GB at 16-bit precision, and around 18 GB even as a decent 4-bit GGUF. PrismML's Bonsai 27B, released today under Apache 2.0, packs the same Qwen3.6-27B architecture into 3.9 GB. That is small enough to run Bonsai 27B locally on an ordinary 8 GB laptop, a single consumer GPU, or (in PrismML's words, but with real numbers behind it) an iPhone 17 Pro Max at roughly 11 tokens per second.

The catch-free version of that sentence doesn't exist, and this post covers the catches too. But the headline holds up better than most low-bit announcements: the weights really are binary, the benchmark drop is measured and published per-task, and the GGUF files are downloadable now.

{{< figure src="https://images.pexels.com/photos/1382195/pexels-photo-1382195.jpeg" alt="A small bonsai tree growing in a shallow terracotta pot on a wooden deck" caption="The name fits: a full-size tree, deliberately grown small. (Photo: Todd Trapani / Pexels, free)" >}}

## What Bonsai 27B actually is

Bonsai is PrismML's family of low-bit models, and the 27B is its new flagship. The smaller members shipped earlier this year (1.7B, 4B, and 8B ternary models in April, an image model in May). All of them follow the same recipe: instead of taking a finished FP16 model and rounding its weights down (post-training quantization), PrismML trains the network with weights constrained to extreme low-bit values from the start. That is the same core idea behind BitNet b1.58, which we covered in [our 1-bit LLMs explainer](/posts/1-bit-llms-bitnet-ternary-weights/); Bonsai 27B is the first time the approach has shipped at flagship scale on top of a current-generation base model.

The base is Qwen3.6-27B, Alibaba's dense hybrid-attention model (roughly 75% linear-attention layers, 25% full attention), with the architecture left unchanged. If you read [our Qwen3.6-35B-A3B guide](/posts/run-qwen3-6-35b-a3b-locally/), this is the dense sibling from the same generation. The hybrid backbone matters here: only 16 of 64 layers grow a full-attention KV cache, which is why a 262K-token context stays practical on a laptop.

Two variants ship today, and the difference is the weight alphabet:

| Variant | Weight values | True bits/weight | Deployed size | PrismML's benchmark avg (15 tests) | vs FP16 |
|---|---|---|---|---|---|
| 1-bit Bonsai 27B | {−1, +1} | 1.125 | ~3.9 GB | 76.11 | 89.5% |
| Ternary Bonsai 27B | {−1, 0, +1} | 1.71 (ships at ~2.125 in GGUF) | ~7.2 GB | 80.49 | 94.6% |
| Qwen3.6-27B Q4_K_XL (reference) | conventional | 5.2 | 17.6 GB | 84.99 | 99.9% |
| Qwen3.6-27B FP16 (reference) | 16-bit | 16.0 | ~54 GB | 85.07 | 100% |

One honesty point the announcement glosses over: the ternary variant's advertised 5.9 GB is the ideal 1.71-bit packing. The GGUF file you actually download today is 7.17 GB, because the current kernels store it at about 2.125 bits per weight. The 1-bit build has no such gap; its 3.9 GB is both the ideal and the shipped size. Both variants also take an optional 0.63 GB vision tower (HQQ 4-bit) that loads only when you send an image, and an optional 1.79 GB speculative-decoding drafter.

## The numbers: what 1.125 bits per weight costs

All benchmark figures here are PrismML's own, run with EvalScope and vLLM on an H100 in thinking mode across 15 benchmarks. No independent evaluations exist yet; treat these as a well-documented vendor claim until someone reproduces them.

The interesting comparison is not against FP16 but against conventional quantization at similar sizes. A standard "2-bit" build of Qwen3.6-27B (IQ2_XXS, really 2.8 bits per weight, 9.4 GB) scores 72.73 average, and its failures concentrate exactly where it hurts: PrismML measured it at 57.5 on AIME26 and 56.4 on LiveCodeBench while it still posts a healthy 88.93 on MMLU-Redux. Casual testing misses that kind of collapse. The 1-bit Bonsai, at less than half that file size, holds 87.08 on AIME26 and 76.40 on LiveCodeBench.

Per category, against the FP16 reference:

| Category | FP16 | 1-bit Bonsai 27B |
|---|---|---|
| Math (GSM8K, MATH-500, AIME25/26) | 95.33 | 91.66 |
| Coding (HumanEval+, MBPP+, LiveCodeBench) | 88.74 | 81.88 |
| Knowledge & reasoning (MMLU-Redux, MuSR) | 83.15 | 73.39 |
| Instruction following (IFEval, IFBench) | 78.47 | 65.74 |
| Agentic / tool calling (BFCL v3, τ²-Bench) | 80.00 | 66.03 |
| Vision (MMMU-Pro, OCR Bench v2) | 72.61 | 59.57 |

The pattern is consistent: the reasoning core survives compression well, while instruction following, tool calling, and vision take double-digit hits. PrismML says so itself in the model card's limitations section, and lists an agentic-coding-tuned variant as next on the roadmap. If your use case is multi-step tool loops, the ternary build's extra 3.3 GB buys back most of the gap.

## How to run Bonsai 27B locally

Here is the second catch: mainline llama.cpp cannot run these files yet. The Q1_0_g128 and ternary formats need custom kernels that live in PrismML's fork, which builds exactly like upstream llama.cpp on CUDA, Metal, and CPU. Apple Silicon users can alternatively use the MLX builds.

On a CUDA machine:

```bash
# PrismML's llama.cpp fork, with the 1-bit hybrid-attention kernels
git clone https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j

# 3.9 GB download
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf --local-dir .

./build/bin/llama-cli \
    -m Bonsai-27B-Q1_0.gguf \
    -p "Explain quantum computing in simple terms." \
    -n 256 \
    --temp 0.7 --top-p 0.95 --top-k 20 \
    -ngl 99
```

On a Mac the build is the same minus the CUDA flag (`cmake -B build && cmake --build build -j`), and `llama-server` works as usual if you want the web UI or an OpenAI-compatible endpoint. PrismML recommends temperature 0.7, top-p 0.95, top-k 20; those are the settings behind all its published scores.

Memory is where this model rewrites the usual math from [our RAM requirements guide](/posts/how-much-ram-for-local-llms/). PrismML's measured peaks for the 1-bit build, language model only, with the KV cache left at FP16:

| Context | 1-bit Bonsai peak | Qwen3.6-27B Q4_K_XL peak |
|---|---|---|
| 4K | 5.2 GB | 19.2 GB |
| 10K | 5.6 GB | 19.6 GB |
| 100K | 11.6 GB | 25.6 GB |

Enable the 4-bit KV cache and the 100K figure drops to about 6.8 GB, with the full 262K window fitting in roughly 9.4 GB. In practice: an 8 GB machine handles the 1-bit model at tens of thousands of tokens of context, and a 16 GB machine can max the window out. The ternary build adds ~3.3 GB to every row.

## Can a phone really run a 27B model?

Yes, with caveats about which phone and how fast. The 1-bit weights ship in an MLX Swift build ([Bonsai-27B-mlx-1bit](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit)) that PrismML measured at about 11 tokens per second on an iPhone 17 Pro Max, and the Locally AI iOS app is the packaged way to try it. An 11 tok/s reading pace is usable for chat and short answers, slow for long reasoning traces (and thinking mode is where this model earns its scores). "First 27B-class model to run on a phone" is PrismML's marketing line, but we can't name a counterexample, and the arithmetic checks out: 3.9 GB of weights plus a few GB of cache and runtime fits the 17 Pro Max's 12 GB of RAM with the screen still running.

{{< figure src="https://images.pexels.com/photos/12368095/pexels-photo-12368095.jpeg" alt="Two hands holding two iPhones with multi-lens camera modules" caption="The 1-bit build targets exactly this hardware class: recent iPhones with 12 GB of RAM, via MLX Swift. (Photo: mktomasik / Pexels, free)" >}}

## Which variant should you download?

If you have 16 GB of RAM or more, take the ternary build: at 7.2 GB deployed it keeps 94.6% of the FP16 average by PrismML's measurements, and its drops in instruction following and tool use are much shallower than the 1-bit model's. The 1-bit build is the right call for 8 GB machines, for phones, and for GPU serving where you want maximum KV headroom on a 24 GB card. If you have 24 GB and quality is everything, the conventional Q4_K_XL of the base Qwen3.6-27B still scores three to five points higher; what Bonsai sells is the tier below that, where until today nothing 27B-shaped existed at all. For the quantization background on why "2-bit" GGUFs usually disappoint, see [our Q4/Q5/Q8 explainer](/posts/gguf-quantization-levels-q4-q5-q8/).

## Speed, and one clever extra

Throughput on the 1-bit build, measured with llama-bench (generation over 128 tokens):

| Platform | Generation | Prompt processing (512 tok) |
|---|---|---|
| Apple M4 Pro (Metal) | 26.0 tok/s | 133 tok/s |
| Apple M5 Pro (Metal) | 44.2 tok/s | 421 tok/s |
| Apple M5 Max (Metal) | 66.4 tok/s | 874 tok/s |
| NVIDIA H100 (CUDA) | 104.8 tok/s | 2,755 tok/s |

The extra: both variants ship with a DSpark speculative-decoding drafter, a compact six-layer draft model trained against the Bonsai target. On CUDA it turns 104.8 tok/s into 143.8, a lossless 1.37x speedup (verification preserves the output distribution exactly). On Apple Silicon it is off by default because batch-1 verification doesn't pay for itself there yet. PrismML also quotes 0.275 mWh per generated token on an M5 Pro with the drafter enabled, an order of magnitude below its measurements of datacenter GPUs per token.

## What we couldn't verify yet

Everything above rests on PrismML's own evaluation harness. The per-benchmark tables and measurement notes in the [whitepaper repo](https://github.com/PrismML-Eng/Bonsai-demo) are unusually detailed for a model launch, which builds confidence, but independent numbers don't exist yet on day one. The other open questions: whether mainline llama.cpp merges the kernels (until then you maintain a fork build), how the model behaves outside thinking mode, and whether the announced agentic-coding variant lands. We will update this post as third-party benchmarks appear.

If you have an 8 GB laptop that has been capped at the 3-4B class from [our best small models guide](/posts/best-small-language-models-2026/), this is the most interesting download of the month: a 3.9 GB file that reasons like something ten times its size, from an approach that until now only existed in research papers and 2B-parameter demos.

## Sources

- [PrismML: Announcing 1-bit Bonsai 27B](https://prismml.com/news/prismml-releases-bonsai-27b)
- [1-bit Bonsai 27B GGUF model card (HuggingFace)](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
- [Ternary Bonsai 27B GGUF model card (HuggingFace)](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
- [Bonsai-27B-mlx-1bit (HuggingFace)](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit)
- [PrismML llama.cpp fork (GitHub)](https://github.com/PrismML-Eng/llama.cpp)
- [Bonsai whitepaper and demo repo (GitHub)](https://github.com/PrismML-Eng/Bonsai-demo)
- [Qwen3.6-27B base model (HuggingFace)](https://huggingface.co/Qwen/Qwen3.6-27B)
- [PrismML: Introducing the Ternary Bonsai family](https://prismml.com/news/prismml-introduces-ternary-bonsai-model-family)
