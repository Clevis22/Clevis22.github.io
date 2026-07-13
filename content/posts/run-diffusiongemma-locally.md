---
title: "DiffusionGemma: Google's Text-Diffusion Model That Generates 4x Faster, and How to Run It Locally"
date: 2026-07-01
draft: false
tags: ["diffusion", "gemma", "moe", "local-inference", "small-models"]
categories: ["small-ai-models"]
description: "DiffusionGemma is Google's open-weight text-diffusion model: 25.2B total, 3.8B active, up to 4x faster generation than Gemma 4. How discrete text diffusion works, honest benchmarks, and how to run DiffusionGemma locally with vLLM, Transformers, and MLX."
slug: "run-diffusiongemma-locally"
---

DiffusionGemma is Google DeepMind's first open-weight text-diffusion model, and it does something no other Gemma release has done: it generates whole blocks of text at once instead of one token at a time, which is what lets it hit up to 4x the throughput of a same-size autoregressive model. Released on June 10, 2026 under Apache 2.0, it carries 25.2B total parameters but activates only 3.8B per step, and a quantized copy fits inside 18GB of VRAM. If you want to run DiffusionGemma locally, the honest summary is this: it is fast and genuinely novel, it is worse than standard Gemma 4 on almost every quality benchmark, and whether that trade is worth it depends entirely on what you are building. This guide covers what text diffusion actually is, what the numbers say, and how to get it running with vLLM, Transformers, or MLX.

{{< figure src="https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/diffusion_architecture.original.png" alt="Diagram of DiffusionGemma's block-autoregressive denoising process, alternating between prefill and parallel denoising stages over a 256-token canvas" caption="DiffusionGemma processes text in blocks: it prefills a canvas, then denoises the whole block in parallel across several passes rather than emitting tokens one by one. (Diagram: Google for Developers, DiffusionGemma developer guide)" >}}

## What is DiffusionGemma?

The repo name is `google/diffusiongemma-26B-A4B-it`. As with other Gemma mixture-of-experts models, the name rounds up: the real total is 25.2B parameters, and "A4B" refers to the roughly four billion (3.8B measured) that fire on any given step. It is built on the Gemma 4 26B-A4B foundation, then fitted with a diffusion head drawn from Google's Gemini Diffusion research. The weights are on Hugging Face under Apache 2.0, the same permissive license as the rest of the Gemma 4 family and a genuine advantage over open-weight releases that ship under custom terms.

The headline specs:

| Property | Value |
|---|---|
| Total parameters | 25.2B |
| Active parameters | 3.8B |
| Experts | 128 total, 8 active + 1 shared |
| Generation | Discrete text diffusion (block-autoregressive) |
| Canvas length | 256 tokens |
| Context length | Up to 256K tokens |
| Modalities | Text, image, video (up to 60s) |
| Vision encoder | ~550M parameters |
| Languages | 35+ (pre-trained on 140+) |
| License | Apache 2.0 |

Two things matter for anyone deciding whether to run it. First, this is still a sparse MoE, so memory tracks the full 25.2B while speed tracks the 3.8B active slice, the same economics behind [Qwen3.6-35B-A3B on a single GPU](/posts/run-qwen3-6-35b-a3b-locally/). Second, it is multimodal in, text out: it accepts images and short video alongside text, but the diffusion head produces text. The vision stack is inherited from Gemma 4.

## How text diffusion actually works

This is the part worth understanding, because the speed and the weaknesses come from the same design choice.

A normal language model is autoregressive. It writes left to right, one token at a time, and each token is conditioned on everything before it. That is why generation latency scales with output length: 500 tokens means 500 sequential forward passes, and you cannot start token 200 until token 199 exists.

A diffusion language model works the other way. It starts with a block of masked or noised positions, a "canvas," and refines the entire block in parallel over a small number of denoising passes. Instead of committing to one token and moving on, it looks at the whole canvas with bidirectional attention (both left and right context) and fills in the positions it is most confident about, pass after pass, until the block resolves.

DiffusionGemma uses a block-autoregressive version of this. It works on a 256-token canvas at a time. Within each block, it denoises in parallel and locks in roughly 15 to 20 tokens per forward pass, using an entropy-bound sampler that only finalizes tokens once the model's uncertainty about them drops below a threshold. When a block is done, it prefills the next one. So the model is autoregressive at the block level but diffusion-based inside each block. Committing 15 to 20 tokens per pass instead of one is where the up-to-4x speedup comes from.

Bidirectional attention has a second consequence beyond speed. Because the model can revise tokens in light of what comes after them, it handles problems where the global structure matters more than left-to-right flow. Google's clearest demonstration is Sudoku: the base model solves 0% of puzzles, but a fine-tuned diffusion variant reaches 80%, because filling a grid is a constraint-satisfaction problem where every cell depends on cells that come "later" in reading order.

{{< figure src="https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/DiffusionGemma-Sudoku-Comparison.original.png" alt="Comparison showing DiffusionGemma solving a Sudoku grid by parallel denoising versus an autoregressive model filling cells one at a time" caption="Parallel denoising suits constraint problems. A fine-tuned DiffusionGemma solves 80% of Sudoku puzzles; the base model solves 0%. (Diagram: Google for Developers, DiffusionGemma developer guide)" >}}

## DiffusionGemma benchmarks vs Gemma 4

Here is the trade laid bare. Google publishes DiffusionGemma against the autoregressive Gemma 4 26B-A4B it was built from, so this is a clean same-architecture, same-size comparison where the only real variable is diffusion versus token-by-token generation.

| Benchmark | DiffusionGemma 26B-A4B | Gemma 4 26B-A4B |
|---|---:|---:|
| MMLU Pro | 77.6% | 82.6% |
| AIME 2026 (no tools) | 69.1% | 88.3% |
| LiveCodeBench v6 | 69.1% | 77.1% |
| Codeforces ELO | 1429 | 1718 |
| GPQA Diamond | 73.2% | 82.3% |
| Tau2 (avg over 3) | 56.2% | 68.2% |
| HLE (no tools) | 11.0% | 8.7% |
| HLE (with search) | 11.9% | 17.2% |
| BigBench Extra Hard | 47.6% | 64.8% |
| MMMLU | 81.5% | 86.3% |
| MMMU Pro | 54.3% | 73.8% |
| MATH-Vision | 70.5% | 82.4% |
| MRCR v2 (8 needle, 128k) | 32.0% | 44.1% |

The pattern is consistent: autoregressive Gemma 4 wins nearly everywhere, and on hard reasoning the gap is wide. AIME 2026 drops from 88.3% to 69.1%, and BigBench Extra Hard falls by 17 points. The one place DiffusionGemma comes out ahead is HLE without tools (11.0% vs 8.7%), a narrow and noisy result at those low absolute scores, so it is not a reason to pick the model. Treat DiffusionGemma as a speed variant that gives up measurable quality, not as a free upgrade. If your priority is the strongest answer per query rather than tokens per second, the standard [Gemma 4 release](/posts/gemma-4-taking-agentic-workflows-to-the-edge/) is the better pick, and our [best small language models in 2026](/posts/best-small-language-models-2026/) roundup covers the wider field.

Where it earns its keep is speed. Google reports 1000+ tokens per second on a single NVIDIA H100 and 700+ on a consumer RTX 5090, with the model card citing over 1100 tokens per second in low-batch settings on an H100 at FP8. For interactive workloads (in-line editing, autocomplete, rapid iteration where you regenerate constantly) that throughput changes how the tool feels to use, even if each individual answer is a notch weaker.

## How to run DiffusionGemma locally

Because the diffusion head is non-standard, runtime support is narrower than a normal Gemma drop. At launch Google shipped integrations for vLLM, Hugging Face Transformers, MLX, and SGLang, with Unsloth and NVIDIA NeMo for fine-tuning. llama.cpp support is listed as arriving soon, so Ollama and LM Studio through the GGUF path are not there yet. If you are weighing runtimes in general, our [Ollama vs LM Studio vs llama.cpp guide](/posts/ollama-vs-lm-studio-vs-llama-cpp/) still applies once GGUF lands.

### With vLLM (recommended)

vLLM is the primary integration and exposes an OpenAI-compatible server. Google's documented launch command:

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --attention-backend TRITON_ATTN \
  --generation-config vllm \
  --hf-overrides '{"diffusion_sampler": "entropy_bound", "diffusion_entropy_bound": 0.1}' \
  --diffusion-config '{"canvas_length": 256}' \
  --enable-chunked-prefill
```

The two diffusion-specific knobs are the sampler and the canvas. `diffusion_entropy_bound` (default 0.1) sets how confident the model must be before it finalizes a token: raise it and generation runs faster but sloppier, lower it for more careful output. `canvas_length` is the block size; 256 is the trained default and the safe starting point. Once the server is up it behaves like any OpenAI endpoint on `http://localhost:8000`.

### With Transformers

For Python integration, the checkpoint runs through the latest Transformers release the same way the rest of the Gemma 4 family does, using the standard chat-template flow. This is the path for scripting, evaluation, or wiring the model into an existing pipeline. Fine-tuning goes through Unsloth or NeMo if you want to replicate the kind of task-specific gains Google showed on Sudoku.

### With MLX on Apple Silicon

Apple Silicon users get an MLX build. The diffusion decode maps well to unified memory, and MLX is the fastest local option on a Mac; our [Apple Silicon guide](/posts/run-small-llms-apple-silicon/) walks through the toolchain if you are new to it.

### Hardware and VRAM

DiffusionGemma quantizes to NVFP4, a 4-bit floating-point format native to NVIDIA's Blackwell cards (the RTX 50 series), and in that form it fits within 18GB of VRAM. That puts a 24GB card like an RTX 4090 or the 32GB RTX 5090 comfortably in range for single-user local inference; the 5090 is also where the 700+ tokens/second figure comes from. This is not a 6GB-laptop model. If your GPU is tighter, our [low-VRAM Windows GPU guide](/posts/run-local-llms-low-vram-windows-gpu/) covers offload strategies, though a 25.2B MoE is a stretch on budget cards, and the [GGUF quantization explainer](/posts/gguf-quantization-levels-q4-q5-q8/) will be the reference once llama.cpp support arrives.

## DiffusionGemma FAQ

**Is DiffusionGemma faster than Gemma 4?** Yes, up to 4x on the same hardware for text generation, because it commits 15 to 20 tokens per forward pass instead of one. It reaches 1000+ tokens/second on an H100 and 700+ on an RTX 5090.

**Is it as accurate as Gemma 4?** No. On Google's own benchmarks the autoregressive Gemma 4 26B-A4B beats it on nearly every task, with double-digit gaps on hard reasoning like AIME and BigBench Extra Hard. You are trading quality for speed.

**Can I run it in Ollama or LM Studio?** Not at launch. The supported runtimes are vLLM, Transformers, MLX, and SGLang. llama.cpp support (which Ollama and LM Studio depend on) is listed as coming soon.

**What hardware do I need?** A GPU with at least 18GB of VRAM after NVFP4 quantization, so realistically a 24GB card or the 32GB RTX 5090. It is not a low-VRAM or Raspberry Pi model.

## Should you use DiffusionGemma?

For most local setups, the plain answer is not yet. If you want the best response per query on a single GPU, standard autoregressive Gemma 4 or a dense small model gives you more accuracy for the same footprint, and the tooling is mature. DiffusionGemma's benchmark deficit is real and consistent.

It becomes interesting in two situations. The first is throughput-bound interactive work: code completion, live editing, or any UI where you regenerate constantly and 4x faster tokens change the experience more than a few points of MMLU would. The second is research and experimentation. This is the first credible open-weight text-diffusion model at this scale under a permissive license, which makes it the obvious starting point if you want to study parallel generation, fine-tune for constraint-satisfaction tasks like the Sudoku example, or benchmark diffusion against autoregression on your own workload. Download it, run the vLLM server, and measure it against the Gemma 4 checkpoint it was built from. The gap between the two is exactly the cost of diffusion, and now you can price it yourself.

## Sources

- [DiffusionGemma model card (Hugging Face)](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- [DiffusionGemma: 4x faster text generation (Google blog)](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)
- [DiffusionGemma: The Developer Guide (Google for Developers)](https://developers.googleblog.com/diffusiongemma-the-developer-guide/)
- [Google AI Releases DiffusionGemma, a 26B MoE Open Model Using Text Diffusion (MarkTechPost)](https://www.marktechpost.com/2026/06/10/google-ai-releases-diffusiongemma-a-26b-moe-open-model-using-text-diffusion-for-up-to-4x-faster-generation/)
