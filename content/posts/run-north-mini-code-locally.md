---
title: "How to Run North Mini Code Locally: Cohere's 30B-A3B Coding Model"
date: 2026-07-08
draft: false
tags: ["small-models", "slm", "cohere", "moe", "agentic-coding", "local-inference"]
categories: ["small-ai-models"]
description: "Run North Mini Code 1.0 locally: Cohere's 30B-A3B open-weight coding MoE. GGUF sizes, the llama.cpp build for the cohere2moe architecture, hardware requirements, and benchmarks."
slug: "run-north-mini-code-locally"
---

North Mini Code 1.0 is Cohere's first agentic coding model, and you can run North Mini Code locally without a datacenter behind it. It is a Mixture of Experts model with 30 billion total parameters but only 3 billion active per token, released under Apache 2.0 on June 9, 2026. The sparsity is the entire point: a 3B active count gives decode speed close to a small dense model, while the full 30B of weights (about 19 GB at Q4) fits on a single 24 GB consumer GPU. This guide covers what the model is, the benchmark numbers, the hardware you need, and the exact steps to get it running with llama.cpp.

One caveat up front, because it will bite you otherwise: North Mini Code uses a new `cohere2moe` architecture that stock llama.cpp and Ollama cannot load yet. You have to build llama.cpp from an open pull request. The steps are below.

## What North Mini Code Is

The model is a decoder-only transformer with sparse MoE feed-forward layers. Each MoE block holds 128 experts and activates 8 of them per token through a router that applies a sigmoid before top-k selection. Every expert is a SwiGLU feed-forward network. That is where the 30B-total, 3B-active split comes from: the router only ever lights up a fraction of the weight file for any given token.

Attention is where Cohere did something less common. The layers interleave two attention types in a 3:1 ratio: sliding-window attention with RoPE positional encoding for three layers, then one layer of global attention with no positional embeddings (NoPE). The windowed layers keep the KV cache bounded on most of the network, which is what makes the advertised 256K context window practical rather than theoretical. Maximum output length is 64K tokens.

Training was two stages of supervised fine-tuning followed by reinforcement learning with verifiable rewards (RLVR), the reward signal coming from whether generated code actually passes tests. The model supports tool use and interleaved thinking, so it can reason between tool calls the way an agent needs to. It is text-only. There is no vision encoder.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/6361a793c12a09b8a3184bff/g-SYXPG1oIxHEwnItd3he.png" alt="Architecture diagram of North Mini Code showing the mixture-of-experts feed-forward blocks with 128 experts and the interleaved sliding-window and global attention layers" caption="North Mini Code's architecture: sparse MoE feed-forward with 128 experts (8 active per token) and interleaved sliding-window / global attention. Source: Cohere Labs (Apache 2.0)." >}}

## Benchmark Results for North Mini Code

Cohere trains this model for one job, software engineering, and the scores reflect that focus rather than broad general knowledge. These are the headline numbers from the model card:

| Benchmark | Score |
|-----------|-------|
| SWE-bench Verified | 67.6% |
| SWE-bench Pro | 40.2% |
| Terminal-Bench v2 | 36% |

For independent context, Artificial Analysis places North Mini Code at 33.4 on its Coding Index. That puts it above GLM-4.7-Flash (25.9) and just below the newer Qwen3.6-35B-A3B (35.2), which is a fair result for a model with a third of the active-parameter budget of most of its rivals on raw quality. Artificial Analysis measured decode throughput at roughly 199 output tokens per second and scored it 27.6 on the broader Intelligence Index, so this is a coding specialist, not a generalist you would reach for on trivia or open-ended writing.

The efficiency claims are the interesting part. Cohere reports up to 2.8x higher output throughput than Devstral Small 2 (a 24B dense model) and a 30% lower inter-token latency, both a direct consequence of the sparse MoE design. On Cohere's own coding evaluations, North Mini Code also comes out ahead of models many times its size, including Devstral 2 at 123B parameters. For a same-class comparison at higher raw quality, our [Qwen3.6-35B-A3B guide](/posts/run-qwen3-6-35b-a3b-locally/) covers a 35B MoE with the same 3B active count that scores 73.4% on SWE-bench Verified but is a heavier download.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/6361a793c12a09b8a3184bff/f8NXK5yKtc6xE-hJ4XWbd.png" alt="Bar chart comparing North Mini Code against Qwen3.5 35B-A3B, Gemma 4 26B-A4B, Nemotron 3 Super, and Mistral Small 4 on SWE-bench Verified, SWE-bench Pro, and Terminal-Bench" caption="North Mini Code versus larger open-weight models on agentic coding benchmarks. Source: Cohere Labs model card (Apache 2.0)." >}}

## Hardware Requirements

Because it is an MoE model, memory is driven by the total parameter count at load time, not the active count. You need the whole 30B in memory even though only 3B compute per token. The upside lands at inference: decode runs at roughly the speed of a dense 3B model once the weights are resident.

The quantized GGUF sizes below come from Unsloth's release. These are the practical download sizes, not the FP8 datacenter build:

| Quantization | File Size | Min VRAM | Notes |
|---|---|---|---|
| UD-IQ2_XXS | 9.78 GB | 12 GB | lowest quality, fits a 12 GB card |
| UD-Q3_K_M | 14.2 GB | 16 GB | good balance for a 16 GB GPU |
| UD-Q4_K_M | 19.2 GB | 24 GB | recommended default |
| UD-Q4_K_XL | 19.3 GB | 24 GB | slightly higher-fidelity 4-bit |
| Q8_0 | 32.4 GB | 40 GB+ | near-lossless, for workstation cards |
| BF16 | 61 GB | 80 GB | full precision, H100 territory |

Q4_K_M at 19.2 GB is the starting point for a 24 GB GPU such as an RTX 3090 or 4090, leaving a little room for context. If you are on 16 GB, drop to UD-Q3_K_M (14.2 GB) and expect a small quality hit on the harder SWE-bench problems. The [GGUF quantization levels explainer](/posts/gguf-quantization-levels-q4-q5-q8/) breaks down exactly what each step down costs you.

No GPU? The 3B active count makes CPU inference more viable here than for a dense 30B. With UD-Q4_K_M loaded into system RAM (budget 24 GB or more), llama.cpp will run it on CPU at a few tokens per second, which is fine for background agent tasks even if it is slow for interactive chat. To split the model across a smaller GPU and system RAM, the offloading techniques in our [low-VRAM Windows GPU guide](/posts/run-local-llms-low-vram-windows-gpu/) apply directly here through the `--n-gpu-layers` flag.

Cohere's own recommended server deployment is a single H100 at FP8 or FP4, which is the on-premise, full-throughput path for teams. The GGUF route above is what makes the model realistic on a home workstation.

## Running North Mini Code Locally with llama.cpp

The `cohere2moe` architecture is not in a stock llama.cpp release at the time of writing. Support lives in an open pull request (#24260), so you build from that branch. Once the PR merges into a tagged release, a normal `llama.cpp` build will work and this step goes away.

```bash
# Clone and check out the cohere2moe PR branch
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch origin pull/24260/head:cohere2-moe
git checkout cohere2-moe

# Build (CUDA example; drop -DGGML_CUDA=ON for CPU-only)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

Download a quant from the Unsloth GGUF repo:

```bash
huggingface-cli download unsloth/North-Mini-Code-1.0-GGUF \
  --include "North-Mini-Code-1.0-UD-Q4_K_XL.gguf" \
  --local-dir ./north-mini-code
```

Then run it. The `--jinja` flag matters: it applies the model's chat template so tool-use and thinking tokens are formatted correctly. Cohere recommends `temperature 1.0` and `top_p 0.95`.

```bash
./build/bin/llama-cli \
  --model ./north-mini-code/North-Mini-Code-1.0-UD-Q4_K_XL.gguf \
  --jinja \
  --n-gpu-layers 99 \
  --ctx-size 16384 \
  --temp 1.0 \
  --top-p 0.95
```

For agent workflows, run it as an OpenAI-compatible server instead and point your coding tool at `http://localhost:8080/v1`:

```bash
./build/bin/llama-server \
  --model ./north-mini-code/North-Mini-Code-1.0-UD-Q4_K_XL.gguf \
  --jinja \
  --n-gpu-layers 99 \
  --ctx-size 16384 \
  --host 0.0.0.0 \
  --port 8080
```

Lower `--n-gpu-layers` if the model does not fit your VRAM; each layer you keep off the GPU spills to system RAM and costs some speed. Raise `--ctx-size` toward the model's 256K ceiling only as far as your memory allows, since the KV cache grows with context even on this windowed-attention design.

## Why Ollama and LM Studio Cannot Run It Yet

Both Ollama and LM Studio use llama.cpp under the hood, so they inherit the same limitation: until the `cohere2moe` pull request lands in a released llama.cpp version they bundle, neither can load these GGUFs. Trying to `ollama run` a North Mini Code GGUF today will fail with an unknown-architecture error. Once the support is merged and those tools update their bundled llama.cpp, importing the GGUF through a Modelfile (Ollama) or the local model folder (LM Studio) will work with no special steps. Until then, the build-from-PR path above is the only local option. If you want a coding model that runs in Ollama today, our [Qwen3-Coder-Next walkthrough](/posts/qwen3-coder-next-local-coding-agent/) has a model with native runtime support.

## Common Questions About North Mini Code

**Is North Mini Code good for local coding agents?**

Yes, that is what it was built for. It supports tool use and interleaved thinking, and it was trained with reinforcement learning against tests that the code has to pass. Point a tool like Aider or OpenCode at the `llama-server` endpoint and it behaves as an agent backend. The 256K context window is large enough to hold most single-repository sessions without chunking.

**How does it compare to Qwen3.6-35B-A3B?**

They share the same shape, 3B active parameters, but Qwen3.6-35B-A3B is the stronger raw coder (73.4% vs 67.6% on SWE-bench Verified) and has native Ollama support today. North Mini Code is a smaller download (19.2 GB vs 22.1 GB at Q4) and Cohere targets it squarely at agentic and terminal tasks. If you want the higher SWE-bench score and easier setup, Qwen wins; if you want the lighter footprint under Apache 2.0 and are comfortable building llama.cpp, North Mini Code is a reasonable pick.

**Can it run on a laptop?**

On a laptop with a 24 GB unified-memory Apple Silicon chip or a 16 GB discrete GPU, yes, using UD-Q4_K_M or UD-Q3_K_M respectively. On a typical 16 GB RAM laptop with no capable GPU it will run on CPU but slowly, so it is better suited to a background task than live pair programming. There is no MLX build published yet, so Apple Silicon users go through the same llama.cpp route rather than mlx_lm.

**Is commercial use allowed?**

Yes. North Mini Code 1.0 is Apache 2.0, which permits commercial use, modification, and redistribution including fine-tuned derivatives, with no royalties or attribution beyond keeping the license file.

**What quantization should I start with?**

UD-Q4_K_M (19.2 GB) if you have 24 GB of VRAM. It is the standard quality-to-size balance and loses very little against the full BF16 weights on coding tasks. Step down to UD-Q3_K_M only if you cannot fit Q4, and reach for Q8_0 or BF16 only if you have the memory to spare and want the last few points of accuracy. If you want to produce your own quant from the original weights, the process in our [llama.cpp quantization guide](/posts/how-to-quantize-model-llama-cpp/) works once you have a cohere2moe-aware build.

## Sources

- [North Mini Code 1.0 model card (Hugging Face)](https://huggingface.co/CohereLabs/North-Mini-Code-1.0)
- [Introducing North Mini Code (Cohere blog)](https://cohere.com/blog/north-mini-code)
- [Introducing North Mini Code (Cohere Labs, Hugging Face blog)](https://huggingface.co/blog/CohereLabs/introducing-north-mini-code)
- [North Mini Code analysis (Artificial Analysis)](https://artificialanalysis.ai/articles/north-mini-code-cohere-s-small-coding-focused-moe-model)
- [Unsloth North-Mini-Code-1.0 GGUF quantizations](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF)
