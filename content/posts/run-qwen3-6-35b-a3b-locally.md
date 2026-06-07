---
title: "How to Run Qwen3.6-35B-A3B Locally: A 35B MoE Model on One GPU"
date: 2026-06-07
draft: false
tags: ["small-models", "slm", "qwen", "moe", "local-inference", "agentic-coding"]
categories: ["small-ai-models"]
description: "Run Qwen3.6-35B-A3B locally with Ollama, llama.cpp, or MLX. Hardware requirements, GGUF quantization sizes, and thinking-mode setup for the 35B model that activates only 3B parameters per token."
slug: "run-qwen3-6-35b-a3b-locally"
---

Qwen3.6-35B-A3B is a Mixture of Experts model released by Alibaba's Qwen team in April 2026 that you can run locally on a single RTX 4090 or M4 Max. Despite weighing in at 35 billion total parameters, it activates only 3 billion of them per forward pass, which keeps decode speed comparable to a dense 3B model while delivering benchmark scores that rival much larger models. On SWE-bench Verified it hits 73.4%, putting it well ahead of open-weight alternatives that require similar hardware. License is Apache 2.0.

This guide covers the architecture, hardware requirements, and step-by-step setup using Ollama, llama.cpp, and MLX for Apple Silicon.

## What Makes Qwen3.6-35B-A3B Different

The model has 256 experts per MoE layer. During inference, 8 routed experts plus 1 shared expert activate per token: 9 out of 256, every time. The remaining parameters sit in memory but do not compute. This is why the decode throughput is near that of a 3B dense model even though the weight file is 22 GB at Q4.

The attention architecture is a hybrid across 40 layers. Three Gated DeltaNet layers (linear attention, better long-context efficiency) alternate with one standard Gated Attention layer in a repeating 3:1 pattern. This lets the model handle its native 262,144-token context window without the quadratic memory growth that full attention incurs at long sequences. With YaRN scaling enabled in config, the context extends to 1,010,000 tokens.

Thinking mode is on by default. Responses include a `<think>...</think>` block containing chain-of-thought reasoning before the final answer. The key improvement over earlier Qwen thinking models is thinking preservation: reasoning context is retained across conversation turns, so an agent running a multi-step task can reference prior reasoning without reprocessing it.

The model is also multimodal, supporting images and video alongside text. If you only need text inference, you can skip the vision encoder by passing `--language-model-only` to vLLM, which reduces memory and improves throughput.

{{< figure src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3.6/Figures/qwen3.6_35b_a3b_score.png" alt="Qwen3.6-35B-A3B benchmark scores across SWE-bench Verified, SWE-bench Multilingual, SWE-bench Pro, and Terminal-Bench 2.0, compared to Qwen3.5-35B-A3B and Qwen3.5-27B" caption="Official benchmark comparison from Qwen model card, showing Qwen3.6-35B-A3B versus previous generation across agentic coding tasks. Source: Qwen Team (Apache 2.0)." >}}

## Benchmark Results

These scores are from the official model card:

| Benchmark | Score |
|-----------|-------|
| SWE-bench Verified | 73.4 |
| SWE-bench Multilingual | 67.2 |
| SWE-bench Pro | 49.5 |
| Terminal-Bench 2.0 | 51.5 |
| MMLU-Pro | 85.2 |
| MMLU-Redux | 93.3 |
| GPQA Diamond | 86.0 |
| MMMU (vision) | 81.7 |
| MMBench EN (vision) | 92.8 |

The jump in agentic coding versus the previous generation is substantial. On SWE-bench Pro, Qwen3.6-35B-A3B scores 49.5 compared to 44.6 for Qwen3.5-35B-A3B, a 4.9-point improvement in the same weight class. On NL2Repo (repository-level code generation), the gap is wider: 29.4 vs 20.5 for the prior generation. For broader context on how this fits in the current open-weight field, see our [best small language models in 2026 roundup](/posts/best-small-language-models-2026/).

## Hardware Requirements

The MoE architecture means memory requirements are driven by total parameter count at load, not active count during inference. You need to fit the full 35B in memory even though only 3B compute per token.

| Quantization | File Size | Min VRAM | Notes |
|---|---|---|---|
| Q4_K_M | 22.1 GB | 24 GB | RTX 4090, RTX 3090, 24 GB M-series |
| IQ4_XS | 17.7 GB | 20 GB | Good quality-to-size tradeoff |
| Q5_K_M | 26.5 GB | 32 GB | RTX 5090, A6000, M4 Max 36 GB+ |
| Q8_0 | 36.9 GB | 48 GB | Dual RTX 4090 or workstation GPU |
| BF16 | 69.4 GB | 80 GB | H100, A100 80 GB |

Q4_K_M at 22.1 GB is the recommended starting point for a 24 GB GPU. The RTX 4090 comfortably loads it; the RTX 3090 (also 24 GB) loads it but leaves little room for long context.

For Windows users on 16 GB or less, the IQ3_S GGUF (13.7 GB) can run on an RTX 4080 with reduced context. Our [low-VRAM Windows GPU guide](/posts/run-local-llms-low-vram-windows-gpu/) covers split inference and CPU offloading if you need to push below 16 GB.

Decode speed scales directly from the MoE sparsity: because only 3B parameters compute per token, throughput is roughly equivalent to running a dense 3B model regardless of which GPU you use.

## Quickstart with Ollama

Ollama shipped native Qwen3.6 support at launch. One command downloads and runs the Q4_K_M variant (24 GB on disk):

```bash
ollama run qwen3.6:35b-a3b
```

To call it from Python using the OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="qwen3.6:35b-a3b",
    messages=[{"role": "user", "content": "Refactor this function to use async/await:..."}],
    temperature=0.6,
    top_p=0.95,
)
print(response.choices[0].message.content)
```

Thinking mode is active by default. The response will start with `<think>` reasoning before the answer. To disable it for a request, add `/no_think` at the end of your prompt, or use the API parameter:

```python
response = client.chat.completions.create(
    model="qwen3.6:35b-a3b",
    messages=[{"role": "user", "content": "Translate: Hello world"}],
    temperature=0.7,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Use thinking mode for coding and multi-step reasoning tasks. Disable it for simple Q&A or translation where the reasoning overhead adds latency without improving quality.

## Setup with llama.cpp

For more control over quantization choice and context length, download a GGUF from Unsloth's collection and run with llama.cpp:

```bash
# Download Q4_K_M (22.1 GB)
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \
  --include "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
  --local-dir ./qwen3.6-35b-a3b

# Run with 8192 context (increase if your VRAM headroom allows)
llama-cli \
  -m ./qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -c 8192 \
  -n 2048 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  -p "You are a helpful coding assistant."
```

For vision inference, add `--mmproj-auto` to let llama.cpp fetch the multimodal projector automatically.

If you are deciding between IQ4_XS (17.7 GB) and Q4_K_M (22.1 GB), IQ4_XS typically scores within 1-2% of Q4_K_M on coding benchmarks while fitting a lower VRAM budget. For a full breakdown of what each quantization level gives up, see the [GGUF quantization levels explainer](/posts/gguf-quantization-levels-q4-q5-q8/).

## Apple Silicon with MLX

MLX-format versions are available from the mlx-community organization on Hugging Face. Because Qwen3.6-35B-A3B includes a vision encoder, use `mlx_vlm` rather than `mlx_lm`:

```bash
pip install mlx-vlm

python -m mlx_vlm.generate \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --max-tokens 2048 \
  --prompt "Explain this error and fix it: ..."
```

The 4-bit MLX variant is 20.4 GB and fits comfortably in 24 GB unified memory. On M4 Max with 36 GB you get headroom for 8K-16K context at that quantization. For notes on memory allocation and performance across M3, M4, and M4 Pro configurations, see the [Apple Silicon local inference guide](/posts/run-small-llms-apple-silicon/).

{{< figure src="https://images.pexels.com/photos/34027172/pexels-photo-34027172.jpeg" alt="Close-up of a Radeon graphics card in blue ambient light showing the GPU chip and PCB" caption="A 24 GB GPU (RTX 3090, RTX 3090 Ti, RTX 4090) is the minimum for Q4_K_M at useful context lengths. Photo: Nikolaos Kofidis via Pexels." >}}

## Agentic Coding with Aider

Qwen3.6-35B-A3B was trained specifically for repository-level coding tasks and integrates directly with Aider via the Ollama backend:

```bash
pip install aider-chat

aider --model ollama/qwen3.6:35b-a3b
```

The thinking preservation feature is practically useful here. When Aider sends follow-up edit instructions in the same session, the model retains its earlier reasoning about the file structure rather than re-deriving it from scratch. On multi-file refactors, this reduces the chance of contradictory changes across files.

For long sessions, the 262K native context window accommodates most codebases without chunking. The YaRN-extended 1M context is available if you modify `config.json` as documented in the model card, though generating full 1M context requires substantial VRAM beyond what a single 4090 provides.

For a wider look at what you can actually do with a local model in a coding workflow, see [what you can do with a local small LLM](/posts/what-can-you-do-with-local-small-llm/).

## Common Questions About Qwen3.6-35B-A3B

**What is the difference between Qwen3.6-35B-A3B and Qwen3.6-27B?**

The 35B-A3B is a Mixture of Experts model (35B total, 3B active). The 27B is a dense model where all 27 billion parameters compute every token. The 35B-A3B is faster at decode for the same VRAM, but the 27B tends to score higher on vision benchmarks and has simpler setup for multimodal tasks. If you primarily want coding or agentic text tasks, the 35B-A3B is the better fit. If vision is important, the 27B is worth the slower decode speed.

**Can it run without a GPU?**

Yes. The IQ3_S GGUF (13.7 GB) runs via CPU inference on llama.cpp. Community benchmarks include tests on Raspberry Pi 5 (16 GB) and Intel Core i7 systems. Expect 1-4 tokens per second on CPU depending on hardware. It is usable for low-frequency tasks but not interactive chat sessions.

**How do I disable thinking mode permanently?**

In Ollama, create a Modelfile with the system prompt parameter:

```
FROM qwen3.6:35b-a3b
SYSTEM "You are a helpful assistant."
PARAMETER temperature 0.7
```

Then call `ollama create qwen3.6-instruct -f Modelfile` and use `/no_think` in prompts, or always pass `enable_thinking: false` in API calls.

**Is commercial use permitted?**

Yes. Apache 2.0 allows commercial use, modification, and redistribution, including fine-tuned derivatives. No royalties or attribution requirements beyond keeping the license file.

**What sampling parameters should I use for coding tasks?**

The Qwen team recommends `temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0` for thinking mode on precise coding tasks. For general reasoning, bump temperature to 1.0 and add `presence_penalty=1.5` to reduce repetition. For non-thinking instruct tasks, use `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.

## Sources

- [Qwen3.6-35B-A3B model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6-35B-A3B official blog post (Qwen Team, April 2026)](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
- [Qwen3.6 GitHub repository](https://github.com/QwenLM/Qwen3.6)
- [Unsloth GGUF quantizations for Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [Ollama library page: qwen3.6:35b-a3b](https://ollama.com/library/qwen3.6:35b-a3b)
- [byteshape GGUF collection with per-quant sizes](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-GGUF)
