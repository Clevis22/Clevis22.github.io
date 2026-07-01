---
title: "How to Run LFM2.5-8B-A1B Locally: Liquid AI's On-Device Tool-Calling MoE"
date: 2026-06-30
draft: false
tags: ["small-models", "slm", "edge-ai", "liquid-ai", "lfm2", "local-inference", "mixture-of-experts"]
categories: ["small-ai-models"]
description: "A practical guide to running LFM2.5-8B-A1B locally: Liquid AI's 8.3B mixture-of-experts activates only 1.5B parameters per token, fits in ~4.8GB at Q4, and ships with day-one Ollama, llama.cpp, and MLX support."
slug: "run-lfm2-5-8b-a1b-locally"
---

LFM2.5-8B-A1B is a mixture-of-experts model from Liquid AI that holds 8.3 billion parameters but activates only 1.5 billion per token, which is what lets it run fast on a laptop CPU while behaving like a much larger model. Released on May 28, 2026, it is built for one job in particular: reliable tool calling on consumer hardware. If you want to run LFM2.5-8B-A1B locally, the Q4 quant is about 4.8GB on disk and fits on any machine with 8GB of RAM. This guide covers Ollama, llama.cpp, and MLX setup, what the benchmark numbers actually mean, and where this model earns its place against Qwen and Gemma.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/qUZVGkns1bg3sZUShBbhv.png" alt="Bar chart of the AA-Omniscience Index comparing LFM2.5-8B-A1B against Qwen3.5-4B, Gemma-4-E4B-IT, gpt-oss-20b, Granite-4.0-H-Tiny and Qwen3-30B-A3B-Thinking" caption="AA-Omniscience Index: higher rewards correct answers and penalizes hallucinations. LFM2.5-8B-A1B posts the least-negative score of the group. (Chart: Liquid AI, LFM2.5-8B-A1B model card)" >}}

## What is LFM2.5-8B-A1B?

It is the successor to LFM2-8B-A1B, which Liquid AI shipped in October 2025. The "A1B" in the name means roughly one billion active parameters: it is a sparse mixture-of-experts, so on any given token the router picks a small subset of the network to run. You store all 8.3B weights in memory, but each forward pass only touches about 1.5B of them. Memory footprint tracks the total; speed tracks the active count. That split is the whole reason MoE makes sense on the edge.

The headline specs:

| Property | Value |
|---|---|
| Total parameters | 8.3B |
| Active parameters | 1.5B |
| Architecture | 24 layers (18 double-gated LIV convolution + 6 GQA attention) |
| Context length | 128,000 tokens |
| Vocabulary | 128,000 |
| Training budget | 38 trillion tokens |
| Languages | English, Arabic, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Spanish |
| License | LFM Open License v1.0 (open weights) |

The architecture follows the same hybrid recipe as the rest of the LFM2 line: most of the stack is short-range gated convolution (Liquid's "LIV" blocks), with only six grouped-query attention blocks mixed in. Convolution is cheaper than full attention at inference time, and that is a large part of why these models decode quickly on CPUs and mobile silicon. The smaller [LFM2.5-1.2B-Thinking](/posts/run-lfm2-5-1-2b-thinking-locally/) uses the same idea at a fraction of the size. For background on Liquid's on-device philosophy, our [Liquid AI Apollo write-up](/posts/ai-in-your-pocket-liquid-ai-apollo/) goes deeper.

One important detail: like the 1.2B thinking model, LFM2.5-8B-A1B is reasoning-only. Every assistant turn contains an explicit chain of thought before the final answer. Liquid's reasoning for making it reasoning-only is worth understanding, because it is specific to MoE. In a compute-bound setting, a low active-parameter count makes each reasoning token cheap, so you get the quality boost of a thinking trace without paying much of a speed penalty.

## What changed since LFM2-8B-A1B

This is a genuine upgrade over the October 2025 model, not a re-tag. Three things changed:

- Context window went from 32,768 to 128,000 tokens.
- Vocabulary doubled from 65,536 to 128,000, mostly to tokenize non-Latin scripts more efficiently. Liquid reports the biggest compression gains in Hindi, Thai, Vietnamese, Indonesian, and Arabic.
- Pre-training scaled from 12 trillion to 38 trillion tokens, followed by large-scale reinforcement learning.

The measured effect on Liquid's own benchmarks is large, and the most interesting jump is not a raw capability score:

| Benchmark | LFM2-8B-A1B | LFM2.5-8B-A1B | Δ |
|---|---:|---:|---:|
| AA-Omniscience Non-Hallucination Rate | 7.46 | 63.47 | +56.01 |
| IFEval | 79.44 | 91.84 | +12.40 |
| IFBench | 26.00 | 56.47 | +30.47 |
| MATH500 | 74.80 | 88.76 | +13.96 |
| AIME25 | 20.00 | 42.53 | +22.53 |
| BFCLv3 (tool calling) | 45.07 | 64.36 | +19.29 |
| Tau² Telecom (agentic) | 13.60 | 88.07 | +74.47 |

The non-hallucination rate going from 7.46 to 63.47 is the story. The old model made things up constantly; this one is far more willing to decline or hedge when it does not know. For an on-device assistant that will be asked questions outside its training, that behaviour matters more than another point of MMLU.

## Benchmarks: what it is good at, and what it isn't

Here is LFM2.5-8B-A1B against a spread of models people actually run locally. The AA-Omniscience Index is the unusual column: it runs from -100 to 100, rewarding correct answers and subtracting for confident wrong ones. Every small model here scores negative (they all hallucinate more than they reliably know), so "least negative" is the win condition.

| Model | Params | AA-Omni. Index | IFEval | BFCLv3 | MATH500 | AIME25 |
|---|---|---:|---:|---:|---:|---:|
| LFM2.5-8B-A1B | 8.3B / 1.5B active | **-24.70** | **91.84** | 64.79 | 88.76 | 42.53 |
| Qwen3.5-4B | 4B | -51.53 | 87.80 | 71.06 | 80.76 | 54.28 |
| Gemma-4-E4B-IT | 8B | -50.67 | 87.74 | 57.31 | 65.00 | 34.33 |
| gpt-oss-20b | 21B / 3.6B active | -49.17 | 86.73 | n/a | n/a | n/a |
| Qwen3-30B-A3B-Thinking-2507 | 30.5B / 3.3B active | -51.31 | 90.82 | 73.39 | 86.48 | 71.67 |

Two things stand out. First, LFM2.5-8B-A1B has by far the least-negative Omniscience Index in the group, roughly half the hallucination penalty of models several times its active size. Second, it tops IFEval, edging out even the 30B Qwen. Instruction following and calibration are where the RL training clearly paid off.

Where it falls short is exactly where Liquid says it will. On hard olympiad math it trails badly: 42.53 on AIME25 versus 54.28 for the 4B Qwen and 71.67 for the 30B. Raw knowledge accuracy is low too. The model card is blunt about this and recommends against using it for heavy programming or knowledge-intensive question answering without retrieval. Treat it as an agent and instruction-follower, pair it with a search or RAG layer when you need facts, and it plays to its strengths. For how this sits against the wider field, see our [best small language models in 2026](/posts/best-small-language-models-2026/) comparison.

Its real party trick is agentic tool use. On the Tau² Telecom benchmark, which tests multi-turn task completion with tools, LFM2.5-8B-A1B scores 88.07, higher than the 30B Qwen3-30B-A3B-Thinking (21.93). A model activating 1.5B parameters that outperforms one activating 3.3B on a realistic agent task is the kind of result that justifies the MoE design.

## How to run LFM2.5-8B-A1B locally

There is day-one support across llama.cpp, MLX, vLLM, SGLang, and LM Studio, plus an official Ollama build. Pick by hardware and how much terminal you want to touch.

### With Ollama (easiest)

Ollama has the model as `lfm2.5:8b`. One command pulls and runs it:

```bash
# Default (Q4_K_M, ~4.8GB)
ollama run lfm2.5:8b

# Or pin a specific quant
ollama run lfm2.5:8b-a1b-q8_0
```

Ollama serves an OpenAI-compatible API on `http://localhost:11434`, so anything you already point at a local model works without code changes. If you are choosing between runtimes, our [Ollama vs LM Studio vs llama.cpp guide](/posts/ollama-vs-lm-studio-vs-llama-cpp/) covers the tradeoffs.

### With llama.cpp (fastest on CPU)

Liquid publishes an official GGUF repo. Note the tokenizer was updated shortly after launch to fix a tool-calling bug in llama.cpp, so pull a current build.

```bash
# Download the Q4_K_M GGUF (~4.8GB)
huggingface-cli download LiquidAI/LFM2.5-8B-A1B-GGUF \
  LFM2.5-8B-A1B-Q4_K_M.gguf --local-dir .

# Interactive chat with Liquid's recommended sampling
llama-cli -m LFM2.5-8B-A1B-Q4_K_M.gguf \
  --temp 0.2 --top-k 80 --repeat-penalty 1.05 \
  -p "Book me a table for four at 7pm and email the confirmation."

# Or run it as a local server
llama-server -m LFM2.5-8B-A1B-Q4_K_M.gguf --port 8080
```

Liquid recommends `temperature 0.2`, `top_k 80`, and a light `1.05` repetition penalty. Because this is a reasoning model, keep the temperature low; high temperature tends to derail the chain of thought.

Available quants and their disk sizes:

| Quantization | File Size |
|---|---|
| Q4_0 | 4.51 GB |
| Q4_K_M | 4.80 GB |
| Q5_K_M | 5.62 GB |
| Q6_K | 6.48 GB |
| Q8_0 | 8.39 GB |
| F16 (full) | 15.78 GB |

Q4_K_M is the right default. For what each level costs in quality, see our [Q4 vs Q5 vs Q8 quantization guide](/posts/gguf-quantization-levels-q4-q5-q8/).

### With MLX or Transformers

On an Apple Silicon Mac, the MLX build is the fastest option. Liquid ships `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` and an 8-bit variant; our [Apple Silicon guide](/posts/run-small-llms-apple-silicon/) walks through the MLX toolchain. For Python integration or fine-tuning, the native checkpoint runs in bfloat16 through Transformers (requires `transformers>=5.0.0`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "LiquidAI/LFM2.5-8B-A1B"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is C. elegans?"}],
    add_generation_prompt=True, return_tensors="pt", tokenize=True,
)["input_ids"].to(model.device)

output = model.generate(
    input_ids, do_sample=True, temperature=0.2, top_k=80,
    repetition_penalty=1.05, max_new_tokens=8192,
)
print(tokenizer.decode(output[0], skip_special_tokens=False))
```

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/yWAChLNCguGTl9lXBL47p.png" alt="Bar charts comparing LFM2.5-8B-A1B CPU inference throughput against other small models" caption="CPU inference throughput. Liquid positions LFM2.5-8B-A1B as the fastest model in its size class on CPU. (Chart: Liquid AI, LFM2.5-8B-A1B model card)" >}}

## How much hardware do you actually need?

**How much RAM does LFM2.5-8B-A1B need?**

At Q4_K_M the weights are 4.8GB on disk, and you want a little headroom for context and runtime overhead, so an 8GB machine handles it comfortably and a 16GB laptop leaves plenty of room for the full 128K context. Because it is MoE, the entire 8.3B has to sit in memory even though only 1.5B runs per token: the footprint is set by total parameters, the speed by active ones. If your Windows GPU is tight on VRAM, our [low-VRAM guide](/posts/run-local-llms-low-vram-windows-gpu/) covers CPU-offload strategies, and this model is a good fit for that pattern since the active slice is small.

**How fast is it?**

On GPU, Liquid measured 18.5K output tokens per second at high concurrency on a single H100, which works out to over 1.6 billion tokens per day. That is a server-side number, but the same sparsity that produces it is what keeps single-user CPU decode quick. Liquid positions it as the fastest model in its size class on both CPU and GPU. If you need concrete numbers for your own box, our guide on [how much RAM local LLMs need](/posts/how-much-ram-for-local-llms/) explains how memory bandwidth, not just capacity, sets your token rate.

## Should you use LFM2.5-8B-A1B?

**Is it a good default local assistant?** For agentic and assistant workloads (chaining tool calls, following multi-step instructions, structured output, multilingual chat) it is one of the strongest options you can run in under 5GB. Its instruction-following and hallucination resistance are class-leading, and the tool-calling scores hold up against models many times larger.

**When should you pick something else?** If your main workload is competition math, code generation, or knowledge recall without retrieval, a dense model like Qwen3.5-4B or Phi-4-mini will serve you better. Compare directly in our [Qwen3.5-4B vs Phi-4-mini](/posts/qwen3-5-4b-vs-phi-4-mini/) breakdown. LFM2.5-8B-A1B is a specialist: it is built to drive tools reliably, not to be an encyclopedia.

**What about licensing?** The weights ship under Liquid AI's LFM Open License v1.0, which makes them freely downloadable. It is not an OSI-approved license like Apache 2.0, so if you plan to ship it inside a commercial product, read the terms before assuming Apache-style permissions.

## Sources

- [LFM2.5-8B-A1B model card (Hugging Face)](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- [LFM2.5-8B-A1B-GGUF (Hugging Face)](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF)
- [LFM2.5-8B-A1B: An Even Better On-Device Mixture of Experts (Liquid AI blog)](https://www.liquid.ai/blog/lfm2-5-8b-a1b)
- [lfm2.5 on Ollama](https://ollama.com/library/lfm2.5)
- [LFM2 Technical Report (arXiv 2511.23404)](https://arxiv.org/abs/2511.23404)
