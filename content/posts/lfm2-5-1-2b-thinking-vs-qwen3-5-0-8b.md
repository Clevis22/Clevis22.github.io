---
title: "LFM2.5-1.2B-Thinking vs Qwen3.5-0.8B: Which Tiny Model Deserves Your Gigabyte?"
date: 2026-07-15
draft: false
tags: ["comparison", "liquid-ai", "qwen", "reasoning", "small-models"]
categories: ["comparisons"]
description: "LFM2.5-1.2B-Thinking vs Qwen3.5-0.8B: benchmarks, architectures, thinking modes, and which sub-1.5B model to run for math, vision, or long documents."
bluesky: "Two thinking models around 1 GB: LFM2.5-1.2B-Thinking wins math by a mile, Qwen3.5-0.8B reads images and 262K tokens. Which earns the spot on your phone?"
slug: "lfm2-5-1-2b-thinking-vs-qwen3-5-0-8b"
---

LFM2.5-1.2B-Thinking vs Qwen3.5-0.8B is the most interesting matchup in the tiny-model weight class right now. Both fit in roughly a gigabyte of memory once quantized. Both can write out a reasoning trace before answering. Both decode at usable speeds on a phone CPU. And they are built on opposite bets about what a model this small should spend its limited capacity on. Liquid AI poured everything into text reasoning: math, instruction following, tool calls. Alibaba spread the budget across image and video input, 201 languages, and a 262,144-token context window.

We have covered each model on its own: see the [LFM2.5-1.2B-Thinking deployment guide](/posts/run-lfm2-5-1-2b-thinking-locally/) and the [Qwen3.5-0.8B architecture breakdown](/posts/qwen3-5-tiny-multimodal-thinking-model/). This post puts them side by side and answers the practical question: which one should get the gigabyte on your device?

{{< figure src="https://images.pexels.com/photos/31403621/pexels-photo-31403621.jpeg?auto=compress&cs=tinysrgb&w=1260" alt="Two boxers wearing protective headgear and gloves exchanging punches in a ring" caption="A genuine lightweight bout: two sub-1.5B models, one spot on your device. (Photo: César O'neill, Pexels, free to use)" >}}

## LFM2.5-1.2B-Thinking vs Qwen3.5-0.8B: the spec sheet

| | LFM2.5-1.2B-Thinking | Qwen3.5-0.8B |
|---|---|---|
| Parameters | 1.17B | 0.8B |
| Input modalities | Text only | Text, image, video |
| Context window | 32,768 tokens | 262,144 tokens |
| Thinking traces | Always on | On demand |
| Languages | 8 | 201 |
| Architecture | 16 layers: 10 gated convolution (LIV) + 6 GQA attention | 24 layers: Gated DeltaNet + gated attention, 3:1 hybrid |
| License | LFM Open License v1.0 | Apache 2.0 |
| Typical local size | 696 MB (Q4_0 GGUF) | 1.0 GB (Ollama) |
| Knowledge cutoff | Mid-2024 | Not published |

One asterisk before anything else: LFM2.5 carries about 46% more parameters than Qwen3.5-0.8B, so this is not a strictly equal fight on model capacity. On deployed footprint it flips the other way. The Qwen Ollama package is 1.0 GB because it ships a vision stack along with the language model, while LFM2.5's Q4_0 GGUF is 696 MB of pure text engine. On disk and in RAM they land close enough that for most people the real question is which feature set they need, not which fits.

## Two opposite architecture bets

Neither model is a plain transformer, and the deviations tell you exactly what each team cared about.

LFM2.5 is mostly not attention at all. Of its 16 layers, ten are Liquid's double-gated short-range convolution blocks and only six are grouped-query attention. Convolution is cheap at inference time, which is why the model decodes at 70 tokens per second on a Samsung Galaxy S25 Ultra CPU while staying inside 719 MB of memory, by Liquid's own measurements. The design goal is tokens per second per watt.

Qwen3.5 keeps attention everywhere but changes its nature. Three out of every four layers use Gated DeltaNet, a linear attention mechanism that replaces the ever-growing KV cache with a fixed-size state, and every fourth layer is full softmax attention for precise recall. That is what makes a 262,144-token context window practical on hardware where a standard transformer's KV cache would have blown past the memory budget long before. The design goal is context per megabyte.

Both teams reached the same underlying conclusion, which is that full softmax attention in every layer is a luxury a 1B-class on-device model cannot afford. They just spent the savings on different things: Liquid on raw speed, Alibaba on context length.

## What the benchmarks show, and what they cannot

A fair warning before the table: these numbers come from each model's official card, and the two labs did not run the same protocol. Liquid reports LFM2.5-1.2B-Thinking with its reasoning traces on. Qwen publishes text benchmarks for the 0.8B in non-thinking mode only. Treat the gap as directional.

| Benchmark | LFM2.5-1.2B-Thinking | Qwen3.5-0.8B (non-thinking) |
|---|---|---|
| MMLU-Pro | 49.65 | 29.7 |
| IFEval | 88.42 | 52.1 |

The gap is too large to be a mode artifact. Liquid also publishes numbers for its instruct sibling, which skips the thinking trace entirely: 44.35 on MMLU-Pro and 86.23 on IFEval. Even without any reasoning tokens, the LFM2.5 base comfortably clears Qwen3.5-0.8B on general knowledge and instruction following. Some of that is the parameter advantage, and some of it is specialization: a text-only model has no vision encoder or 201-language vocabulary competing for its 1.17B parameters.

On math, there is nothing to compare against, which is itself informative. LFM2.5-1.2B-Thinking posts 87.96 on MATH-500, 85.60 on GSM8K, and 31.73 on AIME25. The Qwen3.5-0.8B card publishes no math benchmarks at all for the text side. Alibaba positions its smallest model for prototyping, fine-tuning, and lightweight multimodal work rather than competition math, and the omission is consistent with that.

Then the tables turn completely. Qwen3.5-0.8B scores 49.0 on MMMU, 58.3 on MMStar, and 79.3 on RefCOCO in thinking mode. That last number means a sub-1B model can locate the object you describe in an image with high reliability. LFM2.5 scores nothing here because it cannot see. There is no vision variant, so an entire category of on-device work (receipt parsing, screenshot analysis, photo-based Q&A) is simply out of reach for it.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/Gcq_HUYLVC779xOuut2EI.png" alt="Bar chart of average response length in tokens showing LFM2.5-1.2B-Thinking producing shorter outputs than Qwen3-1.7B in thinking mode across GPQA Diamond, MMLU-Pro, IFEval, Multi-IF, MATH-500 and AIME2025" caption="Average response length across benchmarks, lower is better. LFM2.5-1.2B-Thinking averages 4,543 tokens per response against 5,976 for Qwen3-1.7B in thinking mode; on AIME2025 the gap is 9,133 vs 18,538. (Chart: Liquid AI, LFM2.5-1.2B-Thinking model card)" >}}

One more efficiency point from Liquid's card, shown in the chart above. Reasoning models bill you in tokens, and every thinking token costs battery and latency on device. Against Qwen3-1.7B in thinking mode (the previous-generation Qwen, half again LFM's size, which Liquid picked as its benchmark rival), LFM2.5-1.2B-Thinking averages 4,543 tokens per response versus 5,976, and on AIME2025 it needs 9,133 tokens where Qwen3-1.7B burns 18,538. Qwen3.5-0.8B does not have published equivalents, so read this as evidence that LFM's traces are disciplined rather than as a direct verdict on today's opponent.

## Thinking styles: always on vs on demand

The two models treat reasoning differently at a product level, not just a benchmark level.

LFM2.5-1.2B-Thinking always reasons. Every response starts with a trace inside `<think>` tags, and the recommended sampling is strikingly conservative: temperature 0.05, top_k 50, repetition penalty 1.05. Reasoning chains derail at high temperature, so Liquid keeps the tap nearly closed. If your workload is casual chat, that always-on trace is overhead, and Liquid itself points you to the instruct variant instead.

Qwen3.5-0.8B makes thinking optional, and off by default. You toggle it with Ollama's `--think` flag (or `/set think` inside a session) or with `enable_thinking` in vLLM; the Qwen3 generation's `/think` soft switch is gone in Qwen3.5. The recommended non-thinking text settings sit at the opposite extreme from Liquid's: temperature 1.0 with a presence penalty of 2.0. There is also a documented catch. The model card states plainly that the 0.8B is more prone to entering thinking loops than its larger siblings, where the reasoning chain fails to terminate. Use thinking mode selectively on the smallest Qwen, and cap your token budget when you do.

Do not swap sampling configs between these two models. Each one's settings are close to pathological for the other.

## Running them locally

Both are a one-command install. For LFM2.5-1.2B-Thinking with llama.cpp:

```bash
# Download the Q4_0 GGUF (696 MB)
huggingface-cli download LiquidAI/LFM2.5-1.2B-Thinking-GGUF \
  LFM2.5-1.2B-Thinking-Q4_0.gguf --local-dir .

# Chat with Liquid's recommended sampling
llama-cli -m LFM2.5-1.2B-Thinking-Q4_0.gguf \
  --temp 0.05 --top-k 50 --repeat-penalty 1.05
```

For Qwen3.5-0.8B, Ollama is the easiest path and includes the vision stack:

```bash
# 1.0 GB download, 256K context, image input supported
ollama run qwen3.5:0.8b

# Enable reasoning when you need it
ollama run qwen3.5:0.8b --think
```

LFM2.5 also loads in LM Studio if you prefer a GUI, and both models ship MLX builds for Apple Silicon. If you are unsure which runtime fits your setup, our [Ollama vs LM Studio vs llama.cpp guide](/posts/ollama-vs-lm-studio-vs-llama-cpp/) covers the tradeoffs, and the [Q4 vs Q5 vs Q8 quantization guide](/posts/gguf-quantization-levels-q4-q5-q8/) explains what the quant levels cost you in quality.

## Which one should you run?

### Which is better for math and reasoning?

LFM2.5-1.2B-Thinking, and it is not close. An 87.96 on MATH-500 from a model that fits in 696 MB is the standout result in this weight class, and the instruction-following scores (88.42 IFEval) mean it actually does what you asked on the way to the answer. If your workload is tool calling, multi-step instructions, or math word problems on a device, this is the pick.

### Which is better for documents, images, and long context?

Qwen3.5-0.8B, by an equally wide margin. It is the only option here that accepts images and video, its 262,144-token context window is eight times LFM's 32,768, and the Gated DeltaNet architecture keeps that context affordable in memory. Feed it a phone photo of a receipt or 200 pages of text and it has a real chance; LFM2.5 cannot enter either contest. The 201-language coverage also makes it the default for any multilingual routing task.

### Does the license matter?

It might. Qwen3.5-0.8B ships under Apache 2.0, full stop: fine-tune it, embed it in a commercial product, no questions. LFM2.5 uses Liquid's LFM Open License v1.0, which makes the weights freely downloadable but is not an OSI-approved open-source license. If you are shipping a commercial product, read the terms before you commit to the Liquid stack.

### Can you just run both?

Yes, and it is a legitimate pattern. Together they need about 1.7 GB, which fits alongside the OS on any recent phone-class device and trivially on a laptop. Route math, tool calls, and structured reasoning to LFM2.5-1.2B-Thinking; route anything involving an image, a long document, or a non-European language to Qwen3.5-0.8B. At these sizes the cost of loading a second model is lower than the cost of forcing one model to do a job it was not built for.

## Verdict

This comparison has a cleaner answer than most of our head-to-heads, including the [Jamba Reasoning 3B vs Phi-4-mini-reasoning matchup](/posts/jamba-reasoning-3b-vs-phi-4-mini-reasoning/) one weight class up: the two models barely overlap. LFM2.5-1.2B-Thinking is a reasoning engine that happens to be tiny. Qwen3.5-0.8B is a Swiss-army perception model that happens to reason. Pick by workload: math and agents go to Liquid, eyes and long memory go to Alibaba, and at a combined 1.7 GB you can skip the choice entirely. For where these two sit against everything else we track, the [best small language models in 2026](/posts/best-small-language-models-2026/) pillar has the full field.

## Sources

- [LFM2.5-1.2B-Thinking model card (Hugging Face)](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)
- [LFM2.5-1.2B-Thinking-GGUF (Hugging Face)](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF)
- [LFM2.5-1.2B-Thinking: On-Device Reasoning Under 1GB (Liquid AI blog)](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb)
- [Qwen3.5-0.8B model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Qwen3.5 on the Ollama library](https://ollama.com/library/qwen3.5)
