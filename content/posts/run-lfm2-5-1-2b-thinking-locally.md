---
title: "How to Run LFM2.5-1.2B-Thinking Locally: On-Device Reasoning Under 1GB"
date: 2026-05-28
draft: false
tags: ["liquid-ai", "reasoning", "edge-ai", "local-inference", "small-models"]
categories: ["small-ai-models"]
description: "A practical guide to running LFM2.5-1.2B-Thinking locally: Liquid AI's 1.2B reasoning model fits in under 1GB and runs on a phone CPU, with llama.cpp, LM Studio, and MLX support."
slug: "run-lfm2-5-1-2b-thinking-locally"
---

LFM2.5-1.2B-Thinking is a 1.2-billion-parameter reasoning model from Liquid AI that runs entirely on-device and fits in about 900 MB of memory once quantized. Released on January 20, 2026, it generates explicit chain-of-thought traces before answering, the same pattern used by far larger reasoning models, but small enough to run on a phone CPU. This guide covers how to run LFM2.5-1.2B-Thinking locally with llama.cpp, LM Studio, and Transformers, what the benchmark numbers mean for a model this size, and where it fits against other sub-2B options.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/KfNudLXnOZxAhlLp_1QVo.png" alt="Bar chart comparing LFM2.5-1.2B-Thinking against Qwen3-1.7B, Granite-4.0-1B, Gemma 3 1B and Llama 3.2 1B across GPQA Diamond, MMLU-Pro, BFCLv3, IFEval, IFBench and Multi-IF" caption="LFM2.5-1.2B-Thinking benchmark scores against other sub-2B models. (Chart: Liquid AI, LFM2.5-1.2B-Thinking model card)" >}}

## What is LFM2.5-1.2B-Thinking?

It is the reasoning-tuned variant of Liquid AI's LFM2.5-1.2B, a small model built for on-device deployment. "Thinking" means the model is trained to write out an internal reasoning trace (inside `<think>` tags) before it commits to a final answer. That extra step is what lets a 1.2B model handle multi-step math and logic problems that would otherwise be out of reach at this scale.

The headline specs:

| Property | Value |
|---|---|
| Parameters | 1.17B |
| Architecture | 16 layers (10 LIV convolution blocks + 6 GQA attention blocks) |
| Context length | 32,768 tokens |
| Vocabulary | 65,536 |
| Knowledge cutoff | Mid-2024 |
| Languages | English, Arabic, Chinese, French, German, Japanese, Korean, Spanish |
| License | LFM Open License (open weights) |

The architecture is the interesting part. Instead of a stack of pure attention layers, LFM2.5 uses a hybrid design: most of the network is short-range gated convolution (Liquid's "LIV" blocks), with only six grouped-query attention blocks mixed in. Convolution is cheaper than attention at inference time, which is a large part of why the model decodes fast on CPUs and mobile NPUs. If you want background on why architecture choices like this matter for edge devices, our [Liquid AI Apollo write-up](/posts/ai-in-your-pocket-liquid-ai-apollo/) covers the company's on-device approach in more depth.

## Benchmarks: what a thinking trace buys you

The clearest way to understand this model is to compare it against its own instruct sibling. Same weights base, same size, the only difference is the reasoning training.

| Benchmark | Thinking | Instruct | What it measures |
|---|---|---|---|
| MATH-500 | 87.96 | 63.20 | Competition math |
| AIME25 | 31.73 | 14.00 | Hard olympiad math |
| GSM8K | 85.60 | 64.52 | Grade-school math |
| Multi-IF | 69.33 | 60.98 | Multi-turn instruction following |
| BFCLv3 | 56.97 | 49.12 | Function/tool calling |
| GPQA Diamond | 37.86 | 38.89 | Graduate science Q&A |

The math jump is dramatic: MATH-500 goes from 63.20 to 87.96, and AIME25 more than doubles from 14.00 to 31.73. GPQA Diamond barely moves, which makes sense, that benchmark rewards stored knowledge more than step-by-step reasoning, and a 1.2B model only holds so many facts.

Against other small models, the picture is competitive at its weight class. Liquid's own numbers put LFM2.5-1.2B-Thinking at 49.65 on MMLU-Pro and 88.42 on IFEval, ahead of Granite-4.0-1B, Gemma 3 1B IT, and Llama 3.2 1B Instruct across most of the suite. The nearest real rival is Qwen3-1.7B in thinking mode, which is a larger model: Qwen edges ahead on MMLU-Pro (56.68) and AIME25 (36.27), while LFM2.5 wins on instruction following (IFEval 88.42 vs 71.65) and uses fewer tokens to get there.

That last point matters more than it looks. On the same benchmark suite, LFM2.5-1.2B-Thinking averages 4,543 tokens per response versus 5,976 for Qwen3-1.7B. Shorter reasoning traces mean less compute and lower latency per answer, which is exactly what you care about when the model is running on battery power. For how these scores sit against the broader field of small models in 2026, see our [best small language models comparison](/posts/best-small-language-models-2026/).

## How to run LFM2.5-1.2B-Thinking locally

There is day-one support across the common local runtimes: llama.cpp, MLX, vLLM, ONNX Runtime, and LM Studio. Pick based on your hardware and how much you want to touch a terminal.

### With LM Studio (easiest)

LM Studio is the no-terminal path. Open the Discover tab and search for "LFM2.5-1.2B-Thinking". Download the GGUF build (the Q4_0 quant is around 696 MB on disk), load it, and chat. LM Studio also exposes an OpenAI-compatible API on port 1234, so you can point existing tools at it without code changes. Because the model emits its reasoning inside `<think>` tags, LM Studio will fold the trace into a collapsible block and show you only the final answer by default.

### With llama.cpp (fastest on CPU)

Liquid publishes an official GGUF repo. Grab a quant and run it:

```bash
# Download the Q4_0 GGUF (~696 MB)
huggingface-cli download LiquidAI/LFM2.5-1.2B-Thinking-GGUF \
  LFM2.5-1.2B-Thinking-Q4_0.gguf --local-dir .

# Run an interactive chat
llama-cli -m LFM2.5-1.2B-Thinking-Q4_0.gguf \
  --temp 0.05 --top-k 50 --repeat-penalty 1.05 \
  -p "A train leaves at 2pm going 60mph. Another leaves at 3pm going 80mph. When does the second catch the first?"
```

Liquid recommends a low temperature (0.05) with `top_k 50` and a light repetition penalty for the thinking variant. Reasoning models are sensitive to sampling, high temperature tends to derail the chain-of-thought, so keep it conservative.

If you have not used llama.cpp before, our [Ollama vs LM Studio vs llama.cpp guide](/posts/ollama-vs-lm-studio-vs-llama-cpp/) walks through installing it and explains the tradeoffs between the three runtimes.

### With Transformers (for development)

For Python integration or fine-tuning experiments, the model runs in bfloat16 through Hugging Face Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is C. elegans?"}],
    add_generation_prompt=True, return_tensors="pt", tokenize=True,
).to(model.device)

output = model.generate(
    input_ids, do_sample=True, temperature=0.05, top_k=50,
    repetition_penalty=1.05, max_new_tokens=512,
)
print(tokenizer.decode(output[0], skip_special_tokens=False))
```

On an Apple Silicon Mac, the MLX build is the better choice for speed and battery life. Our [guide to running small LLMs on Apple Silicon](/posts/run-small-llms-apple-silicon/) covers MLX setup if you want to go that route.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/4ODY8nGws22vICfcMTxNx.png" alt="Bar charts of LFM2.5-1.2B-Thinking prefill, decode and memory on AMD Ryzen AI Max+ 395 and Snapdragon 8 Elite for Galaxy CPUs, compared to Granite-4.0-H-1B and Qwen3-1.7B" caption="CPU inference benchmarks (4-bit, 1k prefill, 250 decoded tokens). On a Samsung Galaxy S25 Ultra CPU the model decodes at 70 tok/s in 720 MB. (Chart: Liquid AI, LFM2.5-1.2B-Thinking model card)" >}}

## How fast is it on real hardware?

This is where a sub-1GB reasoning model earns its place. Liquid's measured numbers, using 4-bit quantization, a 1k-token prefill and 250 decoded tokens:

| Device | Inference | Decode (tok/s) | Memory |
|---|---|---|---|
| AMD Ryzen AI Max+ 395 (CPU, llama.cpp) | CPU | 237 | ~853 MB |
| Samsung Galaxy S25 Ultra (Snapdragon 8 Elite) | CPU | 70 | 720 MB |
| Qualcomm Snapdragon Gen4 (ROG Phone 9 Pro) | NPU | 82 | 0.9 GB |
| AMD Ryzen AI 9 HX 370 | NPU | 57 | 1.6 GB |

The phone numbers are the real story. Seventy tokens per second on a Galaxy S25 Ultra CPU, in 720 MB of RAM, is faster than most people read, and it leaves the NPU and GPU free. Two years ago a reasoning model that wrote out its working would have meant a round trip to a data centre. This one runs in airplane mode.

## Should you use LFM2.5-1.2B-Thinking?

**Is it good enough to replace a cloud reasoning model?**

For self-contained reasoning tasks: math word problems, structured tool calls, multi-step instructions, it holds up well for its size. An 87.96 on MATH-500 is genuinely useful. What it cannot do is substitute for a large model's breadth of knowledge. The mid-2024 cutoff and 1.2B parameter budget mean it will miss recent facts and struggle on knowledge-heavy questions (its 37.86 GPQA Diamond reflects that). Treat it as a reasoning engine, not an encyclopedia. Pair it with retrieval if you need current or domain-specific facts.

**When should you pick the Instruct variant instead?**

If your workload is chat, summarization, or simple Q&A, the instruct version is faster and cheaper because it skips the thinking trace. The reasoning overhead only pays off on problems that genuinely need multiple steps. Running both and routing by task type is a reasonable pattern on capable hardware.

**What quantization should you run?**

Q4_0 is the sweet spot for this model: it is what Liquid benchmarked, it fits under 1 GB, and quality loss at 4-bit on a model this small is modest. If you have memory headroom and want maximum fidelity, an 8-bit GGUF is available. Our [Q4 vs Q5 vs Q8 quantization guide](/posts/gguf-quantization-levels-q4-q5-q8/) covers exactly what you trade away at each level.

A note on licensing: LFM2.5 ships under Liquid AI's LFM Open License, which makes the weights freely downloadable. It is not an OSI-approved open-source license like Apache 2.0, so if you plan to ship it inside a commercial product, read the terms first rather than assuming Apache-style permissions.

## Sources

- [LFM2.5-1.2B-Thinking model card (Hugging Face)](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)
- [LFM2.5-1.2B-Thinking-GGUF (Hugging Face)](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF)
- [LFM2.5-1.2B-Instruct model card (Hugging Face)](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- [LFM2.5-1.2B-Thinking: On-Device Reasoning Under 1GB (Liquid AI blog)](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb)
- [Introducing LFM2.5: The Next Generation of On-Device AI (Liquid AI blog)](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
