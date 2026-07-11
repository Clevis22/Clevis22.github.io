---
title: "LFM2.5-8B-A1B vs ZAYA1-8B: Which 8B MoE Reasoning Model Should You Run Locally?"
date: 2026-07-11
draft: false
tags: ["small-models", "slm", "edge-ai", "liquid-ai", "zaya1", "mixture-of-experts", "comparison"]
categories: ["small-ai-models"]
description: "LFM2.5-8B-A1B vs ZAYA1-8B: two 8B-class sparse MoE reasoning models compared on benchmarks, generation speed, memory footprint, deployment, and licensing."
slug: "lfm2-5-8b-a1b-vs-zaya1-8b"
---

LFM2.5-8B-A1B vs ZAYA1-8B is the most evenly matched pairing in local AI right now, at least on paper. Both are sparse mixture-of-experts models released in May 2026. Both hold roughly 8.3 to 8.4 billion total parameters. Both are reasoning models that emit a chain of thought before answering, and both fit in about 5 GB at Q4. Yet they are built for almost opposite jobs, and picking the wrong one for your workload means leaving most of what you paid for (in RAM) on the table. We covered each model separately in our [LFM2.5-8B-A1B guide](/posts/run-lfm2-5-8b-a1b-locally/) and our [ZAYA1-8B local setup guide](/posts/run-zaya1-8b-locally/); this post puts them side by side.

The short version: LFM2.5-8B-A1B is the agent. ZAYA1-8B is the mathematician. And only one of them runs with a single Ollama command today.

## LFM2.5-8B-A1B vs ZAYA1-8B: specs at a glance

| Property | LFM2.5-8B-A1B | ZAYA1-8B |
|---|---|---|
| Developer | Liquid AI | Zyphra |
| Released | May 28, 2026 | May 6, 2026 |
| Total parameters | 8.3B | 8.4B |
| Active per token | ~1.5B | ~760M |
| Architecture | 24 layers: 18 gated LIV convolution + 6 GQA attention, MoE feed-forward | 80 layers, Compressed Convolutional Attention, 16 experts with top-1 routing |
| Context window | 128K tokens | 128K tokens (131,072 positions) |
| Reasoning | Always on (chain of thought every turn) | Always on |
| Training | 38T tokens + large-scale RL | Trained end to end on 1,024 AMD MI300X nodes |
| License | LFM Open License v1.0 | Apache 2.0 |
| Weights at Q4 | ~4.8-5.2 GB (official GGUF) | ~5 GB (estimated; no stock llama.cpp support) |
| Runs on Ollama today | Yes (`lfm2.5:8b`) | No (vLLM fork required) |

The parameter symmetry hides a real architectural difference. Liquid activates about 1.5B parameters per token across a shallow 24-layer stack built mostly from cheap convolution blocks. Zyphra goes deeper and sparser: 80 layers, but each token is routed to exactly one of 16 experts, so only about 760M parameters fire. Two different answers to the same question of how to buy 8B-scale knowledge at small-model speed.

## What each model is actually for

Liquid AI built LFM2.5-8B-A1B to be a personal assistant that lives on a laptop. The training effort went into instruction following, multi-step tool calling, and, unusually, calibration: knowing when it does not know. On the AA-Omniscience benchmark, which penalizes confident wrong answers, its non-hallucination rate of 63.47% is the best Liquid measured in its size class, and its overall index of -24.70 is roughly half the hallucination penalty of Qwen3.5-4B or Gemma-4-E4B. Its model card openly recommends against using it for hard math or knowledge recall without retrieval.

ZAYA1-8B is the mirror image. Zyphra trained it as a reasoning specialist for mathematics and code, and the numbers are startling for something with under a billion active parameters. On AIME'25 it scores 88.3 in a single rollout, and 91.9 with Zyphra's Markovian RSA test-time-compute scheme layered on top. The announcement claims the RSA-boosted model edges out Claude 4.5 Sonnet and GPT-5-High on HMMT'25 (89.6 versus 88.3). Those RSA numbers come from a heavier inference recipe rather than a plain generate call, but even the single-rollout scores put it in frontier-model territory on competition math.

{{< figure src="https://framerusercontent.com/images/ilZyjY37r0kf5zzJvEXuBcL0PTc.png" alt="Bar chart comparing ZAYA1-8B with and without Markovian RSA against DeepSeek-R1, Claude 4.5 Sonnet, Gemini 2.5 Pro, Qwen3-Thinking, DeepSeek-V3.2 and GPT-5-High on AIME'25, HMMT'25 and LiveCodeBench-v6" caption="ZAYA1-8B (orange) against models up to 671B total parameters on math and code benchmarks. Hatched segments show the Markovian RSA boost. (Chart: Zyphra, ZAYA1-8B announcement)" >}}

## Benchmarks head-to-head

Comparing published numbers across two model cards needs care, because vendors do not run identical suites. The table below sticks to benchmarks where both sides report a score on the same version, then lists each model's specialty results separately.

| Benchmark | LFM2.5-8B-A1B | ZAYA1-8B | Winner |
|---|---:|---:|---|
| IFEval (instruction following) | 91.84 | 85.58 | LFM2.5 |
| BFCL v4 (tool calling) | 48.50 | 39.22 | LFM2.5 |
| AIME'25 (competition math) | 42.53 | 88.3 | ZAYA1, by a mile |

The AIME gap is the single most useful number in this comparison. A 46-point spread on the same competition math benchmark, between two models of the same total size, tells you these are not interchangeable. ZAYA1-8B more than doubles LFM2.5's score.

The specialty numbers reinforce the split. LFM2.5-8B-A1B scores 88.07 on Tau² Telecom, a multi-turn agentic benchmark where it beats models with more than twice its active parameter count, and 88.76 on MATH500 (respectable, though MATH500 is far easier than AIME). ZAYA1-8B posts 65.8 on LiveCodeBench-v6, 71.0 on GPQA-Diamond, and 74.2 on MMLU-Pro, none of which Liquid reports for its model.

So the benchmark story is clean. Instruction following and tool calling go to Liquid. Math, code, and graduate-level knowledge questions go to Zyphra, and not by small margins.

## Deployment: one command vs a vLLM fork

This is where the comparison stops being close.

LFM2.5-8B-A1B shipped with day-one support in llama.cpp, Ollama, LM Studio, MLX, vLLM, and SGLang. Getting it running is one line:

```bash
ollama run lfm2.5:8b
```

That pulls roughly 5 GB and serves an OpenAI-compatible API. Official GGUFs from Q4_0 (about 4.8 GB) up to F16 sit on Hugging Face, and Apple Silicon users get official MLX 4-bit and 8-bit builds. Whatever runtime you already use, it works there; our [Ollama vs LM Studio vs llama.cpp comparison](/posts/ollama-vs-lm-studio-vs-llama-cpp/) can help you pick.

ZAYA1-8B uses a custom architecture (`ZayaForCausalLM` with Compressed Convolutional Attention and an MLP-based router) that stock inference engines do not implement. The supported path is Zyphra's own vLLM fork:

```bash
pip install "vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr"
vllm serve Zyphra/ZAYA1-8B --port 8010 \
  --mamba-cache-dtype float32 --dtype bfloat16 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser zaya_xml
```

That works, but it is a Python environment and a GPU-oriented serving stack, not a double-click. Worse, the llama.cpp feature request for ZAYA1 support ([#22776](https://github.com/ggml-org/llama.cpp/issues/22776)) has been closed as "not planned," so native GGUF support is not on the roadmap. Community GGUF files exist, but they do not run on stock llama.cpp builds. If your definition of local inference is Ollama or LM Studio on a laptop, ZAYA1-8B is effectively unavailable to you right now, whatever its benchmark scores say.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/LX3oIXQeDm51eaLQs64an.png" alt="Line chart of LFM2.5-8B-A1B GPU inference throughput by concurrency on an H100, compared against Gemma-4-E2B-IT, gpt-oss-20b, Qwen3.5-4B and Granite 4.0 H Tiny" caption="LFM2.5-8B-A1B output throughput by concurrency on a single H100 under SGLang, reaching ~18.5K tokens/s at high concurrency. (Chart: Liquid AI, LFM2.5-8B-A1B model card)" >}}

### How much RAM do you need to run either model?

Both models need the full weight set resident in memory, because a router can pick any expert for any token. At Q4 that means roughly 5 GB of weights plus a couple of gigabytes for KV cache and runtime, so an 8 GB machine handles either at moderate context and 16 GB is comfortable. The active-parameter count changes speed, not footprint: memory bandwidth still has to serve the full model over time, but per-token compute tracks the 1.5B or 760M slice. Our guide to [how much RAM local LLMs need](/posts/how-much-ram-for-local-llms/) covers why bandwidth, not capacity, usually sets your token rate.

### Which model is better for agents and tool calling?

LFM2.5-8B-A1B, without much of a contest. It leads BFCL v4 by nine points, its Tau² Telecom score of 88.07 is the standout agentic result in the sub-10B class, and its instruction-following scores beat models several times larger. It is also far less likely to hallucinate a plausible-sounding answer when a tool call fails. ZAYA1-8B ships tool-calling support in its vLLM stack, but a 39.22 on BFCL v4 says tool use was not the training priority.

### Which model is better for math and coding?

ZAYA1-8B, and the margin is enormous. An 88.3 single-rollout AIME'25 against LFM2.5's 42.53 is the kind of gap you normally see between different weight classes, not within one. If your workload is competition-style math, algorithmic problem solving, or code generation where correctness matters more than convenience, ZAYA1-8B is one of the strongest things you can run in 5 GB of weights, provided you can live with the vLLM-fork deployment.

### Which one is faster?

Per token, ZAYA1-8B should decode faster in principle: 760M active parameters is roughly half of LFM2.5's 1.5B. In practice the comparison is muddier. LFM2.5's convolution-heavy stack is engineered for CPU and consumer-GPU decode, and Liquid publishes measured numbers (18.5K tokens/s aggregate on an H100, class-leading CPU decode). Zyphra's speed story is mostly architectural (top-1 routing, a compressed attention cache from CCA) and its supported runtime targets data-centre GPUs. On a MacBook or a mid-range Windows box, the model you can actually run efficiently is the faster one, and today that is LFM2.5.

## Verdict: two specialists, one easy answer for most people

If you want a general-purpose local assistant that follows instructions, drives tools, and admits when it does not know something, run LFM2.5-8B-A1B. It installs in one command on every mainstream runtime, and its weaknesses (hard math, deep knowledge recall) are exactly the things you should be delegating to retrieval or a bigger model anyway. Note the license: the LFM Open License v1.0 is open-weights but not OSI-approved, so read it before shipping a commercial product.

If you need serious mathematical or coding reasoning fully offline and you are comfortable maintaining a Python serving stack, ZAYA1-8B offers frontier-adjacent math performance at a fraction of the memory of anything comparable, under a clean Apache 2.0 license. It is a research-grade tool with research-grade ergonomics.

For most readers the deployment story decides it before the benchmarks do. Until ZAYA1-8B lands in a runtime normal people use, LFM2.5-8B-A1B is the 8B MoE to have on your machine, with ZAYA1-8B as the one to watch. Both earn their spots in our [best small language models in 2026](/posts/best-small-language-models-2026/) roundup, in different rows.

## Sources

- [LFM2.5-8B-A1B model card (Hugging Face)](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- [LFM2.5-8B-A1B: An Even Better On-Device Mixture of Experts (Liquid AI blog)](https://www.liquid.ai/blog/lfm2-5-8b-a1b)
- [Zyphra/ZAYA1-8B model card (Hugging Face)](https://huggingface.co/Zyphra/ZAYA1-8B)
- [ZAYA1-8B announcement (Zyphra)](https://www.zyphra.com/post/zaya1-8b)
- [lfm2.5 on Ollama](https://ollama.com/library/lfm2.5)
- [llama.cpp feature request #22776 (ZAYA1 support, closed)](https://github.com/ggml-org/llama.cpp/issues/22776)
