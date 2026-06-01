---
title: "Run ZAYA1-8B Locally: The 8B Reasoning MoE Trained on AMD"
date: 2026-06-01
draft: false
tags: ["small-models", "slm", "edge-ai", "zaya1"]
categories: ["small-ai-models"]
description: "How to run ZAYA1-8B locally: a 8.4B-total, 760M-active reasoning MoE from Zyphra, with setup steps, memory requirements, and benchmark numbers."
slug: "run-zaya1-8b-locally"
---

ZAYA1-8B is the model that finally makes a sparse mixture-of-experts worth running on a single workstation. Zyphra released it on May 6, 2026 under Apache 2.0, and the headline is the ratio: 8.4 billion total parameters but only about 760 million active on any given token. That means the compute cost per token is closer to a sub-1B model while the knowledge capacity sits at the 8B scale. If you want to run ZAYA1-8B locally for math, code, and chain-of-thought reasoning, this guide covers the architecture, the realistic memory footprint, and the two deployment paths that actually work today.

It is also one of the first open reasoning models trained end to end on AMD silicon, which matters more for what it proves than for how you deploy it. The weights are on [Hugging Face](https://huggingface.co/Zyphra/ZAYA1-8B) and the curl examples below talk to a standard OpenAI-compatible endpoint, so nothing here depends on owning an AMD card.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/W8bn6ZAocWKFuicjtjesv.png" alt="Bar chart comparing ZAYA1-8B benchmark scores against other open-weight small models" caption="ZAYA1-8B against same-class open-weight models across math, code, knowledge, and instruction-following benchmarks. Source: Zyphra, ZAYA1-8B model card (Apache 2.0)." >}}

## What is ZAYA1-8B?

ZAYA1-8B is a text-only reasoning model built on Zyphra's MoE++ architecture. The config on Hugging Face spells out the shape: 80 layers, a hidden size of 2,048, a vocabulary of 262,272 tokens, and 16 experts with top-1 routing (`moe_router_topk: 1`). Top-1 routing is the aggressive end of the MoE spectrum. Each token is sent to exactly one expert, which is how the active-parameter count stays under a billion while the total climbs to 8.4B.

Three architecture choices do the heavy lifting. The attention layers use Compressed Convolutional Attention (CCA), a variant Zyphra describes as more efficient than standard attention. Expert selection runs through a novel MLP-based router instead of the usual linear router, which Zyphra says improves routing stability. And the network uses learned residual scaling to keep residual-norm growth in check across all 80 layers at negligible parameter cost. The full design is in the [technical report](https://arxiv.org/abs/2605.05365) (arXiv 2605.05365).

The context window is the part most people get wrong from the press coverage. The model config sets `max_position_embeddings` to 131,072, so ZAYA1-8B handles up to 128k tokens. Separately, Zyphra's test-time-compute method, Markovian RSA, budgets around 40k tokens for intermediate chain-of-thought, but that is a sampling configuration, not the model's ceiling.

## Benchmarks: what the numbers actually say

Two sets of numbers float around for this model, and conflating them is the easiest way to overstate it. The first set is the base model scored normally. The second set is the base model with Markovian RSA, a test-time-compute method that spends extra tokens to push reasoning scores higher. Here are the base scores from the official model card:

| Benchmark | ZAYA1-8B (base) | Category |
|---|---|---|
| AIME'26 | 89.1 | Math |
| HMMT Feb.'26 | 71.6 | Math |
| IMO-AnswerBench | 59.3 | Math |
| LiveCodeBench-v6 | 65.8 | Code |
| GPQA-Diamond | 71.0 | Knowledge |
| MMLU-Pro | 74.2 | Knowledge |
| IFEval | 85.58 | Instruction following |
| BFCL-v4 | 39.22 | Agentic / tool use |

An AIME score near 89 from a model with under a billion active parameters is the genuinely surprising result here. The technical report states that the base model "matches or exceeds DeepSeek-R1-0528 on several challenging mathematics and coding benchmarks." With Markovian RSA layered on top, Zyphra reports 91.9% on AIME'25 and 89.6% on HMMT'25, and claims the latter edges out Claude 4.5 Sonnet and GPT-5-High on HMMT'25 (89.6 versus 88.3). Treat the RSA figures as the result of a heavier inference recipe, not the out-of-the-box behavior you get from a plain `generate` call.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/f5tbexK3BumixnJuBZxo_.png" alt="Time-to-capability line chart showing ZAYA1-8B reasoning performance versus compute" caption="Time-to-capability: how ZAYA1-8B's reasoning scores scale with test-time compute against larger baselines. Source: Zyphra, ZAYA1-8B model card (Apache 2.0)." >}}

## How much memory do you need to run ZAYA1-8B locally?

This is where the MoE design trips people up. The active-parameter count drives how fast generation runs, but it does not reduce how much memory you need. All 8.4B parameters have to sit in memory because the router can pick any expert for any token. So size your hardware against the full 8.4B, not the 760M.

Rough footprints, scaled from the 8.4B total parameter count:

| Precision | Approx. weights size | Fits comfortably on |
|---|---|---|
| bf16 (full) | ~17 GB | 24 GB GPU (RTX 4090/3090) or a 32 GB Mac |
| Q8_0 | ~9 GB | 12 GB GPU or 16 GB unified memory |
| Q4_K_M | ~5 GB | 8 GB GPU or 16 GB RAM on CPU |

Add a couple of gigabytes on top for the KV cache, more if you push toward the full 128k context. These are estimates from the parameter count, so verify against the actual file you download. For a deeper breakdown of what each quant level costs you in quality, see our guide to [GGUF quantization levels](/posts/gguf-quantization-levels-q4-q5-q8/), and if VRAM is your constraint, our notes on [running local LLMs on a low-VRAM Windows GPU](/posts/run-local-llms-low-vram-windows-gpu/) apply directly here.

## Running ZAYA1-8B with vLLM (the supported path)

ZAYA uses a custom architecture (`ZayaForCausalLM`), so it does not load on a stock install of either Transformers or vLLM yet. Zyphra ships forks with the `zaya` code merged. This is the path that works today and the one Zyphra recommends.

Install vLLM from Zyphra's branch:

```bash
pip install "vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr"
```

Then serve the model with an OpenAI-compatible endpoint:

```bash
vllm serve Zyphra/ZAYA1-8B --port 8010 \
  --mamba-cache-dtype float32 --dtype bfloat16 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser zaya_xml
```

The `--mamba-cache-dtype` flag is a hint about what is under the hood: the CCA layers keep a recurrent-style cache that vLLM handles separately from the standard KV cache, and Zyphra recommends keeping it in float32 for stability. The `--reasoning-parser qwen3` and `--tool-call-parser zaya_xml` flags tell vLLM how to split the model's thinking traces and tool calls out of the raw output.

Once it is up, query it like any OpenAI endpoint:

```bash
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Zyphra/ZAYA1-8B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Prove that the square root of 2 is irrational."}
    ]
  }'
```

Zyphra's recommended sampling settings are temperature 1.0, top_p 0.95, top_k -1 for general use, and temperature 0.6 for agentic or coding work. Reasoning models are sensitive to sampling, so start there before tuning.

## What about Ollama and llama.cpp?

If you would rather avoid Python and run a single quantized file, the honest answer is that GGUF support is experimental right now. Community GGUF quants exist on Hugging Face, but ZAYA1-8B support in llama.cpp is still an open feature request ([ggml-org/llama.cpp#22776](https://github.com/ggml-org/llama.cpp/issues/22776)) at the time of writing. The custom CCA attention and the top-1 MoE router are not in a released llama.cpp build yet, so a quant that downloads fine may not run on stock binaries.

If you want to try the GGUF route anyway, community uploads document commands like:

```bash
llama-cli --hf-repo lainlives/ZAYA1-8B-GGUF --hf-file ZAYA1-8B-Q4_K_M.gguf \
  -p "The meaning of life is" -ngl 99 -c 4096
```

Expect to build llama.cpp from a branch with the ZAYA changes until the work lands in master. For a primer on which runtime fits your setup once support stabilizes, our [Ollama vs LM Studio vs llama.cpp comparison](/posts/ollama-vs-lm-studio-vs-llama-cpp/) walks through the trade-offs. For a model in this class that runs on plain llama.cpp today, [Jamba Reasoning 3B](/posts/run-jamba-reasoning-3b-locally/) is the safer pick while ZAYA1-8B support matures.

## Is ZAYA1-8B worth running over a dense 8B model?

For reasoning-heavy workloads, the case is strong. You get the math and code scores of a much larger reasoning model at the generation speed of a sub-1B model, because only one 760M-parameter expert fires per token. On a CPU or a modest GPU, that speed difference is the whole point of a sparse MoE: a dense 8B model touches every parameter on every token, while ZAYA1-8B touches a small slice.

The catch is memory and tooling. You still pay for 8.4B parameters of RAM or VRAM, and right now the clean deployment path is vLLM from a fork rather than a one-line Ollama pull. If your work is general chat or you need it running on a phone-class device, a dense 3B model is simpler. If you are doing local math, competition-style reasoning, or agentic coding and you have at least 8 GB of memory to spare, ZAYA1-8B is one of the most capable things you can run at this footprint. It is a strong entry on our list of the [best small language models in 2026](/posts/best-small-language-models-2026/).

## Quick reference

- Total parameters: 8.4B; active per token: ~760M (16 experts, top-1 routing)
- Context window: 128k tokens (131,072 positions)
- License: Apache 2.0
- Best deployment today: vLLM from `github.com/Zyphra/vllm@zaya1-pr`
- Minimum practical memory: ~5 GB at Q4, ~17 GB at bf16
- Trained on 1,024 AMD Instinct MI300X nodes with Pensando Pollara interconnect

## Sources

- [Zyphra/ZAYA1-8B model card (Hugging Face)](https://huggingface.co/Zyphra/ZAYA1-8B)
- [ZAYA1-8B config.json](https://huggingface.co/Zyphra/ZAYA1-8B/raw/main/config.json)
- [Zyphra: ZAYA1-8B announcement](https://www.zyphra.com/post/zaya1-8b)
- [ZAYA1-8B technical report (arXiv 2605.05365)](https://arxiv.org/abs/2605.05365)
- [llama.cpp feature request #22776 (ZAYA support)](https://github.com/ggml-org/llama.cpp/issues/22776)
- [lainlives/ZAYA1-8B-GGUF (community quants)](https://huggingface.co/lainlives/ZAYA1-8B-GGUF)
