---
title: "Granite 4.1 8B vs Ministral 3 8B: Which 8B Model Should You Run Locally?"
date: 2026-07-16
draft: false
tags: ["granite", "mistral", "comparison", "local-inference", "small-models"]
categories: ["small-ai-models"]
description: "A head-to-head comparison of IBM Granite 4.1 8B and Mistral's Ministral 3 8B: benchmarks, tool calling, vision, context windows, and which one to run locally."
bluesky: "Granite 4.1 8B and Ministral 3 8B are both Apache 2.0, both ~8B, both around 6GB. Their model cards share almost no benchmarks. Here's how to actually choose."
slug: "granite-4-1-8b-vs-ministral-3-8b"
---

If you have 8GB of VRAM and want a capable local model, Granite 4.1 8B and Ministral 3 8B are the two most obvious Apache 2.0 candidates. Both land around 6GB on disk, both do tool calling, both run with one Ollama command. Comparing Granite 4.1 8B vs Ministral 3 8B looks like it should be a simple matter of reading two benchmark tables side by side. It isn't: IBM and Mistral publish almost entirely different benchmarks, and the two numbers they nominally share are measured differently enough to be useless. This post works out what can actually be compared, and what that leaves you with.

## Granite 4.1 8B vs Ministral 3 8B: the specs that are directly comparable

These are the facts both vendors state plainly, and they're where the real decision lives.

| | Granite 4.1 8B | Ministral 3 8B |
|---|---|---|
| Parameters | 8B dense | 8.4B LM + 0.4B vision = 8.8B |
| Architecture | Dense decoder-only (GQA, RoPE, SwiGLU) | Dense, with vision encoder |
| Context window | 128K on the model card (512K claimed in training) | 256K |
| Vision | No | Yes |
| Variants | Instruct | Base, instruct, reasoning |
| Ollama download | 5.3 GB | 6.0 GB |
| Languages | 12 named | 11 named |
| License | Apache 2.0 | Apache 2.0 |
| Released | April 29, 2026 | December 2, 2025 |
| Published tool-calling score | BFCL v3: 68.27 | None published |

Two rows decide most cases. Ministral 3 ships a 0.4B vision encoder in every size and Granite 4.1 has no vision at all, so if you need to pass an image, the comparison ends there. And Ministral 3 ships a reasoning variant that emits chain-of-thought by default, where Granite 4.1 has only an instruct model.

The context row needs a caveat. IBM's research blog headlines a 512K context window for the 8B, but the published model card lists a 131,072 token (128K) sequence length, and Ollama serves 128K. The 512K figure describes what the final training phase extended the model to, not what you get when you pull it. Ministral's 256K is stated consistently across Mistral's card, the Ollama entry, and the technical report.

{{< figure src="https://images.pexels.com/photos/9606950/pexels-photo-9606950.jpeg" alt="Close-up of a neatly arranged set of wrenches in an open toolbox" caption="Tool calling is the clearest split between these two models: IBM publishes a BFCL v3 score, Mistral publishes none. (Photo: Anastasia Shuraeva, Pexels)" >}}

## Why you can't just compare the benchmark tables

Here is what each vendor reports for its 8B, taken from the official model cards:

| Granite 4.1 8B (instruct) | Score | Ministral 3 8B | Score |
|---|---|---|---|
| MMLU (5-shot) | 73.84 | MMLU (5-shot, base) | 76.1 |
| GSM8K (8-shot) | 92.49 | MATH Maj@1 (instruct) | 87.6 |
| Minerva Math (0-shot CoT) | 80.10 | MATH CoT 2-shot (base) | 62.6 |
| HumanEval (pass@1) | 85.37 | LiveCodeBench (reasoning) | 61.6 |
| MBPP (pass@1) | 87.30 | AIME 2025 (reasoning) | 78.7 |
| GPQA (0-shot CoT) | 41.96 | GPQA Diamond (reasoning) | 66.8 |
| ArenaHard | 68.98 | Arena Hard (instruct) | 50.9 |
| IFEval Avg | 87.06 | WildBench (instruct) | 66.8 |
| BFCL v3 | 68.27 | — | — |
| MTBench Avg | 8.61 | MM MTBench (instruct) | 8.08 |

Read down those columns and you'll notice the overlap is close to zero. GSM8K against MATH Maj@1 is not a comparison. HumanEval against LiveCodeBench is not a comparison. MTBench against MM MTBench is not a comparison, because the latter is the multimodal variant.

Even the two rows that share a name don't share a method. Granite's 68.98 and Ministral's 50.9 are both called ArenaHard, but ArenaHard scores move substantially with the judge model and the benchmark version, and neither card pins down which it used. IBM's charts also carry the footnote "All evaluations are done with Thinking=Off." Treating 68.98 vs 50.9 as an 18-point Granite win is exactly the mistake this table invites.

The MMLU row has a subtler problem. Granite's 73.84 is from the instruct model; Mistral only reports MMLU for the base model, at 76.1. Instruction tuning usually moves MMLU by a point or two in either direction, so the honest reading is that these two models are within noise of each other on general knowledge, not that Ministral leads by 2.3.

One more reason to hold these numbers loosely: IBM's own two sources disagree with each other. The 8B model card lists HumanEval at 85.37, while IBM's HuggingFace blog post puts it at 87.20 for the same instruct model. The gap is small and doesn't change any recommendation here, but if a vendor's own pages differ by two points on a benchmark they both chose to publish, cross-vendor comparisons at that resolution are not meaningful.

### Does Ministral 3 8B really beat Granite on GPQA?

Not in a way you can bank. Ministral's 66.8 is GPQA Diamond (the harder subset) from the reasoning variant, which spends tokens on chain-of-thought before answering. Granite's 41.96 is full GPQA from the instruct model with thinking off. A reasoning model scoring higher on a harder subset than a non-reasoning model does on an easier one is a genuine signal that Ministral's reasoning variant is strong, but it is not evidence that Ministral's instruct model beats Granite's. Those two numbers describe different models doing different work.

## Tool calling: the one place with a real answer

IBM benchmarks Granite 4.1 on BFCL v3, the Berkeley Function Calling Leaderboard, and publishes 68.27 for the 8B. Mistral describes Ministral 3 as having "native tool use" and publishes no function-calling benchmark for it at any size.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/6658c911e238275ea9efc339/0eQZvn83EJBg-dZT-Uoo0.png" alt="Bar chart of BFCL V3 tool-calling scores showing Granite-4.1-8B at 68.3, above Gemma-4-26B-A4B-it at 67.8 and Qwen3-30B-A3B-Instruct at 65.1" caption="Granite 4.1 8B on BFCL v3 (68.3), against Gemma 4 and Qwen 3 competitors. Ministral 3 does not appear in IBM's comparison set. Source: IBM Granite / HuggingFace" >}}

Worth reading that chart carefully in both directions. Granite-4.1-8B (68.3) edges Gemma-4-26B-A4B-it (67.8) and Qwen3-30B-A3B-Instruct-2507 (65.1), which have three to four times its total parameters and therefore need far more memory. But both of those are MoEs activating 4B and 3B parameters per token, fewer than Granite's dense 8B, so this is a win on memory footprint rather than on compute per token. IBM's footnote also states all evaluations ran with thinking off.

Ministral 3 is not in the chart at all. Vendors pick flattering comparison sets, so its absence isn't evidence about Ministral either way. It just means nobody has published the number that would settle this.

If you're building an agent loop where the model picks functions and emits valid JSON arguments, Granite is the model with evidence behind it. That's not the same as Granite being better at tool calling; it's that one vendor measured and the other didn't. Our [Granite 4.1 setup guide](/posts/run-ibm-granite-4-1-locally/) has a working tool-calling example if you want to try it against your own schema, which is the only test that really counts.

## Long context: 128K with data beats 256K without

Granite's other measured advantage is long-context retrieval. IBM publishes RULER scores for the base models:

| Model | 32K | 64K | 128K |
|---|---|---|---|
| granite-4.1-3b-base | 75.0 | 66.6 | 58.0 |
| granite-4.1-8b-base | 83.6 | 79.1 | 73.0 |
| granite-4.1-30b-base | 85.2 | 84.6 | 76.7 |

The 8B holds 73.0 at 128K, which is a real number showing real degradation: you lose about 10 points going from 32K to 128K, and you should plan for that rather than assume a full-context prompt works as well as a short one.

Mistral publishes no long-context retrieval benchmark for Ministral 3's 256K window. So Ministral advertises twice the context and IBM is the only one of the two showing what happens out there. Note also that IBM's own RULER table stops at 128K despite the 512K training claim, which is consistent with the model card's 128K sequence length.

## Running both

Both are one command, and both expose an OpenAI-compatible API, so swapping between them to test on your own workload is cheap. See our [Ollama vs LM Studio vs llama.cpp comparison](/posts/ollama-vs-lm-studio-vs-llama-cpp/) if you haven't picked a runtime.

```bash
# Granite 4.1 8B: 5.3 GB, 128K context, text only
ollama run granite4.1:8b

# Ministral 3 8B: 6.0 GB, 256K context, vision capable
ollama run ministral-3:8b
```

Ministral takes an image directly:

```bash
ollama run ministral-3:8b "What does this chart show?" /path/to/chart.png
```

Both default to Q4_K_M in Ollama, which is the right starting point on an 8GB card. If you have headroom and want to know what you're giving up at each level, our [GGUF quantization guide](/posts/gguf-quantization-levels-q4-q5-q8/) covers the tradeoffs.

Throughput is the one area where neither vendor publishes anything useful. Our own testing put the Granite 8B at roughly 25 to 60 tokens per second on an RTX 4060, and Ministral 3 8B in the same broad range on comparable hardware. Because these came from different machines on different days, they are not a benchmark, and you shouldn't pick between the two on that basis. Both are dense models of near-identical size, so expect near-identical speed on the same card.

## Which one should you actually run?

Pick on capability shape rather than scores, because the scores don't line up. Granite 4.1 8B is the better default for pipelines that call tools or read long documents, since those are the two things IBM actually measured. Ministral 3 8B is the better default if you need image input or a reasoning mode, since Granite offers neither at any size. On plain text chat the two are close enough that only your own prompts will separate them.

**When is Granite 4.1 8B the better pick?**

Choose Granite when the model is a component in a pipeline rather than a chat partner. It's the one with a published tool-calling score, the one with published long-context retrieval data, and it's half a gigabyte smaller on disk. If you're wiring up function calls, structured JSON output, or RAG over long documents, you're choosing the model whose vendor measured the things you care about.

**When is Ministral 3 8B the better pick?**

Choose Ministral when you need image input or hard reasoning. The vision encoder is not optional on Granite's side, it simply doesn't exist, so any workload touching screenshots, charts, or scanned pages goes to Ministral by default. The reasoning variant is the other draw: 78.7 on AIME 2025 and 61.6 on LiveCodeBench are strong for 8B, and Granite has no equivalent mode. The cost is tokens, since the reasoning variant thinks before every answer. Our [Ministral 3 setup guide](/posts/run-ministral-3-locally/) covers switching between the instruct and reasoning variants in LM Studio.

**What if you just want a good local chat model?**

Pull both and try them on your own prompts. On general knowledge they're within noise of each other, and the benchmark tables can't separate them because they barely overlap. Ministral gives you three variants to pick from and vision for 700MB more; Granite gives you a smaller download and better-documented behaviour under load. For a wider field of options at this size, see our [best small language models of 2026](/posts/best-small-language-models-2026/) roundup.

The broader lesson generalises past these two models. When two vendors report disjoint benchmark suites, the comparison table you build from their cards is mostly decorative. Pick on capability shape (vision or not, reasoning variant or not, measured tool calling or not), then benchmark the survivors on your own task.

## Sources

- [Introducing the IBM Granite 4.1 family of models — IBM Research](https://research.ibm.com/blog/granite-4-1-ai-foundation-models)
- [Granite 4.1 LLMs: How They're Built — HuggingFace Blog](https://huggingface.co/blog/ibm-granite/granite-4-1)
- [ibm-granite/granite-4.1-8b — HuggingFace](https://huggingface.co/ibm-granite/granite-4.1-8b)
- [granite4.1 on Ollama](https://ollama.com/library/granite4.1)
- [Introducing Mistral 3 — Mistral AI](https://mistral.ai/news/mistral-3)
- [mistralai/Ministral-3-8B-Instruct-2512 — HuggingFace](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- [mistralai/Ministral-3-8B-Base-2512 — HuggingFace](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512)
- [Ministral 3 technical report — arXiv:2601.08584](https://arxiv.org/abs/2601.08584)
- [ministral-3 on Ollama](https://ollama.com/library/ministral-3)
