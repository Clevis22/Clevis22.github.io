---
title: "Granite 4.1 8B vs LFM2.5-8B-A1B: Dense vs MoE at 8B for Local Inference"
date: 2026-07-18
draft: false
tags: ["granite", "liquid-ai", "moe", "comparison", "small-models"]
categories: ["comparisons"]
description: "A head-to-head comparison of IBM Granite 4.1 8B and Liquid AI's LFM2.5-8B-A1B: dense vs mixture-of-experts, tool-calling benchmarks, speed, licensing, and which 8B model to run locally."
bluesky: "Granite 4.1 8B runs all 8B per token; LFM2.5-8B-A1B fires just 1.5B. Both are ~5GB tool-callers, but on the one benchmark they share, the dense model wins."
slug: "granite-4-1-8b-vs-lfm2-5-8b-a1b"
---

If you have 8GB of RAM and want a capable local tool-caller, Granite 4.1 8B and LFM2.5-8B-A1B are two of the more interesting options, and they arrive at the same size from opposite directions. Comparing Granite 4.1 8B vs LFM2.5-8B-A1B is really a comparison of two design philosophies: IBM's model is a dense 8B transformer that runs every parameter on every token, while Liquid AI's is a sparse mixture-of-experts that holds 8.3B in memory but fires only about 1.5B per token. Both land near 5GB at Q4, both do one-command Ollama installs, and both are pitched at agentic use. This post works out what that difference actually buys you, and which one to run locally.

## Granite 4.1 8B vs LFM2.5-8B-A1B: the specs side by side

These are the facts both vendors state plainly. Two rows carry most of the decision: the architecture split, and the license.

| | Granite 4.1 8B | LFM2.5-8B-A1B |
|---|---|---|
| Total parameters | 8B | 8.3B |
| Active per token | 8B (dense) | ~1.5B (MoE) |
| Architecture | Dense decoder-only (GQA, RoPE, SwiGLU) | Hybrid MoE: 18 gated-convolution + 6 GQA attention layers |
| Context window | 128K (512K from training) | 128K |
| Answering style | Instruct (direct) | Reasoning-only (chain-of-thought every turn) |
| Vision | No | No |
| Ollama download (Q4) | 5.3 GB | 5.2 GB |
| Languages | 12 named | 10 named |
| License | Apache 2.0 | LFM Open License v1.0 |
| Released | April 29, 2026 | May 28, 2026 |

Neither has a vision encoder, so if you need image input, this comparison ends before it starts and you want something like Ministral 3 8B instead (covered in our [Granite 4.1 8B vs Ministral 3 8B](/posts/granite-4-1-8b-vs-ministral-3-8b/) breakdown). What's left is a clean contrast between a dense model and an MoE at nearly identical memory footprint, which is a rare enough matchup to be worth studying carefully.

## Dense vs mixture-of-experts: where the two designs actually differ

The number that matters for an MoE is not the total parameter count, it is the active count. LFM2.5-8B-A1B stores all 8.3B weights but routes each token through a small subset of the network, so a forward pass only touches roughly 1.5B parameters. Granite 4.1 8B has no routing: every token runs the full 8B. That single fact sets the whole tradeoff.

Memory tracks the total, speed tracks the active count. Both models weigh about 5GB at Q4_K_M and sit comfortably on an 8GB machine, so on the memory axis they are near-twins. On the speed axis they are not. Granite pays for 8B of compute per token; LFM pays for about 1.5B. On a CPU, where you are compute-bound rather than bandwidth-bound, that gap is large, which is exactly why Liquid positions LFM2.5-8B-A1B as the fastest model in its size class on consumer processors. The catch is that the dense model has a higher ceiling per token: all 8B of Granite's parameters contribute to every prediction, where LFM only ever brings a fraction of its network to bear on any given step.

{{< figure src="https://images.pexels.com/photos/30832094/pexels-photo-30832094.jpeg" alt="Railway tracks converging and splitting through switch points at dusk" caption="A mixture-of-experts router switches each token onto a different subset of the network, the way track points send a train down one branch. LFM2.5-8B-A1B routes; Granite 4.1 8B runs every path at once. (Photo: BREAKS OUT, Pexels)" >}}

LFM's architecture pushes the efficiency further than routing alone. Only 6 of its 24 layers use full grouped-query attention; the other 18 are Liquid's gated short-convolution blocks, which are cheaper to run than attention at inference time. Granite is a conventional dense transformer stack (grouped-query attention, RoPE positions, SwiGLU feed-forwards) with no such substitution. If you want the full setup details for either, our [Granite 4.1 guide](/posts/run-ibm-granite-4-1-locally/) and [LFM2.5-8B-A1B guide](/posts/run-lfm2-5-8b-a1b-locally/) each cover installation and quant choices.

## The benchmarks that actually line up

IBM and Liquid publish almost entirely different benchmark suites, so most of a side-by-side table would be decorative. IBM leads with MMLU, GSM8K, HumanEval and MBPP; Liquid leads with the AA-Omniscience Index, MATH500, AIME25 and an agentic Tau² score. GSM8K against MATH500 is not a comparison, and HumanEval against nothing is not either. Two rows do line up, and they happen to be the two that matter most for the agentic use both models target.

| Benchmark | Granite 4.1 8B | LFM2.5-8B-A1B |
|---|---:|---:|
| BFCL v3 (tool calling) | 68.27 | 64.36 |
| IFEval (instruction following) | 87.06 | 91.84 |

The split is clean and slightly surprising. LFM2.5-8B-A1B is explicitly built and marketed for reliable tool calling on consumer hardware, yet on BFCL v3, the standard function-calling benchmark, Granite's 68.27 edges LFM's 64.36 (Liquid's own card lists it as 64.79 in a second table, so read it as roughly 64 to 65 either way; both numbers trail Granite). Go the other way on IFEval and LFM's 91.84 comfortably beats Granite's 87.06, which is where Liquid's large-scale instruction-following RL clearly paid off. So the dense model is marginally the better raw tool-caller, and the MoE is the better instruction-follower, which is close to the opposite of what the marketing on each side would lead you to expect.

Read both numbers with the usual caution. Granite's evaluations run with thinking off, while LFM always reasons before answering, so the two are not measured under identical conditions even when the benchmark name matches. A four-point BFCL gap is real but not decisive, and the only test that settles it is your own tool schema. For where each sits against the wider field, our [best small language models in 2026](/posts/best-small-language-models-2026/) roundup has the broader table.

### Does the MoE really lose at tool calling?

On the published number, narrowly, yes: Granite 4.1 8B scores 68.27 on BFCL v3 against 64.36 for LFM2.5-8B-A1B. But BFCL v3 measures function selection and JSON argument correctness in a single-shot setting, and LFM's real strength shows up on multi-turn agentic tasks instead. On Liquid's Tau² Telecom benchmark, which tests completing a task across several tool-using turns, LFM2.5-8B-A1B scores 88.07, and IBM publishes no comparable multi-turn figure for Granite. So the honest reading is that Granite looks better at picking the right function in one shot, while LFM is built to hold a tool-using conversation together across many steps. Which one wins depends entirely on whether your agent makes one call or twenty.

## Reasoning-only vs instruct: the token tax

LFM2.5-8B-A1B is reasoning-only. Every assistant turn opens with an explicit chain of thought before the final answer, and you cannot turn that off. Liquid's argument for the design is specific to MoE: because each reasoning token only activates 1.5B parameters, the thinking trace is cheap, so you get the quality of deliberate reasoning without paying much of a speed penalty per token. Keep the temperature low (Liquid recommends 0.2 with top_k 80 and a 1.05 repetition penalty) or the chain of thought tends to wander.

Granite 4.1 8B is a plain instruct model. It answers directly, with no forced reasoning trace, and IBM benchmarks it with thinking off. For a tool-calling loop where you want a JSON call back as fast as possible, that directness matters more than the per-token cost: Granite emits the call immediately, while LFM spends a paragraph of reasoning tokens first. The dense model costs more compute per token but produces fewer tokens to first action; the MoE costs less per token but writes more of them. Latency to a usable result is the metric to watch, and it does not always favor the sparser model.

{{< figure src="https://images.pexels.com/photos/6636500/pexels-photo-6636500.jpeg" alt="Macro photograph of a black computer microchip and surrounding circuit components" caption="Same 5GB memory budget, two different bets on how to spend it: full-network compute per token, or a routed fraction of a larger network. (Photo: Sergei Starostin, Pexels)" >}}

## Licensing: Apache 2.0 vs the LFM Open License

This is the row people skip and then regret. Granite 4.1 ships under Apache 2.0, an OSI-approved license with no ambiguity about commercial use, redistribution, or fine-tuning. LFM2.5-8B-A1B ships under the LFM Open License v1.0. The weights are freely downloadable, but it is not an OSI-approved license, and it carries its own terms. If you plan to embed the model in a commercial product, read the LFM license before assuming Apache-style permissions. For a hobby project or internal tooling, neither license will get in your way; for something you ship and charge for, the difference is worth a lawyer's ten minutes.

## Running both locally

Both are a single Ollama command and both expose an OpenAI-compatible API on `localhost`, so swapping between them to test on your own workload costs nothing but disk. If you have not picked a runtime yet, our [Ollama vs LM Studio vs llama.cpp comparison](/posts/ollama-vs-lm-studio-vs-llama-cpp/) covers the tradeoffs.

```bash
# Granite 4.1 8B: 5.3 GB, dense, direct answers
ollama run granite4.1:8b

# LFM2.5-8B-A1B: 5.2 GB, MoE, reasoning-only
ollama run lfm2.5:8b
```

Both default to Q4_K_M in Ollama, the right starting point on an 8GB card. LFM in particular wants Liquid's recommended sampling (`--temp 0.2 --top-k 80 --repeat-penalty 1.05` under llama.cpp) because it is a reasoning model; Granite is less fussy. For what each quantization level costs in quality, see our [Q4 vs Q5 vs Q8 guide](/posts/gguf-quantization-levels-q4-q5-q8/).

On throughput, neither vendor publishes a directly comparable single-user number, and the two figures that exist come from different hardware, so treat them as directional rather than a benchmark. Our own testing put the Granite 8B at roughly 25 to 60 tokens per second on an RTX 4060. Liquid reports server-side numbers (18.5K tokens per second at high concurrency on an H100) that reflect the same MoE sparsity which keeps single-user CPU decode quick. The reliable statement is qualitative: on a CPU, the MoE decodes faster per token; on a GPU with headroom for either, the gap narrows and the dense model's directness can win on latency-to-first-action.

## Which one should you run?

Pick on capability shape, because the scores barely overlap and the two you can compare split one each. Granite 4.1 8B is the better default when the model is a component in a pipeline: it has the higher single-shot tool-calling score, Apache 2.0 licensing with no caveats, and it answers directly without a reasoning tax. LFM2.5-8B-A1B is the better default when you want the fastest decode on a CPU, the strongest instruction-following, and multi-turn agentic behavior, and you are comfortable with a reasoning-only model under a non-OSI license.

### Is LFM2.5-8B-A1B faster than Granite 4.1 8B?

On a CPU, yes, and by a meaningful margin. LFM2.5-8B-A1B activates about 1.5B parameters per token where Granite 4.1 8B runs all 8B, and on compute-bound CPU inference that difference in active parameters translates fairly directly into decode speed. The qualifier is that LFM emits a reasoning trace before every answer, so if you measure time-to-final-answer on a short prompt rather than raw tokens per second, Granite's direct reply can close or reverse the gap. On a GPU with enough VRAM for either model, both run comfortably and the speed difference shrinks.

### Do they need the same amount of RAM?

Effectively yes. At Q4_K_M, Granite 4.1 8B is about 5.3GB on disk and LFM2.5-8B-A1B about 5.2GB, and both fit on an 8GB machine with room for a reasonable context. Because LFM is a mixture-of-experts, its entire 8.3B has to sit in memory even though only 1.5B runs per token, so its footprint is set by the total parameter count, not the active one. The MoE design buys speed, not a smaller memory bill. A 16GB laptop runs either with plenty of headroom for their full 128K context windows.

### Which is the safer choice for a commercial product?

Granite 4.1 8B, on licensing grounds alone. Apache 2.0 is OSI-approved and imposes no restrictions that would surprise a product team, while the LFM Open License v1.0 is a custom license you need to read before shipping. On capability, both are viable, and if your product is agentic you should benchmark both against your own tool schema. But if two models are otherwise close and one carries a standard permissive license, that is the tiebreaker.

For more same-size matchups at 8B, our [Granite 4.1 8B vs Ministral 3 8B](/posts/granite-4-1-8b-vs-ministral-3-8b/) and [LFM2.5-8B-A1B vs ZAYA1-8B](/posts/lfm2-5-8b-a1b-vs-zaya1-8b/) comparisons cover the neighboring options, including another sparse-MoE contender that pushes the active count even lower.

## Sources

- [Introducing the IBM Granite 4.1 family of models — IBM Research](https://research.ibm.com/blog/granite-4-1-ai-foundation-models)
- [Granite 4.1 LLMs: How They're Built — HuggingFace Blog](https://huggingface.co/blog/ibm-granite/granite-4-1)
- [ibm-granite/granite-4.1-8b — HuggingFace](https://huggingface.co/ibm-granite/granite-4.1-8b)
- [granite4.1 on Ollama](https://ollama.com/library/granite4.1)
- [LFM2.5-8B-A1B model card — HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- [LFM2.5-8B-A1B: An Even Better On-Device Mixture of Experts — Liquid AI blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b)
- [lfm2.5 on Ollama](https://ollama.com/library/lfm2.5)
