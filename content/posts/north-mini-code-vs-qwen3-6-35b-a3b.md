---
title: "North Mini Code vs Qwen3.6-35B-A3B: Which Local Coding MoE Wins?"
date: 2026-07-10
draft: false
tags: ["comparison", "cohere", "qwen", "agentic-coding", "small-models"]
categories: ["small-ai-models"]
description: "North Mini Code 1.0 vs Qwen3.6-35B-A3B head-to-head: coding benchmarks, GGUF sizes, VRAM headroom on a 24 GB GPU, context windows, and which 3B-active MoE to pick."
slug: "north-mini-code-vs-qwen3-6-35b-a3b"
---

If you have a 24 GB GPU and want a local coding agent, the choice in mid-2026 comes down to North Mini Code vs Qwen3.6-35B-A3B. Both are sparse Mixture of Experts models that activate roughly 3 billion parameters per token, both carry an Apache 2.0 license, and both were trained specifically for agentic software engineering. They land within 3 GB of each other at 4-bit quantization, they run through the same tooling, and Cohere's own model card benchmarks one directly against the other. We have covered each one individually ([North Mini Code](/posts/run-north-mini-code-locally/), [Qwen3.6-35B-A3B](/posts/run-qwen3-6-35b-a3b-locally/)); this post puts them side by side.

## The Short Answer

Qwen3.6-35B-A3B is the stronger model on almost every coding benchmark, and not by a little: 73.4 vs 67.6 on SWE-bench Verified, 49.5 vs 40.2 on SWE-bench Pro, and 51.5 vs 36.0 on Terminal-Bench v2. If your hardware fits the 22.1 GB Q4 file with enough left over for context, run Qwen.

North Mini Code earns its place on tighter memory budgets. Its Q4 weights are 19.2 GB, which leaves 3 GB more KV-cache headroom on a 24 GB card, and it is the rare specialist that beats Qwen anywhere at all (SciCode, 38.2 vs 35.8). It is also text-only, so you are not paying for a vision encoder you will never use in a terminal.

## Spec Comparison

| | North Mini Code 1.0 | Qwen3.6-35B-A3B |
|---|---|---|
| Total parameters | 30B | 35B |
| Active per token | 3B | 3B |
| Experts | 128 (8 active) | 256 (8 routed + 1 shared) |
| Attention | Sliding-window + global (3:1) | Gated DeltaNet + Gated Attention (3:1) |
| Context window | 256K | 262K native, 1M via YaRN |
| Max output | 64K tokens | standard |
| Modalities | Text only | Text, image, video |
| Thinking | Interleaved thinking for tool use | On by default, preserved across turns |
| License | Apache 2.0 | Apache 2.0 |
| Released | June 2026 | April 2026 |
| Q4_K_M GGUF | 19.2 GB | 22.1 GB |

The headline similarity is the 3B active count. Because decode speed in an MoE is driven by activated parameters rather than the full weight file, both models generate tokens at roughly the pace of a dense 3B model. Neither one will feel faster than the other in day-to-day use on the same GPU; the differences are in quality, memory, and what happens at long context.

## Benchmarks: North Mini Code vs Qwen3.6-35B-A3B

Cohere's own model card includes a direct comparison against Qwen3.6, which makes this unusually easy to source. These numbers come from that table (Qwen's figures match its official model card, so neither vendor is disputing the other's results):

| Benchmark | North Mini Code | Qwen3.6-35B-A3B |
|---|---|---|
| SWE-bench Verified | 67.6 | 73.4 |
| SWE-bench Pro | 40.2 | 49.5 |
| Terminal-Bench v2 | 36.0 | 51.5 |
| Terminal-Bench Hard | 31.1 | 35.0 |
| LiveCodeBench v6 | 70.3 | 80.4 |
| SciCode | 38.2 | 35.8 |

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/62668f725fb8d521d94d8451/xR7kZ3X9RKEZrbgD6hpG1.png" alt="Benchmark table from the North Mini Code model card comparing it against Qwen3.6-35B-A3B, Poolside XS.2, Gemma4 26B-A4B, and Devstral Small 2 on Terminal-Bench, SWE-bench, SciCode, and LiveCodeBench" caption="Cohere's published comparison, including Qwen3.6-35B-A3B in the next column over. Source: North Mini Code 1.0 model card, Cohere Labs (Apache 2.0)." >}}

Qwen sweeps the agentic rows. The Terminal-Bench v2 gap (51.5 vs 36.0) is the most damaging one for Cohere, because terminal-driven agent work is exactly what North Mini Code was marketed for. The SWE-bench Pro spread of more than 9 points is similarly hard to argue around: on the harder, contamination-resistant repository tasks, Qwen simply solves more issues.

Independent measurement agrees with the vendor numbers. Artificial Analysis scores Qwen3.6-35B-A3B at 35.2 on its Coding Index against 33.4 for North Mini Code, a narrower gap than the SWE-bench spread suggests, and it clocked North Mini Code at roughly 199 output tokens per second on Cohere's API. The same analysis puts North Mini Code's Intelligence Index at 27.6, with the note that it falls off sharply on non-coding agentic tasks. That matches the design brief: this is a coding tool, not a general assistant.

North Mini Code's SciCode win (38.2 vs 35.8) is worth a sentence rather than a footnote. SciCode tests code generation for problems drawn from real research papers across the natural sciences. If your workload is numerical or research code rather than web-app issue fixing, the smaller model has a real, measured edge there.

Outside coding, it is not close. Qwen3.6-35B-A3B posts 85.2 on MMLU-Pro and 86.0 on GPQA Diamond, and it accepts image and video input. North Mini Code does not publish general-knowledge scores, and Artificial Analysis's independent testing confirms there is no hidden generalist inside it.

## Memory: Where the 5B Difference Actually Bites

MoE memory cost is set by total parameters, not active ones, so the 30B vs 35B split shows up directly in the file sizes:

| Quantization | North Mini Code | Qwen3.6-35B-A3B |
|---|---|---|
| ~3-bit | 14.2 GB (UD-Q3_K_M) | 13.7 GB (UD-IQ3_S) |
| 4-bit | 19.2 GB (UD-Q4_K_M) | 22.1 GB (Q4_K_M) |
| 8-bit | 32.4 GB (Q8_0) | 36.9 GB (Q8_0) |

On a 24 GB card (RTX 3090 or 4090), both fit at Q4, but the margins differ. Qwen's 22.1 GB file leaves under 2 GB for KV cache once the runtime's own overhead is counted, which caps you at a few thousand tokens of context before layers start spilling to system RAM. North Mini Code's 19.2 GB leaves roughly 4 to 5 GB free, enough for a 16K+ context window entirely on-GPU. For agent sessions that accumulate long tool transcripts, that headroom is the difference between staying fast and paging.

There is a counterweight on the Qwen side: its Gated DeltaNet layers are linear-attention blocks whose cache grows far more slowly with sequence length than standard attention. North Mini Code bounds its cache differently, with sliding-window attention on three of every four layers. In practice both handle long context better than a plain transformer, but Qwen's hybrid is the reason it can advertise a 1M-token mode (via YaRN) at all. At the context lengths a single consumer GPU can realistically hold, the file-size difference matters more than the cache-growth difference.

At the 3-bit tier the sizes nearly converge, so 16 GB GPU owners do not get a meaningful memory reason to prefer either. Quality at 3-bit degrades on the hardest benchmark problems for both; our [GGUF quantization levels explainer](/posts/gguf-quantization-levels-q4-q5-q8/) covers what each step down costs.

{{< figure src="https://images.pexels.com/photos/4573596/pexels-photo-4573596.jpeg" alt="Two RTX 2080 graphics cards side by side on a white surface" caption="Both models target the same hardware class: a single 24 GB consumer GPU. Photo: Nana Dua via Pexels (free license)." >}}

## Setup: The Gap Has Closed

When we published our North Mini Code guide, running it meant building llama.cpp from an open pull request because the `cohere2moe` architecture was not in any release. That is no longer true. The PR merged into llama.cpp in June 2026, and Ollama now hosts the model natively:

```bash
# North Mini Code (19 GB download, Q4_K_M default)
ollama run north-mini-code-1.0

# Qwen3.6-35B-A3B (Q4_K_M default)
ollama run qwen3.6:35b-a3b
```

Both expose OpenAI-compatible endpoints through Ollama, so pointing Aider, OpenCode, or any other agent frontend at either one is the same one-line change. Ollama also lists MLX-format tags for North Mini Code (`mlx-nvfp4` at 20 GB), which closes the Apple Silicon gap our original guide flagged.

Sampling defaults differ and are worth setting explicitly. Cohere recommends temperature 1.0 with top_p 0.95 for North Mini Code. The Qwen team recommends temperature 0.6 with top_p 0.95 for thinking-mode coding, and its thinking is on by default, so expect `<think>` blocks in raw output unless you disable them.

## Which One Should You Run?

Run Qwen3.6-35B-A3B if you want the most capable local coding agent in this weight class and can live with the tighter fit at Q4. It wins the repository-level and terminal benchmarks by wide margins, doubles as a competent general model when you are not coding, and its preserved thinking across turns genuinely helps on multi-step refactors.

Run North Mini Code if VRAM headroom is your constraint, if you want longer on-GPU context out of a 24 GB card, or if your work leans toward scientific computing where it holds the only benchmark lead. A 19 GB coding specialist that decodes like a 3B model is still a strong tool; it just is not the class leader.

If neither fits your hardware, this class of model wants 20 GB or more of memory. Our [best small language models in 2026](/posts/best-small-language-models-2026/) roundup covers what to run below that line, down to models that fit in 8 GB.

## Common Questions

### Is North Mini Code better than Qwen3.6-35B-A3B for coding?

No, not on the published evidence. Qwen3.6-35B-A3B leads on SWE-bench Verified (73.4 vs 67.6), SWE-bench Pro (49.5 vs 40.2), Terminal-Bench v2 (51.5 vs 36.0), and LiveCodeBench v6 (80.4 vs 70.3). North Mini Code's single benchmark win is SciCode (38.2 vs 35.8), which covers scientific programming. Cohere's own model card publishes this comparison, so the gap is not a matter of cherry-picked vendor numbers.

### Which model is faster on the same GPU?

They are effectively tied. Both activate about 3B parameters per token, so decode throughput on identical hardware is close to a dense 3B model for each. North Mini Code's slightly smaller total size can help marginally when memory pressure forces offloading, but on a card that fits either model fully, you will not notice a speed difference.

### Can both run in Ollama and LM Studio now?

Yes. Qwen3.6-35B-A3B has had native support since launch, and North Mini Code gained it once llama.cpp merged the `cohere2moe` architecture in June 2026. The `north-mini-code-1.0` tag on Ollama's library has seen over 20K pulls. LM Studio inherits the same support through its bundled llama.cpp.

### Do I need 24 GB of VRAM for these models?

For comfortable Q4 use, yes. Both offer roughly 14 GB 3-bit quants that run on 16 GB cards with reduced quality and context, and both run on CPU at a few tokens per second if you have 24 GB or more of system RAM. The offloading approach in our [low-VRAM Windows GPU guide](/posts/run-local-llms-low-vram-windows-gpu/) applies to either model.

## Sources

- [North Mini Code 1.0 model card (Hugging Face)](https://huggingface.co/CohereLabs/North-Mini-Code-1.0)
- [Qwen3.6-35B-A3B model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [North Mini Code analysis (Artificial Analysis)](https://artificialanalysis.ai/articles/north-mini-code-cohere-s-small-coding-focused-moe-model)
- [llama.cpp pull request #24260: cohere2-MoE architecture support](https://github.com/ggml-org/llama.cpp/pull/24260)
- [Ollama library: north-mini-code-1.0](https://ollama.com/library/north-mini-code-1.0)
- [Ollama library: qwen3.6:35b-a3b](https://ollama.com/library/qwen3.6:35b-a3b)
- [Unsloth North-Mini-Code-1.0 GGUF quantizations](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF)
- [Unsloth Qwen3.6-35B-A3B GGUF quantizations](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
