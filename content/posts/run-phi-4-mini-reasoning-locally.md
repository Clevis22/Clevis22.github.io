---
title: "How to Run Phi-4-mini-reasoning Locally: Microsoft's 3.8B Math Model"
date: 2026-06-08
draft: false
tags: ["phi", "reasoning", "ollama", "local-inference", "small-models"]
categories: ["small-ai-models"]
description: "Step-by-step guide to running Phi-4-mini-reasoning locally with Ollama and llama.cpp: Microsoft's 3.8B model that outscores 7B distillation models on AIME and MATH-500."
slug: "run-phi-4-mini-reasoning-locally"
---

Phi-4-mini-reasoning is a 3.8B model built for one purpose: mathematical reasoning. Microsoft released it in April 2025 as a fine-tuned variant of Phi-4-mini, trained on 150 billion tokens of synthetic math content. The result is a model that scores 94.6% on MATH-500 and 57.5% on AIME 2024 while fitting into a 2.49 GB GGUF file. This guide covers how to run Phi-4-mini-reasoning locally using Ollama and llama.cpp, with the hardware requirements and deployment options you actually need.

{{< figure src="https://images.pexels.com/photos/6238050/pexels-photo-6238050.jpeg" alt="Mathematical equations and formulas written in chalk on a blackboard, representing the kind of multi-step reasoning Phi-4-mini-reasoning was trained to solve" caption="Phi-4-mini-reasoning was trained on 150B tokens of synthetic math problems spanning middle school through PhD-level competition math. Photo: [Monstera Production](https://www.pexels.com/@gabby-k), Pexels" >}}

## What Phi-4-mini-reasoning is (and what it isn't)

The model shares its 3.8B architecture with Phi-4-mini but the training objective diverges sharply. Where the base model is a general-purpose 23-language assistant trained on 5 trillion tokens, Phi-4-mini-reasoning is a math specialist whose training pipeline prioritized step-by-step logical correctness over breadth.

| Spec | Value |
|---|---|
| Parameters | 3.8B |
| Architecture | Dense Transformer (GQA) |
| Context length | 128,000 tokens |
| License | MIT |
| Language | English (primary) |
| Data cutoff | February 2025 |
| Release date | April 2025 |

The training used four stages: mid-training on distilled chain-of-thought data, supervised fine-tuning on curated high-quality solutions, preference optimization via Rollout DPO, and reinforcement learning with verifiable rewards. Each math problem in the dataset had eight independently verified solution rollouts before it was included.

One limitation to state clearly: Phi-4-mini-reasoning was not evaluated on general downstream tasks. General conversation, coding, and non-English text are outside the training distribution. For those use cases, the [Phi-4-mini base model](/posts/run-phi-4-mini-locally/) or a different model is the better choice.

## Benchmark results against larger models

The comparison that makes Phi-4-mini-reasoning interesting is how it lines up against 7-8B models that are specifically reasoning-focused:

| Model | Size | AIME 2024 | MATH-500 | GPQA Diamond |
|---|---|---|---|---|
| o1-mini* | closed | 63.6% | 90.0% | 60.0% |
| DeepSeek-R1-Distill-Qwen-7B | 7B | 53.3% | 91.4% | 49.5% |
| DeepSeek-R1-Distill-Llama-8B | 8B | 43.3% | 86.9% | 47.3% |
| OpenThinker-7B | 7B | 31.3% | 83.0% | 42.4% |
| Bespoke-Stratos-7B | 7B | 20.0% | 82.0% | 37.8% |
| Phi-4-mini (base) | 3.8B | 10.0% | 71.8% | 36.9% |
| **Phi-4-mini-reasoning** | **3.8B** | **57.5%** | **94.6%** | **52.0%** |

*Source: Microsoft HuggingFace model card. Asterisk indicates a closed proprietary model.*

Two results stand out. On MATH-500, Phi-4-mini-reasoning scores 94.6% against o1-mini's 90.0%: a 3.8B open-weights model outscoring a closed frontier system on a standardized benchmark. On AIME 2024, it scores 57.5% against DeepSeek-R1-Distill-Qwen-7B's 53.3%, with half the parameters.

GPQA Diamond (graduate-level multi-domain science reasoning) is more balanced: 52.0% for Phi-4-mini-reasoning versus 60.0% for o1-mini. That gap is meaningful for real science Q&A use cases. GPQA is harder to narrow through targeted fine-tuning than pure math benchmarks.

## Running with Ollama

Ollama provides direct support. The download is 3.2 GB:

```bash
ollama pull phi4-mini-reasoning
ollama run phi4-mini-reasoning
```

Once running, the model generates detailed chain-of-thought before its final answer. This is by design: the training explicitly rewards step-by-step reasoning traces. On complex problems, a full response can run 3,000-6,000 tokens before the final answer appears.

If you're calling via the Ollama API, set `num_predict` high enough to capture the full reasoning chain:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "phi4-mini-reasoning",
  "prompt": "Find all integer solutions to x^2 - 5y^2 = 4.",
  "stream": false,
  "options": { "num_predict": 8192 }
}'
```

For interactive use in the CLI, the default generation limit is usually fine. The `num_predict` setting only matters when you're hitting the limit mid-solution.

## Running with llama.cpp

Bartowski's GGUF collection covers the full quantization range for Phi-4-mini-reasoning. The Q4_K_M variant (2.49 GB) is the practical default.

```bash
# Download Q4_K_M via huggingface-cli
huggingface-cli download bartowski/microsoft_Phi-4-mini-reasoning-GGUF \
  microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf \
  --local-dir ./phi4-mini-reasoning

# Run inference
./llama-cli \
  -m ./phi4-mini-reasoning/microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf \
  -p "Find all integer solutions to x^2 - 5y^2 = 4." \
  -n 8192 \
  --temp 0.7
```

Set `-n` (max new tokens) to at least 4096. On harder problems, reasoning traces expand significantly before reaching the final answer. If the model cuts off mid-solution, increasing `-n` is the first thing to try.

For LM Studio users, the same bartowski GGUF repository is directly importable from within the app's model browser.

## Which quantization to pick

| Quantization | File size | Use when |
|---|---|---|
| Q4_K_M | 2.49 GB | Default for most setups |
| Q5_K_M | 2.85 GB | More precision on edge cases in complex proofs |
| Q6_K | 3.16 GB | Near-lossless; worth it if you have 4 GB free RAM |
| Q8_0 | 4.08 GB | Near full precision; rarely necessary |
| BF16 | 7.68 GB | Full precision; needs 10+ GB RAM |

Q4_K_M is the right starting point. Q6_K is a worthwhile step up if you're running competition-level problems and 670 MB of extra RAM is available. The gap between Q4_K_M and Q8_0 on arithmetic tasks is smaller than the raw size difference suggests.

For a detailed breakdown of what you actually give up at each quant level, the [GGUF quantization explainer](/posts/gguf-quantization-levels-q4-q5-q8/) covers the tradeoffs.

## Running with Python (Transformers)

For programmatic use, the HuggingFace Transformers library works with `transformers>=4.51.3`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-reasoning")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-reasoning",
    torch_dtype="auto",
    device_map="auto"
)

messages = [{"role": "user", "content": "Solve: x^2 - 5y^2 = 4 for integer solutions."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=32768)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Set `max_new_tokens=32768` to avoid cutting off complex solutions mid-reasoning. The model's chain-of-thought traces are longer than typical instruct responses.

{{< figure src="https://images.pexels.com/photos/22690751/pexels-photo-22690751.jpeg" alt="Close-up of complex mathematical equations and formulas written densely on a blackboard" caption="The model generates detailed step-by-step reasoning traces for each problem before producing a final answer. Photo: [Vitaly Gariev](https://www.pexels.com/@silverkblack), Pexels" >}}

## Common questions

**Can Phi-4-mini-reasoning run on a laptop without a GPU?**

Yes. The Q4_K_M GGUF (2.49 GB) needs roughly 4 GB of available RAM and runs on CPU with llama.cpp. Any laptop with 8 GB total RAM can handle it, though throughput on CPU will be lower than GPU. On an M-series Mac, MLX inference will be faster than llama.cpp CPU. For a hardware-specific breakdown, the [Apple Silicon local LLM guide](/posts/run-small-llms-apple-silicon/) covers throughput expectations by chip generation.

**Is the 128K context window usable locally?**

Partially. The model supports 128K tokens in its architecture, but processing very long contexts on CPU or a small GPU gets slow fast. For math use cases, most problems fit well within 8K tokens, so the full 128K range rarely matters in practice. Where it does matter: feeding a full textbook chapter or problem set as context.

**How does this differ from DeepSeek-R1 distillation models?**

Both are reasoning-focused fine-tunes trained with distilled chain-of-thought data, and both prioritize math over general capability. The main difference is size: Phi-4-mini-reasoning is 3.8B versus the 7-8B DeepSeek distills. On the benchmarks in Microsoft's model card, Phi-4-mini-reasoning leads DeepSeek-R1-Distill-Qwen-7B on all three reported metrics: AIME 2024 (57.5% vs 53.3%), MATH-500 (94.6% vs 91.4%), and GPQA Diamond (52.0% vs 49.5%). It also leads DeepSeek-R1-Distill-Llama-8B on each. The practical upside is roughly 40% less RAM for comparable math performance.

**What tasks is it actually good for?**

Structured math pipelines: step-by-step solution generation for tutoring apps, automated verification of symbolic computation output, feeding long mathematical documents through the 128K context window, and formal proof sketching. The chain-of-thought output also functions as a debugging tool; even when the final answer is wrong, the reasoning trace usually shows exactly where the derivation went off track.

## What the model doesn't cover

Phi-4-mini-reasoning is math only. It wasn't trained or evaluated for code generation, factual Q&A, summarization, or multilingual tasks. Microsoft's own documentation marks factual recall and extended conversation as known weak points due to the 3.8B size. For a model that handles general-purpose work at similar memory requirements, the [best small language models comparison](/posts/best-small-language-models-2026/) has per-task recommendations across the current generation.

## Sources

- [microsoft/Phi-4-mini-reasoning — HuggingFace model card](https://huggingface.co/microsoft/Phi-4-mini-reasoning)
- [Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math (arXiv 2504.21233)](https://arxiv.org/abs/2504.21233)
- [phi4-mini-reasoning — Ollama library](https://ollama.com/library/phi4-mini-reasoning)
- [bartowski/microsoft_Phi-4-mini-reasoning-GGUF — HuggingFace](https://huggingface.co/bartowski/microsoft_Phi-4-mini-reasoning-GGUF)
