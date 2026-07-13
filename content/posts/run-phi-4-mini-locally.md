---
title: "How to Run Phi-4-mini Locally: Microsoft's 3.8B Model with 128K Context"
date: 2026-05-18
draft: false
tags: ["phi", "ollama", "gguf", "local-inference", "small-models"]
categories: ["small-ai-models"]
description: "Step-by-step guide to running Phi-4-mini locally with Ollama, llama.cpp, and Python — with verified benchmarks and RAM requirements for Microsoft's 3.8B reasoning model."
slug: "run-phi-4-mini-locally"
---

Phi-4-mini is Microsoft's 3.8B parameter model and one of the most practical models to run locally right now. The GGUF download is 2.49 GB at Q4_K_M, it handles a 128,000-token context window — rare at this scale — and it scores 88.6% on GSM8K math benchmarks while competing against models that score 30 points lower. This guide covers exactly how to run Phi-4-mini locally with Ollama, llama.cpp, and Python's Transformers library, with verified RAM requirements and benchmark context throughout.

{{< figure src="https://images.pexels.com/photos/1472443/pexels-photo-1472443.jpeg" alt="Close-up view of a Raspberry Pi circuit board with microchips and electronic components, representing accessible local AI hardware" caption="Phi-4-mini's Q4_K_M GGUF weighs 2.49 GB — small enough to run on the hardware you already own. Photo: [Alessandro Oliverio](https://www.pexels.com/photo/1472443/), Pexels" >}}

## What Phi-4-mini is

Released in February 2025, Phi-4-mini is a dense decoder-only transformer trained on 5 trillion tokens, with deliberate emphasis on reasoning-dense synthetic data rather than raw text volume. Microsoft tuned it for structured tasks — math, logic, and code — which shows up clearly in its benchmark profile.

| Spec | Value |
|---|---|
| Parameters | 3.8B |
| Architecture | Dense Transformer (GQA) |
| Context length | 128,000 tokens |
| Training tokens | 5 trillion |
| Data cutoff | June 2024 |
| License | MIT |
| Supported languages | 23 |

The 128K context is the spec that stands out at this parameter count. Most models in the 3B class cap at 8K or 32K — which forces RAG pipelines or chunking for anything longer. Phi-4-mini's context handling makes it more directly useful for document-length inputs without additional infrastructure.

## How to run Phi-4-mini with Ollama

Ollama is the lowest-friction path and works on macOS, Linux, and Windows. Requires Ollama 0.5.13 or later.

```bash
# Pull and start an interactive session
ollama pull phi4-mini
ollama run phi4-mini
```

This pulls the 2.5 GB default Q4 quantization. For the OpenAI-compatible API instead of the chat prompt:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi4-mini",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Solve 2x + 3 = 7. Show each step."}
    ]
  }'
```

Any OpenAI-compatible client works against that endpoint with a `base_url` swap. If you are still deciding between Ollama and llama.cpp for your workflow, [our runtime comparison](/posts/ollama-vs-lm-studio-vs-llama-cpp/) covers the tradeoffs in detail.

## How to run Phi-4-mini with llama.cpp and GGUF

Unsloth maintains the full GGUF quantization range at `huggingface.co/unsloth/Phi-4-mini-instruct-GGUF`. File sizes across quantization levels, verified from the model card:

| Quantization | File size | Notes |
|---|---|---|
| Q2_K | 1.68 GB | Lowest quality, minimal memory |
| Q3_K_M | 2.12 GB | Reasonable floor for CPU inference |
| Q4_K_M | 2.49 GB | Recommended starting point |
| Q5_K_M | 2.85 GB | Better quality, modest size increase |
| Q6_K | 3.16 GB | High quality, 6-bit |
| Q8_0 | 4.08 GB | Near-lossless |
| BF16 | 7.68 GB | Full precision |

Q4_K_M is where most people should start. Unsloth's dynamic quantization selectively avoids quantizing attention layers, which preserves more accuracy than a flat 4-bit pass. For a deeper look at what these formats actually trade off against each other, see our [GGUF vs ONNX vs MLX guide](/posts/gguf-vs-onnx-vs-mlx/).

```bash
# Download the Q4_K_M GGUF
huggingface-cli download unsloth/Phi-4-mini-instruct-GGUF \
  Phi-4-mini-instruct-Q4_K_M.gguf --local-dir ./phi4-mini

# Run with llama.cpp CLI
./llama-cli \
  -m ./phi4-mini/Phi-4-mini-instruct-Q4_K_M.gguf \
  -n 512 \
  --chat-template phi4 \
  --prompt "Solve this step by step: 2x + 3 = 7"
```

RAM usage at Q4_K_M: approximately 3.2 GB total (2.49 GB model weights plus ~0.75 GB llama.cpp runtime overhead). Add 1–2 GB on top of that if you push toward the full 128K context window — the KV cache grows with context length. [Our RAM requirements guide](/posts/how-much-ram-for-local-llms/) explains how to estimate total memory for your specific context window target.

## How to run Phi-4-mini with Python

Microsoft's official deployment path uses the Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

messages = [
    {"role": "system", "content": "You are a precise math assistant."},
    {"role": "user", "content": "Solve: 2x + 3 = 7. Show each step."},
]

output = pipe(messages, max_new_tokens=500, temperature=0.0, do_sample=False)
print(output[0]["generated_text"][-1]["content"])
```

One hardware note: Phi-4-mini uses flash attention by default, which requires an NVIDIA A100, A6000, or H100. On a V100 or older GPU, you need to disable it explicitly:

```python
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    attn_implementation="eager",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
```

For vLLM deployments, the model is served with `vllm serve microsoft/Phi-4-mini-instruct`. The model card includes working SGLang and Docker Model Runner configurations as well.

## Benchmark results

Scores from the official HuggingFace model card, comparing Phi-4-mini against its predecessor, same-size competitors, and one larger model:

| Benchmark | Phi-4-mini (3.8B) | Phi-3.5-mini (3.8B) | Llama-3.2-3B | Qwen2.5-7B | GPT-4o-mini |
|---|---|---|---|---|---|
| BigBench Hard (0-shot CoT) | **70.4** | 63.1 | 55.4 | 72.4 | 80.4 |
| MMLU (5-shot) | 67.3 | 65.5 | 61.8 | **72.6** | 77.2 |
| MMLU-Pro (0-shot CoT) | **52.8** | 47.4 | 39.2 | 56.2 | 62.8 |
| GSM8K (8-shot CoT) | **88.6** | — | — | — | — |
| MATH (0-shot CoT) | **64.0** | — | — | — | — |
| ARC Challenge (10-shot) | 83.7 | — | — | — | — |

The math numbers are the headline. 88.6% GSM8K and 64% MATH are results typical of models with 7B–14B parameters. On BigBench Hard — the multi-step reasoning benchmark — Phi-4-mini at 70.4 beats Llama-3.2-3B (55.4) by 15 points and its predecessor Phi-3.5-mini (63.1) by 7, while landing within 2 points of Qwen2.5-7B (72.4) despite being half the size.

MMLU tells a different story. The 67.3% score falls below Qwen2.5-7B's 72.6 and GPT-4o-mini's 77.2 — broad factual recall is not this model's focus. If your workload leans heavily on general knowledge rather than structured reasoning, a larger model or Qwen2.5-7B will score better, but it will also cost you more RAM. For a full comparison across the 3B–8B range, see [our best small language models guide](/posts/best-small-language-models-2026/).

{{< figure src="https://images.pexels.com/photos/5276099/pexels-photo-5276099.jpeg" alt="Blue relay modules with colorful wires on an electronic circuit board, representing embedded and edge computing hardware" caption="At 2.49 GB for Q4_K_M, Phi-4-mini fits on hardware too small to run most 7B models. Photo: [Malte Luk](https://www.pexels.com/photo/5276099/), Pexels" >}}

## Where Phi-4-mini performs best

Math and multi-step reasoning are the real differentiators. 88.6% GSM8K and 64% MATH mean it handles arithmetic logic, equation solving, and chain-of-thought math reliably — not occasionally. Coding assistants that involve numerical logic or algorithm tracing benefit from this directly.

The 128K context makes it practical for tasks where smaller models break down. A 50,000-word document, a full Python codebase, or an extended conversation with long history can all load into context without chunking. That changes what you can ask it to do.

Function calling is natively supported, so it wires into tool-use workflows without prompt engineering workarounds. The catch is that its coding training skews heavily toward Python. Other languages work but expect worse results on anything outside of JavaScript and the major statically-typed languages.

Multilingual coverage spans 23 languages, but performance is uneven below the top tier. English and Chinese get the most reliable results. French, German, Spanish, and Japanese are reasonable. Languages further from the training distribution can produce noticeably weaker outputs.

## Is Phi-4-mini the right model for your use case?

Phi-4-mini is the right choice when you have a 4–8 GB RAM budget and need math, reasoning, or code quality rather than broad factual recall. The Q4_K_M GGUF at 2.49 GB keeps total memory under 4 GB for short contexts, and the 128K ceiling is the largest context window available at this parameter count.

If factual recall and general knowledge matter more than math performance, Qwen2.5-7B at 72.6% MMLU is the better fit — it just needs roughly twice the RAM. If you are RAM-constrained below 3 GB, the tradeoff shifts to smaller models, but at a meaningful benchmark cost.

## Sources

- [microsoft/Phi-4-mini-instruct — HuggingFace model card](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [unsloth/Phi-4-mini-instruct-GGUF — quantization file sizes](https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF)
- [Ollama phi4-mini library page](https://ollama.com/library/phi4-mini)
- [Phi-4 technical report (Microsoft)](https://aka.ms/phi-4-multimodal/techreport)
