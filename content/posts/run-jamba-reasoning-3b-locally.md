---
title: "How to Run Jamba Reasoning 3B Locally: AI21's Hybrid SSM-Transformer With a 256K Context"
date: 2026-05-27
draft: false
tags: ["jamba", "reasoning", "ssm", "local-inference", "small-models"]
categories: ["small-ai-models"]
description: "A practical guide to running AI21's Jamba Reasoning 3B locally with Ollama and llama.cpp: a hybrid Mamba/Transformer 3B model with a 256K context window in under 2 GB at Q4."
slug: "run-jamba-reasoning-3b-locally"
---

Jamba Reasoning 3B is AI21 Labs' open-weight reasoning model, released October 8, 2025 under Apache 2.0. It is a 3B-parameter hybrid of Mamba (state-space) and Transformer attention layers with a 256K-token context window, the first sub-4B model that can actually fit a quarter-million tokens into memory on a laptop. This guide covers what makes the architecture different, how to run Jamba Reasoning 3B locally with Ollama and llama.cpp, and where its benchmark numbers actually land against Llama 3.2 3B, Gemma 3 4B, and Granite 4.0 Micro.

{{< figure src="https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B-GGUF/resolve/main/assets/Benchmark%20Performance%20-%20Jamba%20Reasoning%203B.png" alt="Bar chart comparing Jamba Reasoning 3B against Gemma 3 4B, Llama 3.2 3B, and Granite 4.0 Micro on MMLU-Pro, Humanity's Last Exam, and IFBench" caption="Jamba Reasoning 3B vs other sub-4B models on MMLU-Pro, Humanity's Last Exam, and IFBench. (Image: AI21 Labs model card, Apache 2.0)" >}}

## What is Jamba Reasoning 3B?

The interesting part of Jamba Reasoning 3B is the architecture, not just the parameter count. Standard transformers compute attention over every token pair, which costs quadratically in context length. State-space models (SSMs) like Mamba sidestep that by processing the sequence as a recurrence, which gives linear-time inference and constant memory regardless of context.

AI21's Jamba family interleaves the two. The full breakdown from the model card:

- 28 total layers: **26 Mamba layers and 2 Transformer attention layers**
- 20 multi-query attention heads (20 query, 1 KV)
- 64K vocabulary
- 256K-token native context window (AI21 advertises stable behaviour up to 1M)
- BF16 weights
- Trained for reasoning tasks, with thinking traces compatible with the DeepSeek-R1 reasoning parser

The "two attention layers across the whole stack" detail is what makes the long context tractable. Almost all the work happens in Mamba layers, which scale linearly with context. The handful of attention layers carry the long-range token-to-token comparisons that pure SSMs handle poorly. The result is a model that processes 256K tokens with a memory footprint closer to a standard 32K-context dense model.

## Hardware requirements

The GGUF release ships eleven quantization variants. The two endpoints from AI21's own GGUF repo:

| Quant | File size | Approx RAM needed | Notes |
|---|---|---|---|
| Q4_K_M | 1.93 GB | ~3 GB | Best balance for most hardware |
| F16 | 6.4 GB | ~8 GB | Full precision, GPU recommended |

Q4_K_M is the right default. A 1.93 GB model file means Jamba Reasoning 3B fits inside the RAM budget of a Raspberry Pi 5 with 8 GB, an iPhone Pro, or any laptop made in the last five years. If you are deciding between quantization tiers, our [GGUF quantization levels guide](/posts/gguf-quantization-levels-q4-q5-q8/) walks through the trade-offs.

If you want long context, budget more memory. A 256K-token KV cache is not free even with the hybrid architecture; plan for roughly 6–8 GB of RAM at full context with Q4 weights.

## Benchmarks

AI21's headline comparison is against three other small models released in 2025: Gemma 3 4B, Llama 3.2 3B, and Granite 4.0 Micro.

| Benchmark | Jamba Reasoning 3B | Gemma 3 4B | Llama 3.2 3B | Granite 4.0 Micro |
|---|---|---|---|---|
| MMLU-Pro | 61.0% | 42.0% | 35.0% | 44.7% |
| Humanity's Last Exam | 6.0% | 5.2% | 5.2% | 5.1% |
| IFBench | 52.0% | 28.0% | 26.0% | 24.8% |

A few things to read out of that table. The MMLU-Pro gap is large enough that it cannot be explained by training data overlap alone; Jamba Reasoning 3B is doing genuinely better on the harder graduate-knowledge subset of MMLU. IFBench, which tests instruction-following on multi-step prompts, is where the gap is widest. Humanity's Last Exam scores are uniformly low across the entire sub-4B class, which says more about the benchmark than the models. None of these small models are close to solving the hardest tier of academic questions.

The other claim worth checking is speed. AI21's launch post claims 40 tokens per second on an M3 MacBook Pro at 32K context, and 2 to 5x efficiency gains over competing small models. That ratio comes from the Mamba-heavy architecture: SSM layers do not slow down as the context grows the way attention layers do.

## How to run Jamba Reasoning 3B locally with Ollama

Ollama is the fastest setup path. It downloads the GGUF, applies sensible defaults, and exposes an OpenAI-compatible API on `localhost:11434`. If you haven't installed Ollama, our [Ollama vs LM Studio vs llama.cpp guide](/posts/ollama-vs-lm-studio-vs-llama-cpp/) covers the install and a side-by-side of the three runtimes.

Pull and run the model directly from the HuggingFace GGUF repo:

```bash
ollama run hf.co/ai21labs/AI21-Jamba-Reasoning-3B-GGUF:Q4_K_M
```

That pulls the 1.93 GB Q4_K_M GGUF the first time you run it and drops you into an interactive prompt. For programmatic access, use the chat endpoint:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "hf.co/ai21labs/AI21-Jamba-Reasoning-3B-GGUF:Q4_K_M",
  "messages": [
    {"role": "user", "content": "Summarise the differences between Mamba and Transformer architectures in three sentences."}
  ],
  "options": {"temperature": 0.6}
}'
```

Temperature 0.6 matches AI21's recommended sampling parameter. The model card also recommends `max_new_tokens` of 4096 for reasoning outputs because thinking traces can run long.

## How to run Jamba Reasoning 3B with llama.cpp

If you want fine-grained control or are deploying on hardware without an Ollama build, llama.cpp is the better choice. Make sure you are on a recent build; Mamba/Jamba support landed in late 2025.

```bash
./build/bin/llama-server --jinja \
  --hf-repo ai21labs/AI21-Jamba-Reasoning-3B-GGUF \
  --hf-file jamba-reasoning-3b-Q4_K_M.gguf \
  -ngl -1 \
  --host 127.0.0.1 \
  --port 8000
```

The `-ngl -1` flag offloads every layer to GPU if available. On a CPU-only box, drop it. The `--jinja` flag tells llama.cpp to apply the chat template baked into the GGUF, which matters for thinking-mode formatting.

For programmatic use from Python, the llama-cpp-python bindings work the same way they do for any other GGUF:

```python
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="ai21labs/AI21-Jamba-Reasoning-3B-GGUF",
    filename="jamba-reasoning-3b-Q4_K_M.gguf",
)

llm = Llama(
    model_path=model_path,
    n_ctx=128000,
    n_gpu_layers=-1,
    flash_attn=True,
)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Explain SSMs in one paragraph."}],
    temperature=0.6,
    max_tokens=4096,
)
print(output["choices"][0]["message"]["content"])
```

If you want to push context beyond 128K, increase `n_ctx`. The model supports 256K natively, but expect the KV cache to grow accordingly.

{{< figure src="https://images.pexels.com/photos/8623281/pexels-photo-8623281.jpeg" alt="Silver MacBook on a dark background" caption="AI21 reports 35 tok/s at 32K context on a MacBook Pro for Jamba Reasoning 3B. (Photo: Nana Dua, Pexels)" >}}

## vLLM and Transformers

If you want to serve Jamba Reasoning 3B at scale or use it from Python without going through GGUF, vLLM has first-class support starting at version 0.11.0:

```bash
vllm serve "ai21labs/AI21-Jamba-Reasoning-3B" \
  --mamba-ssm-cache-dtype float32 \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Two arguments worth pointing out. `--mamba-ssm-cache-dtype float32` keeps the SSM cache in higher precision, which AI21 recommends to avoid quality degradation on long context. `--reasoning-parser deepseek_r1` tells vLLM how to parse the model's thinking traces, since Jamba Reasoning uses the same reasoning trace format as DeepSeek-R1.

The HuggingFace Transformers route also works straight from the model card:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/AI21-Jamba-Reasoning-3B",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Reasoning-3B")
```

## Which small model should you actually pick?

Jamba Reasoning 3B is the right pick if you need long context and reasoning on a laptop or edge device. The combination of 256K context, 3B parameters, and Apache 2.0 licensing is currently unique in the sub-4B tier.

If you do not need long context, Phi-4-mini or SmolLM3-3B are still strong defaults at the same parameter scale. Our [Qwen3.5-4B vs Phi-4-mini comparison](/posts/qwen3-5-4b-vs-phi-4-mini/) covers how those two stack up for general use, and the [best small language models in 2026](/posts/best-small-language-models-2026/) pillar has the broader picture.

The architecture matters here. Mamba-heavy hybrids like Jamba Reasoning 3B do not just trade off differently than pure transformers; they have a flatter cost curve as context length grows. If your workload spends most of its tokens above 32K (long document Q&A, codebase ingestion, multi-document RAG), the speed advantage compounds.

## Sources

- [Introducing Jamba Reasoning 3B – AI21 blog](https://www.ai21.com/blog/introducing-jamba-reasoning-3b/)
- [AI21-Jamba-Reasoning-3B model card – HuggingFace](https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B)
- [AI21-Jamba-Reasoning-3B-GGUF – HuggingFace](https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B-GGUF)
- [AI21 Releases Jamba Reasoning 3B – Radical Data Science](https://radicaldatascience.wordpress.com/2025/10/08/ai21-releases-jamba-reasoning-3b-fast-tiny-reasoning-model/)
- [Jamba Reasoning 3B redefines what 'small' means in LLMs – VentureBeat](https://venturebeat.com/ai/ai21s-jamba-reasoning-3b-redefines-what-small-means-in-llms-250k-context-on)
