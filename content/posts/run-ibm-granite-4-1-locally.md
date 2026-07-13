---
title: "Run IBM Granite 4.1 Locally: 3B, 8B, and 30B Setup Guide"
date: 2026-06-04
draft: false
tags: ["granite", "ollama", "llama-cpp", "local-inference", "small-models"]
categories: ["small-ai-models"]
description: "How to run IBM Granite 4.1 locally with Ollama or llama.cpp. Full setup guide covering the 3B, 8B, and 30B models, hardware requirements, benchmark scores, and tool-calling."
slug: "run-ibm-granite-4-1-locally"
---

IBM released Granite 4.1 on April 29, 2026, and the headline result is unusual: the 8B model consistently matches the previous-generation Granite 4.0-H-Small (32B MoE) on most benchmarks. If you want to run IBM Granite 4.1 locally, the 8B fits in about 5.3GB on disk and needs less than 6GB of VRAM. This guide covers Ollama and llama.cpp setup for all three sizes, with hardware requirements and the benchmark numbers that matter.

## What Makes Granite 4.1 Different

Granite 4.1 is a family of dense decoder-only transformers trained from scratch on approximately 15 trillion tokens across five phases. The training pipeline starts with broad web and code data, progresses through a dedicated math and code phase, and ends with a multi-stage long-context extension. That final stage takes the 8B and 30B to a 512K token context window; the 3B stops at 128K.

The full family is Apache 2.0 licensed. All three sizes support tool calling with structured JSON output, RAG, fill-in-the-middle code completions, and multilingual inference across twelve languages: English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese.

IBM used a four-stage reinforcement learning pipeline after supervised fine-tuning. The stages cover multi-domain reasoning, RLHF on chat quality, identity and knowledge calibration, and a final math RL pass. The RLHF stage improved Alpaca-Eval scores by roughly 18.9 points; the math RL stage recovered about 3.8 points on GSM8K that were lost during general RLHF.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/6658c911e238275ea9efc339/GWOOEjb2Nr07aJFaJZXmx.png" alt="Granite 4.1 model family benchmark comparison showing instruct performance across 3B, 8B, and 30B sizes" caption="Benchmark comparison across the Granite 4.1 instruct family. Source: IBM Granite / HuggingFace" >}}

## Model Sizes and Hardware Requirements

| Model | Disk Size | VRAM Needed | Context Window |
|---|---|---|---|
| granite4.1:3b | 2.1 GB | ~3 GB | 128K |
| granite4.1:8b | 5.3 GB | ~6 GB | 128K default (512K trained) |
| granite4.1:30b | 17 GB | ~20 GB | 128K default (512K trained) |

The 3B runs on any modern CPU, including a Raspberry Pi 5. See our [Raspberry Pi 5 LLM guide](/posts/run-llms-raspberry-pi-5/) for performance expectations on that hardware. The 8B fits on any GPU with 8GB VRAM and on Apple Silicon Macs with 16GB RAM. The 30B needs 24GB VRAM at Q4 quantization, which means a 3090 or 4090. If your Windows GPU has less than 8GB, read our [low-VRAM local inference guide](/posts/run-local-llms-low-vram-windows-gpu/) first — the 3B is the right choice there.

## Benchmark Scores

All numbers below are for the instruct variants, from IBM's official release benchmarks.

| Benchmark | 3B | 8B | 30B |
|---|---|---|---|
| MMLU (5-shot) | 67.0 | 73.8 | 80.2 |
| GSM8K math (8-shot) | 86.9 | 92.5 | 94.2 |
| HumanEval code (pass@1) | 79.3 | 87.2 | 89.6 |
| BFCL v3 tool calling | 60.8 | 68.3 | 73.7 |
| IFEval | 82.3 | 87.1 | 89.7 |
| ArenaHard | 37.8 | 69.0 | 71.0 |

The code and math numbers are the ones worth focusing on. 79.3 HumanEval at 3B and 87.2 at 8B are strong for those parameter counts — comparable to or better than much larger models from a year ago. The BFCL v3 scores matter if you're building agentic pipelines: 60.8 at 3B and 68.3 at 8B indicate these models can select tools and produce valid JSON arguments reliably enough for real use.

{{< figure src="https://cdn-uploads.huggingface.co/production/uploads/6658c911e238275ea9efc339/C7F0iXCkhlEREl_1Z3Fnn.png" alt="Chart comparing Granite 4.1-8B instruct against Granite 4.0-H-Small 32B MoE across multiple benchmarks" caption="Granite 4.1-8B vs Granite 4.0-H-Small (32B MoE) across key benchmarks. Source: IBM Granite / HuggingFace" >}}

## Run with Ollama

Ollama has all three sizes available. Pull and run:

```bash
# 3B — for CPU, low-VRAM laptops, or Raspberry Pi
ollama run granite4.1:3b

# 8B — the practical default for most setups
ollama run granite4.1:8b

# 30B — for 24GB VRAM cards
ollama run granite4.1:30b
```

Ollama exposes a local HTTP API on `http://localhost:11434` using the OpenAI schema. If you're already running another model through Ollama, you change the model name in your config and the rest of your stack works as-is.

On an RTX 4060, the 8B runs at roughly 25 to 60 tokens per second with first-token latency under 500ms. On an M2 MacBook Pro, expect 15 to 30 tokens per second for the 8B depending on memory bandwidth.

## Run with llama.cpp and GGUF

Unsloth has published GGUF variants for the 3B. Download llama.cpp from [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and pull a GGUF file from [unsloth/granite-4.1-3b-GGUF](https://huggingface.co/unsloth/granite-4.1-3b-GGUF).

```bash
# Run the 3B Q4_K_M GGUF directly
llama-cli -m granite-4.1-3b-instruct-Q4_K_M.gguf \
  -p "Write a Python function that reads a CSV and returns a list of dicts." \
  -n 512

# Or run as a local server
llama-server -m granite-4.1-3b-instruct-Q4_K_M.gguf --port 8080
```

Available quant sizes for the 3B:

| Quantization | File Size |
|---|---|
| Q4_K_M | 2.15 GB |
| Q5_K_M | 2.45 GB |
| Q8_0 | 3.62 GB |
| BF16 (full) | 6.81 GB |

Q4_K_M is the right default for most use cases. For a full breakdown of what each quantization level costs in quality, see our [GGUF quantization levels guide](/posts/gguf-quantization-levels-q4-q5-q8/).

## Run with Transformers

For Python inference, the standard transformers setup works across all three sizes. Use `device_map="auto"` to let the library handle placement across CPU and GPU:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "ibm-granite/granite-4.1-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

messages = [{"role": "user", "content": "Summarize this document in three bullet points."}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

output = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

For the 3B on CPU, replace `model_path` with `ibm-granite/granite-4.1-3b-instruct` and drop `torch_dtype`.

## Tool Calling

Tool calling is the capability that sets Granite 4.1 apart from most models at this size. The BFCL v3 benchmark tests structured function selection and argument generation under realistic conditions; 68.3 at 8B is competitive with models that require significantly more hardware.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search an internal document store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    }
]

chat = [{"role": "user", "content": "Find documents about quarterly revenue."}]
prompt = tokenizer.apply_chat_template(
    chat, tokenize=False, tools=tools, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

The model outputs a JSON tool call you parse and execute in your application. The 3B is workable for simple single-tool pipelines; the 8B handles multi-tool chains without degrading.

## Which Size to Actually Use

### 3B: CPU, edge, and constrained hardware

The 3B is the right choice when you have no discrete GPU, are targeting an embedded or edge device, or need to keep the memory footprint small. 128K context covers most RAG use cases. Runs on a Raspberry Pi 5 and any modern laptop CPU.

### 8B: The default for local deployments

Start here unless hardware forces you elsewhere. 5.3GB on disk, 512K context, and benchmark scores that beat much larger models from last year. Ollama makes setup a one-line command, and the OpenAI-compatible API means it drops into any existing local inference stack without changes.

### 30B: Long documents and complex structured output

The 30B is for 512K-context document pipelines, high-quality function calling, and multi-step agent workflows where quality matters more than cost. It needs 24GB VRAM. The step up from 8B is most visible on ArenaHard (69.0 to 71.0) and MMLU (73.8 to 80.2), not raw coding or math where the 8B is already close.

## What Is IBM Granite 4.1?

IBM Granite 4.1 is a family of dense open-weight language models in 3B, 8B, and 30B sizes, released on April 29, 2026. All variants use Apache 2.0 licensing. They were trained on approximately 15 trillion tokens across five data phases, with context windows of 128K (3B) and 512K (8B, 30B). The 8B instruct model matches the prior-generation Granite 4.0-H-Small 32B MoE on most benchmarks. The models are available via Ollama, GGUF/llama.cpp, and the Hugging Face transformers library.

## Sources

- [Introducing the IBM Granite 4.1 family of models — IBM Research](https://research.ibm.com/blog/granite-4-1-ai-foundation-models)
- [Granite 4.1 LLMs: How They're Built — HuggingFace Blog](https://huggingface.co/blog/ibm-granite/granite-4-1)
- [granite4.1 on Ollama](https://ollama.com/library/granite4.1)
- [unsloth/granite-4.1-3b-GGUF — HuggingFace](https://huggingface.co/unsloth/granite-4.1-3b-GGUF)
- [ibm-granite/granite-4.1-language-models — GitHub](https://github.com/ibm-granite/granite-4.1-language-models)
