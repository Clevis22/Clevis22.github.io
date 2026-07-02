---
title: "How to Quantize a Model with llama.cpp: From Safetensors to GGUF"
date: 2026-07-02
draft: false
tags: ["small-models", "slm", "edge-ai", "gguf", "quantization", "llama-cpp"]
categories: ["small-ai-models"]
description: "Step-by-step guide to quantize a model with llama.cpp: convert HuggingFace safetensors to GGUF, run llama-quantize, and improve low-bit quants with an importance matrix."
slug: "how-to-quantize-model-llama-cpp"
---

[![Close-up of two woolly llamas in a fenced outdoor area](https://images.pexels.com/photos/6778677/pexels-photo-6778677.jpeg)](https://www.pexels.com/photo/6778677/)
*The project mascot, presumably moments before converting something to GGUF. Photo by [Cosmin on Pexels](https://www.pexels.com/photo/6778677/), free to use.*

Most of the time you don't need to quantize a model yourself. Someone has already uploaded a Q4_K_M to HuggingFace, you download it, done. But sooner or later you hit a gap: a brand-new release with no GGUF yet, a fine-tune you made yourself, or a repo that only ships Q4 and Q8 when what you actually want is Q6_K. This guide covers how to quantize a model with llama.cpp, start to finish: converting HuggingFace safetensors to GGUF, running `llama-quantize`, and using an importance matrix to keep low-bit quants usable.

If you're still deciding which quantization level to target, read our [GGUF quantization levels explainer](/posts/gguf-quantization-levels-q4-q5-q8/) first. This post is about producing the files; that one is about choosing between them.

## What you need before starting

The pipeline has two halves, and they have different requirements.

The conversion step (safetensors to GGUF) is a Python script, `convert_hf_to_gguf.py`, that lives in the [llama.cpp repository](https://github.com/ggml-org/llama.cpp). You need to clone the repo and install its Python dependencies to use it.

The quantization step is a compiled binary, `llama-quantize`. You can build it from the same repo, or skip compiling entirely: the [pre-built release archives](https://github.com/ggml-org/llama.cpp/releases) on GitHub include the full tool set, and the package manager builds are kept current with each release.

```powershell
# Windows
winget install llama.cpp
```

```bash
# macOS / Linux
brew install llama.cpp
```

For the Python side, clone the repo and install the requirements:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
python3 -m pip install -r requirements.txt
```

Disk space matters more than you'd expect. The intermediate conversion is written at bf16 or f16, which is the full-precision size: roughly 6 GB for a 3B model, 15 GB for an 8B. You need room for the original safetensors, the f16 GGUF, and every quant you produce, all at once.

## Step 1: Convert safetensors to GGUF

`convert_hf_to_gguf.py` reads a HuggingFace model directory (config, tokenizer, safetensors shards) and writes a single self-contained GGUF file. The `--remote` flag, marked experimental in the script, reads the safetensors straight from the Hub so you can skip a separate multi-gigabyte download (the config and tokenizer files are still fetched locally).

Using [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) as the example:

```bash
python convert_hf_to_gguf.py --outfile SmolLM3-3B-bf16.gguf \
  --outtype bf16 --remote HuggingFaceTB/SmolLM3-3B
```

If you already have the model on disk (say, a fine-tune you trained), point the script at the directory instead:

```bash
python convert_hf_to_gguf.py ./my-finetuned-model --outfile my-model-bf16.gguf --outtype bf16
```

Two notes on `--outtype`. First, always convert to bf16 or f16 here rather than trying to quantize in the same step. The converter's only quantized output type is Q8_0; the K-quants and IQ quants people actually run can only be produced by `llama-quantize`, and you'll want the full-precision GGUF around anyway for the imatrix step below. Second, if the original model was trained in bf16 (SmolLM3 was, and so are almost all modern releases), use bf16 to avoid a lossy detour through f16.

The script supports a long list of architectures, but not everything. If a model just came out and the converter throws an unknown-architecture error, check the llama.cpp issue tracker; support for new architectures typically lands within days of a significant release, and there's nothing you can do locally except wait or write the mapping yourself.

## Step 2: Quantize the model with llama-quantize

This is the short part. `llama-quantize` takes the full-precision GGUF, an output filename, and a quantization type:

```bash
llama-quantize SmolLM3-3B-bf16.gguf SmolLM3-3B-Q4_K_M.gguf Q4_K_M
```

On a modern desktop CPU this takes a few minutes for a 3B model. Run it once per quant level you want; each run reads the same bf16 input.

How much you save depends on the level. These are the official numbers from the [llama-quantize documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md), measured on Llama 3.1 8B:

| Quant type | Bits/weight | File size (8B model) |
|---|---|---|
| F16 | 16.00 | 14.96 GiB |
| Q8_0 | 8.50 | 7.95 GiB |
| Q5_K_M | 5.70 | 5.33 GiB |
| Q4_K_M | 4.89 | 4.58 GiB |
| Q3_K_M | 3.99 | 3.74 GiB |
| IQ2_XXS | 2.38 | 2.23 GiB |
| IQ1_S | 2.00 | 1.87 GiB |

As a sanity check against real-world output: the official [ggml-org SmolLM3-3B GGUF repo](https://huggingface.co/ggml-org/SmolLM3-3B-GGUF) ships the f16 at 6.16 GB, Q8_0 at 3.28 GB, and Q4_K_M at 1.92 GB. If your own Q4_K_M of a 3B model comes out wildly different from ~2 GB, something went wrong in conversion.

A few flags worth knowing:

```bash
# Leave output.weight unquantized (slightly bigger file, better quality)
llama-quantize --leave-output-tensor input-bf16.gguf output-Q4_K_M.gguf Q4_K_M

# Requantize an already-quantized GGUF (works, but quality suffers; avoid if possible)
llama-quantize --allow-requantize input-Q8_0.gguf output-Q4_K_M.gguf Q4_K_M
```

That second one deserves emphasis: always quantize from the bf16/f16 original. Requantizing a Q8_0 down to Q4_K_M compounds two rounds of rounding error, which is why the tool makes you opt in with `--allow-requantize` and warns you about it.

[![Stack of opened hard disk drives showing internal platters](https://images.pexels.com/photos/32892856/pexels-photo-32892856.jpeg)](https://www.pexels.com/photo/32892856/)
*One conversion run produces a full-precision file plus every quant you make from it. Budget disk space accordingly. Photo by [Marta Branco on Pexels](https://www.pexels.com/photo/32892856/), free to use.*

## Step 3: Add an importance matrix for low-bit quants

At Q4_K_M and above, plain quantization is fine. Below that, quality falls off fast unless you give the quantizer information about which weights matter. That's what the importance matrix (imatrix) does: you run the full-precision model over a calibration text file, `llama-imatrix` records which activations carry the most signal, and `llama-quantize` uses that record to spend its limited bits where they count.

```bash
# 1. Compute the importance matrix (offload to GPU with -ngl for speed)
llama-imatrix -m SmolLM3-3B-bf16.gguf -f calibration-data.txt -ngl 99

# 2. Quantize using it
llama-quantize --imatrix imatrix.gguf SmolLM3-3B-bf16.gguf SmolLM3-3B-IQ3_M.gguf IQ3_M
```

The output file defaults to `imatrix.gguf`. The calibration file is just plain text; general-purpose mixes of web text, code, and multilingual content are popular because the matrix should reflect the kind of input the model will actually see. If your model is a coding fine-tune, calibrate on code. A long run can be checkpointed and resumed with `--chunk`, and `--output-frequency` controls how often intermediate results are saved (every 10 chunks by default).

The entire IQ series (IQ1_S through IQ4_XS) is designed around imatrix data. Quantizing to IQ2 or IQ3 without one produces a file that runs but performs noticeably worse than the same size with calibration. For Q4_K_M and up, an imatrix still helps at the margins, which is why the big quant repos like Bartowski's use one for every level they publish.

## Do you need to quantize models yourself?

Direct answer: only in four situations. A model is new enough that no GGUF exists yet. You fine-tuned something and need to deploy it. The specific quant level you want isn't published (Q6_K and Q5_K_M are frequently skipped by uploaders). Or you want a custom imatrix calibrated on your own domain data rather than a generic mix.

For everything else, download an existing GGUF. The `ggml-org`, `unsloth`, and `bartowski` accounts on HuggingFace cover most significant releases within a day or two, with imatrix quants at every level. Your own quant made without an imatrix will be slightly worse than their published one at the same size.

There's also a middle path: [gguf-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo), an official HuggingFace Space run by ggml-org. You point it at a Hub model, pick a quant type, and it converts and uploads the GGUF to your own account, no local compute or disk required. For one-off quants of public models it's hard to beat. The local pipeline in this guide is for fine-tunes you can't upload, models needing custom calibration, or anything you'd rather not push through someone else's infrastructure.

## Test the result

Before you deploy the file anywhere, load it and generate something:

```bash
llama-cli -m SmolLM3-3B-Q4_K_M.gguf
```

Ask a few questions you know the answers to, and if it's a coding model, have it write a function. A conversion that silently mangled the tokenizer or chat template produces a model that loads fine and outputs garbage, so a thirty-second smoke test saves real debugging later. If output looks degraded at Q4 but fine at Q8, your bf16 conversion is good and the model is simply sensitive to quantization; try Q5_K_M or add an imatrix.

The finished GGUF works everywhere the format does: llama.cpp directly, or imported into Ollama or LM Studio. Our [Ollama vs LM Studio vs llama.cpp comparison](/posts/ollama-vs-lm-studio-vs-llama-cpp/) covers which runtime fits your setup, and if you're wondering whether GGUF is even the right format for your hardware, the [GGUF vs ONNX vs MLX guide](/posts/gguf-vs-onnx-vs-mlx/) walks through that decision.

## Sources

- [llama.cpp repository (ggml-org)](https://github.com/ggml-org/llama.cpp)
- [llama-quantize tool documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [llama-imatrix tool documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
- [llama.cpp installation guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)
- [ggml-org/SmolLM3-3B-GGUF on HuggingFace](https://huggingface.co/ggml-org/SmolLM3-3B-GGUF)
- [gguf-my-repo HuggingFace Space](https://huggingface.co/spaces/ggml-org/gguf-my-repo)
- [SmolLM3-3B model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
