---
title: "GGUF Quantization Levels Explained: Q4, Q5, Q8, and IQ Quants"
date: 2026-05-27
draft: false
tags: ["quantization", "gguf", "local-inference", "small-models"]
categories: ["small-ai-models"]
description: "What Q4_K_M, Q5_K_M, Q8_0, and the IQ-series quants actually cost you in file size, quality, and speed, with real benchmark numbers."
slug: "gguf-quantization-levels-q4-q5-q8"
---

When you search for a model on HuggingFace, you'll find the same model offered in a dozen or more GGUF files: Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ2_M, IQ4_XS. Each one is a different quantized version of the same weights. Pick the right one and you get the best quality your hardware can fit. Pick the wrong one and you either waste VRAM on a model too small for your GPU, or you load a model that barely fits and runs slower than a lighter quant would.

This post covers how each quantization level works, what you actually lose at each step, and how to pick the right one for your hardware. If you're deciding between GGUF, MLX, and ONNX format entirely, that's covered in the [GGUF vs ONNX vs MLX guide](/posts/gguf-vs-onnx-vs-mlx/). This post assumes you've already chosen GGUF.

## What the numbers mean

Model quantization reduces the precision of each weight value. A full-precision model stores each weight as a 16-bit float (FP16), using two bytes per weight. A Q4 quantization stores each weight in approximately 4 bits, cutting the file size roughly in half.

The "K" in Q4_K_M stands for K-quant, a mixed-precision scheme introduced in llama.cpp in 2023. Instead of applying the same compression to every layer, K-quants apply more bits to attention layers (which are more sensitive to precision loss) and fewer bits to feed-forward layers (which tolerate more compression). The result is better quality at the same file size compared to the older, uniform Q4_0 format.

The suffix matters:
- **_M** (medium): slightly more bits than _S, better quality
- **_S** (small): more aggressive compression, smaller file
- **_K** (no suffix on Q6_K): single mixed-precision K-quant, one variant

Unless you're seriously VRAM-constrained, always prefer the _M variant when it exists.

## File sizes across model families

File sizes scale roughly linearly with parameter count and bits per weight. These are real measurements from HuggingFace model repos:

| Quant | Bits/weight | Llama 3.1 8B | Phi-4-mini 3.8B | Qwen3.5-4B (approx) |
|---|---|---|---|---|
| FP16 | 16 | 15.3 GB | 7.6 GB | 7.8 GB |
| Q8_0 | 8 | 8.1 GB | 4.1 GB | 4.2 GB |
| Q6_K | ~6.5 | 6.3 GB | 3.1 GB | 3.2 GB |
| Q5_K_M | 5.5 | 5.5 GB | 2.9 GB | 2.9 GB |
| Q4_K_M | 4.5 | 4.6 GB | 2.5 GB | 2.6 GB |
| IQ4_XS | 4.25 | 4.4 GB | 2.3 GB | 2.4 GB |
| IQ3_M | ~3.4 | 3.5 GB | 1.8 GB | 1.8 GB |
| IQ2_M | ~2.9 | 2.7 GB | 1.4 GB | 1.5 GB |

FP16 values are the unquantized baseline. Everything below that is a tradeoff between size and quality.

{{< figure src="https://images.pexels.com/photos/19599849/pexels-photo-19599849.jpeg" alt="Open hard drive showing internal disk platters and read head mechanism" caption="Quantization is fundamentally about storage efficiency: fitting more model into less space without losing what matters. (Photo: William Warby, Pexels, free to use)" >}}

## What you actually lose at each level

The most rigorous public evaluation of llama.cpp quantization levels is a January 2026 study from arXiv (2601.14277), which ran 13 quantization configurations against Llama 3.1 8B-Instruct. The results:

| Quant | Perplexity (WikiText-2) | MMLU (5-shot) | File size vs FP16 |
|---|---|---|---|
| FP16 baseline | 7.32 | 63.5% | 100% |
| Q6_K | 7.35 | 63.2% | 41% |
| Q5_K_M | 7.40 | 62.8% | 36% |
| Q4_K_M | 7.56 | 62.4% | 30% |

The key takeaway: Q4_K_M drops MMLU accuracy by about 1.1 percentage points versus the FP16 baseline. That's the cost of cutting the file from 15.3 GB to 4.6 GB. For most chat, coding, and RAG tasks, a 1% accuracy gap is not something you'd notice in practice. The perplexity numbers tell the same story: Q4_K_M's perplexity of 7.56 versus FP16's 7.32 is a 3.3% increase, which is measurably worse but not dramatically so.

Q6_K is where the numbers become hard to argue against: 59% of the file size, with perplexity only 0.03 higher than FP16 (7.35 vs 7.32) and MMLU within 0.3 percentage points. If you can fit Q6_K in your VRAM, the quality-to-size tradeoff is better than Q5_K_M or Q4_K_M. The catch is that Q6_K for an 8B model requires 6.3 GB VRAM, which rules it out on a 6 GB card.

## Inference speed by quant level

This is where things get counterintuitive. Smaller quants don't automatically mean faster inference. On CPU (Intel Xeon with AVX-512, from the same arXiv study):

| Quant | Decode speed (tokens/sec) |
|---|---|
| Q4_K_M | 5.12 |
| Q5_K_M | 6.85 |
| Q6_K | 6.33 |
| Q8_0 | 5.03 |

Q5_K_M is actually faster than Q4_K_M on that hardware, and Q8_0 is the slowest. The reason: CPU SIMD kernels are highly optimized for specific bit widths, and Q5_K_M's particular combination of mixed precision aligns well with AVX-512 instruction widths on that Xeon. Q4_K_M wins on GPU, where memory bandwidth is the bottleneck and smaller files simply move faster.

The practical rule: on GPU, pick the smallest quant that gives acceptable quality, because you're bandwidth-bound. On CPU, Q4_K_M is fine for throughput, but Q5_K_M may actually be faster depending on your hardware. Test both if speed matters.

{{< figure src="https://images.pexels.com/photos/3520692/pexels-photo-3520692.jpeg" alt="Black and white close-up of a circuit board showing intricate electronic components" caption="The quantization level you choose is determined as much by your hardware's capabilities as by the model's parameter count. (Photo: Miguel Á. Padriñán, Pexels, free to use)" >}}

## The IQ series: importance-matrix quants

The IQ (importance quantization) formats were introduced to llama.cpp in late 2023 and have become progressively better. Where K-quants use a fixed mixed-precision strategy, IQ-quants use a precomputed importance matrix (imatrix) to identify which specific weights matter most for the model's output. High-importance weights get more bits; low-importance weights get fewer.

The result: an IQ4_XS (4.25 bits/weight) generally outperforms Q4_K_M on quality benchmarks while using slightly less space. For memory-constrained scenarios, IQ2_M is the sweet spot for 2-bit-class models. It gets more usable output than IQ2_XS or IQ2_S by spending more bits on critical weights.

A few things to know about IQ quants:
- They require an importance matrix file to quantize properly. Pre-quantized IQ files from reputable sources (Bartowski, Unsloth, LM Studio team) include this.
- If you're quantizing your own model, the [llama.cpp quantize tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) accepts an imatrix file via `--imatrix` flag.
- IQ4_XS is a drop-in replacement for Q4_K_M in most cases; it saves around 250 MB on an 8B model.

## Which quant to pick

### Q4_K_M: the default starting point

Use Q4_K_M as your baseline. It fits a 7–8B model in 6 GB VRAM, loses about 1% on benchmarks compared to FP16, and is supported in every inference tool. If you're unsure, start here.

### Q5_K_M and Q6_K: when quality matters

Use Q5_K_M or Q6_K when your GPU has 8–12 GB VRAM and you want noticeably better quality on tasks that expose quantization errors: multilingual text, structured output, complex reasoning chains. The jump from Q4 to Q5 is more perceptible in practice than the benchmark gap suggests.

### Q8_0: near-lossless, but legacy

Q8_0 is within rounding error of FP16 quality (perplexity difference of roughly 0.01 at 8B scale). Use it when VRAM is not a constraint and you want to eliminate quantization error as a variable in your testing. Worth noting: Q8_0 is technically a legacy round-to-nearest format in the GGUF spec, predating the K-quant series. It's still reliable and widely available, but Q8_K_M would be preferable where available.

### IQ4_XS: a better Q4

Use IQ4_XS instead of Q4_K_M if the IQ version is available for the model you want. Same VRAM requirement, slightly better quality, and about 250 MB smaller on an 8B model. There's no meaningful downside.

### IQ2_M and IQ3_M: last resort

Use these only when RAM forces you to. On a 4 GB GPU running a 7B model, IQ2_M is the only option that fits. Output quality degrades noticeably at 2-bit precision. These are "can I get the model to run at all" scenarios, not production configurations.

## Frequently asked questions

### Will I actually notice the difference between Q4 and Q8?

For short conversational responses: no. For extended reasoning, coding tasks with subtle bugs, or multilingual text: sometimes. The degradation from Q4_K_M is not random noise. It tends to cause the model to slightly lose track of nuanced context in long prompts, or produce outputs that are almost but not quite correct in edge cases. If you're doing simple Q&A or summarization on English text, Q4_K_M is fine. If you're using the model as a reasoning engine for code review, Q5_K_M or Q6_K is worth the extra VRAM.

### Does quantization affect all model sizes equally?

Smaller models suffer more from aggressive quantization. A 1B model at Q4_K_M loses proportionally more capability than an 8B model at Q4_K_M, because there are fewer weights available to absorb the precision loss. If you're running a 1.5–3B model, prefer Q5_K_M or Q6_K if your hardware allows it. See the [best small language models in 2026](/posts/best-small-language-models-2026/) for hardware recommendations matched to specific model sizes.

### What about models that are already "pre-quantized" on HuggingFace?

Some model cards advertise quantized variants as the official release. These pre-quantized versions are generally fine to use, but check who created the quantized files. Bartowski, Unsloth, and the official lab releases use calibration datasets and importance matrices. Unknown uploaders may have used default settings that produce lower quality output.

### Can I quantize a model myself?

Yes. The process:

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
pip install -r requirements.txt
cmake -B build && cmake --build build --config Release -j

# Convert HuggingFace model to FP16 GGUF
python convert_hf_to_gguf.py /path/to/model --outtype f16 --outfile model-f16.gguf

# Quantize to Q4_K_M
./build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# Quantize to IQ4_XS with importance matrix (better quality)
./build/bin/llama-quantize --imatrix calibration.imatrix model-f16.gguf model-iq4_xs.gguf IQ4_XS
```

To generate the importance matrix yourself, you need a calibration dataset (a sample of the kind of text you'll run through the model) and the `llama-imatrix` tool. For most use cases, downloading a pre-quantized file from a reputable source is faster and produces comparable results.

## Putting it together

For a 7–8B model on a typical 8 GB GPU, Q5_K_M is the better default than Q4_K_M if you care about output quality and your VRAM fits it. If you're at 6 GB VRAM, Q4_K_M or IQ4_XS is your ceiling. If you're doing CPU-only inference on a machine with 16 GB RAM, Q6_K is worth loading; the quality is nearly indistinguishable from FP16 at a fraction of the size.

The IQ formats are underused. IQ4_XS has no meaningful downside over Q4_K_M: it uses slightly less space and scores better on benchmarks. If a model has both available, pick IQ4_XS.

For running any of these on a specific hardware platform, the [Apple Silicon local LLM guide](/posts/run-small-llms-apple-silicon/), [Raspberry Pi 5 guide](/posts/run-llms-raspberry-pi-5/), and [low-VRAM Windows GPU guide](/posts/run-local-llms-low-vram-windows-gpu/) each include hardware-specific quant recommendations.

## Sources

- [Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct (arXiv:2601.14277)](https://arxiv.org/html/2601.14277v1)
- [llama.cpp quantize README — ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Difference in different quantization methods — llama.cpp Discussion #2094](https://github.com/ggml-org/llama.cpp/discussions/2094)
- [Common AI Model Formats — HuggingFace Blog](https://huggingface.co/blog/ngxson/common-ai-model-formats)
- [GGUF format documentation — HuggingFace Hub](https://huggingface.co/docs/hub/gguf)
