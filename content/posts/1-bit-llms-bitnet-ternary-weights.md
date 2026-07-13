---
title: "1-bit LLMs Explained: How BitNet's Ternary Weights Actually Work"
date: 2026-06-01
draft: false
tags: ["bitnet", "quantization", "edge-ai", "small-models"]
categories: ["small-ai-models"]
description: "A deep guide to 1-bit LLMs: how 1.58-bit ternary weights work, BitNet b1.58 2B4T benchmarks, running them on CPU with bitnet.cpp, and whether they beat 4-bit quantization."
slug: "1-bit-llms-bitnet-ternary-weights"
hardware: "CPU / edge"
---

A 1-bit LLM stores each weight as one of three values: minus one, zero, or plus one. That single design choice is why a 2-billion-parameter model can hold its non-embedding weights in 0.4 GB of memory and decode tokens on a laptop CPU using a fraction of the energy a normal model burns. This guide explains how 1-bit LLMs actually work, walks through the real released model you can run today (BitNet b1.58 2B4T), and answers the question everyone asks once they see the memory numbers: if these are so efficient, why isn't everything 1-bit?

The short version is that 1-bit LLMs are not a quantization trick you apply after training. They are a different way of building the model from the first step, and that distinction explains both their promise and their current limits.

{{< figure src="https://arxiv.org/html/2402.17764v1/x1.png" alt="Pareto curve showing 1-bit LLMs reducing inference cost while maintaining performance" caption="The core pitch: ternary 1-bit LLMs sit on a better cost/performance Pareto front than full-precision models. Source: Ma, Wang et al., 'The Era of 1-bit LLMs' (arXiv 2402.17764)." >}}

## What is a 1-bit LLM, and why is it called "1.58-bit"?

The names are confusing, so it helps to separate two things. The original [BitNet paper](https://arxiv.org/abs/2310.11453) from October 2023 used true 1-bit weights: every value was either minus one or plus one. A follow-up in February 2024, [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764), added a third allowed value, zero, giving weights of {-1, 0, +1}. Adding zero lets the model express "this connection does not matter," which turns out to be important for quality.

Three possible states is no longer one bit of information. Storing one of three values requires log₂(3) ≈ 1.58 bits, which is where the "1.58-bit" label comes from. So when people say "1-bit LLM" today, they almost always mean the ternary 1.58-bit variant. The two terms get used interchangeably, and this post follows that convention while keeping the distinction clear where it matters.

Why does the zero matter so much? In a full-precision network, a weight near zero contributes almost nothing, and the model relies on having millions of these near-zero weights to shape its behavior. Pure 1-bit weights cannot represent "almost nothing," only strong positive or strong negative. The ternary scheme restores the ability to switch a connection off, and that single addition is what closes most of the quality gap with full-precision models.

## How 1-bit LLMs actually work

The mechanism lives in a layer called BitLinear, introduced in the 2023 BitNet paper as a drop-in replacement for the standard `nn.Linear` layer in a transformer. Everything else about the architecture stays recognizable. BitNet b1.58 2B4T uses rotary position embeddings, a squared-ReLU feed-forward activation, and subln normalization, all standard choices. The weights are what change.

Two quantization steps run during the forward pass. Weights are mapped to ternary values using an absmean scheme: the weight matrix is scaled by its average absolute value, then each entry is rounded to the nearest of {-1, 0, +1}. Activations flowing through the layer are quantized separately to 8-bit integers using an absmax scheme applied per token. The shorthand for this configuration is W1.58A8, meaning 1.58-bit weights and 8-bit activations.

The payoff is in the matrix multiplication. A normal transformer multiplies 16-bit weights by 16-bit activations, which means millions of floating-point multiply-accumulate operations. When the weights are only {-1, 0, +1}, multiplication disappears. Multiplying an activation by minus one is a sign flip, by zero is a skip, and by plus one is a copy. The expensive matmul collapses into additions and subtractions, which is why the energy savings are so large on the right hardware.

### The part most people get wrong: native training versus quantization

This is the single most important thing to understand about 1-bit LLMs. You cannot take Llama 3 or Qwen and "quantize it to 1-bit" the way you make a Q4 GGUF. Crushing a model trained in 16-bit down to ternary weights after the fact destroys it, because the model never learned to work within those constraints.

Instead, 1-bit LLMs are trained from scratch with the quantization built into every forward pass, a technique called quantization-aware training. The model keeps a higher-precision master copy of the weights for the gradient updates during training, but it always sees the ternary version when computing its outputs, so it learns weights that survive the rounding. The released BitNet b1.58 2B4T model went through three stages: pre-training on 4 trillion tokens, supervised fine-tuning, and direct preference optimization. The result is a model whose weights were ternary the entire time, not squeezed down afterward. For a refresher on the more conventional approach this replaces, see our breakdown of [Q4, Q5, and Q8 quantization levels](/posts/gguf-quantization-levels-q4-q5-q8/).

## A short history: from BitNet to b1.58 2B4T

Three milestones matter. In October 2023, Microsoft Research published BitNet, proving a 1-bit transformer could be trained from scratch and followed a scaling law similar to full-precision models. In February 2024, the same group released the 1.58-bit ternary variant and showed it matching FP16 LLaMA on perplexity and downstream tasks at scale. Then in April 2025, they shipped [BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), the first open-weight, natively-trained 1-bit LLM at a useful size, under an MIT license.

The 2024 paper's results explain why the 2025 release was worth making. The researchers compared BitNet b1.58 against FP16 LLaMA at matched sizes:

| Model size | BitNet b1.58 perplexity | LLaMA FP16 perplexity | BitNet zero-shot avg | LLaMA zero-shot avg |
|---|---|---|---|---|
| 700M | 12.87 | 12.33 | 44.3 | 45.5 |
| 1.3B | 11.29 | 11.25 | 45.4 | 46.2 |
| 3B | 9.91 | 10.04 | 50.2 | 49.7 |
| 3.9B | 9.62 | n/a | 51.2 | n/a |

The crossover is the headline. At 700M the ternary model lags, but by 3B it matches FP16 LLaMA on perplexity (9.91 versus 10.04) and edges ahead on average zero-shot accuracy (50.2 versus 49.7). Bigger ternary models do not just stay competitive, they catch up. The same paper reported that at 3B size the ternary model used 3.55× less memory and ran 2.71× faster, with the gap widening at larger scales (4.1× lower latency at 70B).

## BitNet b1.58 2B4T: the 1-bit LLM you can run today

The 2025 release is the one to actually try. It is a 2-billion-parameter model trained on 4 trillion tokens, with a 4,096-token context window and the LLaMA 3 tokenizer (128,256 vocab). Here is how it lands against full-precision small models of similar size, from the official technical report:

| Benchmark | BitNet b1.58 2B | Qwen2.5 1.5B | SmolLM2 1.7B | Llama 3.2 1B | Gemma-3 1B |
|---|---|---|---|---|---|
| Memory (non-emb) | **0.4 GB** | 2.6 GB | 3.2 GB | 2 GB | 1.4 GB |
| CPU decode latency | **29 ms** | 65 ms | 67 ms | 48 ms | 41 ms |
| Energy (estimated) | **0.028 J** | 0.347 J | 0.425 J | 0.258 J | 0.186 J |
| ARC-Challenge | **49.91** | 46.67 | 43.52 | 37.80 | 38.40 |
| GSM8K (math) | **58.38** | 56.79 | 45.11 | 38.21 | 31.16 |
| WinoGrande | **71.90** | 62.83 | 68.98 | 59.51 | 58.48 |
| MMLU | 53.17 | **60.25** | 49.24 | 45.58 | 39.91 |
| HumanEval+ (code) | 38.40 | **50.60** | 28.00 | 31.10 | 37.20 |
| Average (all tasks) | 54.19 | **55.23** | 48.70 | 44.90 | 43.74 |

Read this honestly and two things stand out. First, the efficiency numbers are not marginal. The non-embedding weights fit in 0.4 GB, roughly six times smaller than Qwen2.5 1.5B, and the estimated per-inference energy is more than ten times lower. Second, on quality the model is genuinely competitive but not dominant. It tops the table on ARC-Challenge, GSM8K, and WinoGrande, yet Qwen2.5 1.5B still wins the overall average (55.23 versus 54.19) and leads clearly on MMLU and coding. A 1-bit model trading blows with a strong full-precision model at this size is the actual achievement here.

{{< figure src="https://images.pexels.com/photos/8033476/pexels-photo-8033476.jpeg" alt="Macro close-up of a computer processor chip" caption="1-bit LLMs are built for the CPU. Their efficiency advantage shows up on commodity processors, not high-end GPUs. Photo: Robert Pügner / Pexels (free to use)." >}}

## The catch: efficiency only shows up with bitnet.cpp

Here is the trap that disappoints most people on their first try. If you load BitNet b1.58 2B4T through the standard Hugging Face `transformers` library, you get none of the speed or energy benefits. Microsoft states this directly on the model card: the execution paths in `transformers` do not contain the specialized kernels needed to exploit ternary weights, so the model just runs like a small bf16 model with extra steps.

The efficiency lives in [bitnet.cpp](https://github.com/microsoft/BitNet), Microsoft's official C++ inference framework (a cousin of llama.cpp built for this weight format). Its measured gains come from the framework's optimized kernels, not the model file alone:

| CPU type | Inference speedup | Energy reduction |
|---|---|---|
| ARM (e.g. Apple silicon, mobile) | 1.37× to 5.07× | 55.4% to 70.0% |
| x86 (Intel/AMD desktop) | 2.37× to 6.17× | 71.9% to 82.2% |

The larger the model, the bigger the speedup, which is the opposite of how GPU inference usually scales. The framework's most quoted demo is running a 100B-parameter BitNet model on a single CPU at 5 to 7 tokens per second, around human reading speed, with no GPU involved. That is the genuine reason to care about this architecture: it moves capable inference onto hardware that does not have a discrete accelerator. If your target is a Raspberry Pi or a fanless mini-PC, this is the relevant frontier, and it pairs well with the approaches in our [guide to running LLMs on a Raspberry Pi 5](/posts/run-llms-raspberry-pi-5/).

## How to run BitNet b1.58 2B4T locally

The supported path is bitnet.cpp with the GGUF build of the model. The setup mirrors llama.cpp but uses Microsoft's fork:

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet

# 2. Install Python deps
pip install -r requirements.txt

# 3. Download the official GGUF weights
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T

# 4. Set up the environment / quantization kernel
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# 5. Run inference
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Explain ternary weights in one paragraph." -cnv
```

The whole download is a few hundred megabytes rather than the gigabytes a comparable model needs, and it runs comfortably on a machine with no GPU. If you only want to test output quality and do not care about speed, the `transformers` route works too (it needs a specific fork pinned on the model card), but treat that as a correctness check, not a benchmark. For help choosing between this and the more common runtimes, our [Ollama vs LM Studio vs llama.cpp comparison](/posts/ollama-vs-lm-studio-vs-llama-cpp/) covers the trade-offs.

## Beyond Microsoft: Falcon-E and the wider ternary effort

Microsoft is no longer the only group shipping native 1-bit models. In May 2025, the Technology Innovation Institute released [Falcon-Edge](https://huggingface.co/blog/tiiuae/falcon-edge), a series of 1.58-bit BitNet models at 1B and 3B sizes, each with base and instruction-tuned variants, pre-trained on roughly 1.5 trillion tokens. TII also published `onebitllms`, a small Python package for fine-tuning ternary models, and reported the BitNet training overhead at around 20% compared to standard pre-training. That fine-tuning toolkit matters, because one early criticism of 1-bit LLMs was that nobody outside the original lab could adapt them.

bitnet.cpp already supports several model families beyond the flagship: the Falcon3 series, a Llama3-8B model trained on 100B tokens in the 1.58-bit format, and smaller BitNet checkpoints at 0.7B and 3.3B. The ecosystem is small but it is no longer a single model.

## 1-bit LLMs vs 4-bit quantization: which should you use?

This is the practical decision, and the honest answer depends on what you are optimizing for.

| | Native 1-bit (BitNet b1.58) | 4-bit GGUF (Q4_K_M) |
|---|---|---|
| How it's made | Trained ternary from scratch | Any model, quantized after training |
| Model selection | A handful of models | Nearly every open model |
| Memory | Lowest (~0.4 GB non-emb at 2B) | Low (~1.1 GB at 2B) |
| CPU speed/energy | Best, with bitnet.cpp | Good, with llama.cpp |
| Best frontier model available | ~2B to 8B class | Up to 70B+ |

If you want the most capable model your hardware can hold, 4-bit quantization wins today, simply because you can take a strong 8B or 14B model and run a Q4 build of it. The selection of native 1-bit models is still small, and the largest openly released ones top out well below the best 4-bit-quantizable models. Our [best small language models of 2026](/posts/best-small-language-models-2026/) roundup is dominated by models you would run at Q4 or Q5, not ternary.

If your constraint is energy, CPU-only deployment, or fitting into very little memory on an edge device, native 1-bit is the more interesting direction, and it is the only approach where the efficiency is baked into the math rather than bolted on afterward. For sizing either option against your machine, our [how much RAM you need for local LLMs](/posts/how-much-ram-for-local-llms/) guide applies to both.

## Frequently asked questions

**Can I convert an existing model like Llama to 1-bit?**
Not in any way that preserves quality. 1-bit LLMs must be trained from scratch with quantization-aware training so the weights learn to work as ternary values. Post-training quantization to 1-bit collapses the model. This is the fundamental difference from making a 4-bit GGUF, which is a post-training step.

**Are 1-bit LLMs actually as good as full-precision models?**
At small sizes there is a gap, but it narrows with scale. The 2024 research showed BitNet b1.58 matching FP16 LLaMA on perplexity by 3B parameters. The 2025 BitNet b1.58 2B4T scores a 54.19 task average against 55.23 for full-precision Qwen2.5 1.5B, close but not ahead overall.

**Do I need a GPU to run a 1-bit LLM?**
No, and that is the point. 1-bit LLMs are designed for CPU inference. The bitnet.cpp framework reports running a 100B model on a single CPU at 5 to 7 tokens per second, and the 2B model runs easily on a laptop without a discrete GPU.

**Why doesn't transformers make BitNet faster?**
The standard `transformers` library lacks the specialized kernels for ternary matrix operations, so it runs the model without the efficiency gains. You need bitnet.cpp to see the real speed and energy numbers.

## Where this leaves you

If you want to run something useful today, a 4-bit build of a strong small model is still the pragmatic choice, and the model selection is far wider. But 1-bit LLMs are the one quantization approach where the efficiency is structural rather than a compression applied after the fact, and BitNet b1.58 2B4T is the first release that proves a ternary model can trade blows with full-precision peers at the same size. Download the GGUF, build bitnet.cpp, and watch a 2B model answer questions on a CPU using 0.4 GB of weights. That is the clearest way to understand why this architecture is worth tracking.

## Sources

- [BitNet b1.58 2B4T model card (Hugging Face)](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [BitNet b1.58 2B4T Technical Report (arXiv 2504.12285)](https://arxiv.org/abs/2504.12285)
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (arXiv 2402.17764)](https://arxiv.org/abs/2402.17764)
- [BitNet: Scaling 1-bit Transformers for Large Language Models (arXiv 2310.11453)](https://arxiv.org/abs/2310.11453)
- [microsoft/BitNet (bitnet.cpp) GitHub](https://github.com/microsoft/BitNet)
- [Falcon-Edge: 1.58-bit language models (TII / Hugging Face)](https://huggingface.co/blog/tiiuae/falcon-edge)
