---
title: "The Best Local Vision Language Models in 2026: Small VLMs You Can Actually Run"
date: 2026-07-09
draft: false
tags: ["vision", "comparison", "qwen", "edge-ai", "small-models"]
categories: ["comparisons"]
description: "A practical comparison of the best local vision language models in 2026, from 4B VLMs to 32B, with real MMMU and DocVQA benchmarks and VRAM estimates."
slug: "best-local-vision-language-models-2026"
---

If you want the best local vision language models in 2026, the honest answer is that two families dominate the sub-35B range: Alibaba's Qwen3-VL and Google's Gemma 3. Both give you a model that reads an image, answers questions about it, and pulls text out of a photographed document, all on a single consumer GPU. This post compares the small VLMs worth running today, with real benchmark numbers instead of vibes, and tells you which one to pick for your hardware and task.

We already have a [roundup of the best small language models in 2026](/posts/best-small-language-models-2026/) for text-only work. This is the vision counterpart: models that take pixels as input, small enough to run offline.

![Close-up of a security camera lens against a plain wall](https://images.pexels.com/photos/7635126/pexels-photo-7635126.jpeg)
*A vision language model is a machine that looks and describes what it sees. In 2026 you can run one on a laptop GPU. Photo by Erik Mclean on Pexels (free to use).*

## What counts as a local VLM

For this comparison a model has to clear three bars. It accepts image input alongside text. It fits under roughly 35B total parameters, which is the ceiling for comfortable local inference on a 24 GB card or a 32 GB Mac. And it has an actual quantized build people run today, not a research checkpoint. That rules out frontier multimodal models that only exist behind an API.

One number matters more than any benchmark: total parameters, because that sets your memory floor. A vision model also needs extra memory beyond the weights for its image encoder and the image tokens themselves, so budget more headroom than you would for a text model of the same size. Our guide on [how much RAM you actually need for local LLMs](/posts/how-much-ram-for-local-llms/) covers the weights math; add a gigabyte or two for the vision side.

## The comparison

Here are the current small VLMs from the two leading open families, with MMMU (a college-level multimodal reasoning benchmark) and DocVQA (document question answering) scores for the Instruct editions. VRAM figures are rough estimates for a Q4_K_M GGUF, weights only.

| Model | Params (total) | MMMU (val) | DocVQA | Context | License | ~VRAM @ Q4 |
|---|---|---|---|---|---|---|
| Gemma 3 4B | 4B | 48.8 | 75.8 | 128K | Gemma | ~3 GB |
| Qwen3-VL-4B | 4B | 67.4 | 95.3 | 256K | Apache 2.0 | ~3.5 GB |
| Qwen3-VL-8B | 9B | 69.6 | 96.1 | 256K | Apache 2.0 | ~6 GB |
| Gemma 3 12B | 12B | 59.6 | 87.1 | 128K | Gemma | ~8 GB |
| Gemma 3 27B | 27B | 64.9 | 86.6 | 128K | Gemma | ~16 GB |
| Qwen3-VL-32B | 32B | 76.0 | 96.9 | 256K | Apache 2.0 | ~20 GB |

Scores are for the standard Instruct checkpoints. The Qwen3-VL "Thinking" editions post higher numbers on reasoning-heavy tasks at the cost of longer, slower outputs. Gemma 3 figures come from its technical report; Qwen3-VL figures come from the official Qwen3-VL model cards.

The table has one finding that jumps out. Qwen3-VL-4B scores 67.4 on MMMU, above Gemma 3 27B's 64.9, while using roughly a seventh of the parameters. That gap is mostly a release-date story: Gemma 3 landed in March 2025, Qwen3-VL in October 2025, and seven months is a long time in this field. It does not make Gemma 3 a bad choice, but it does mean that if you are starting fresh in 2026, Qwen3-VL is the stronger default.

## The best local VLM for most people: Qwen3-VL-8B

Qwen3-VL-8B (9B total parameters, dense) is the model I would install first. It scores 69.6 on MMMU and 96.1 on DocVQA, runs in about 6 GB at Q4, and carries an Apache 2.0 license, so you can use it commercially without reading a lawyer's worth of fine print. Native context is 256K tokens, extensible toward 1M, which matters when you feed it a long PDF as a sequence of page images.

Beyond static images it handles video input, OCR across 32 languages, and GUI grounding, meaning it can look at a screenshot, identify buttons and fields, and drive an agentic loop. That last capability is what separates a 2026 VLM from the caption-and-VQA models of two years ago.

![Qwen3-VL-4B and 8B Instruct benchmark comparison chart](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_4b_8b_vl_instruct.jpg)
*Qwen3-VL-4B and 8B Instruct across multimodal benchmarks. Chart from the official Qwen3-VL model card (Apache 2.0).*

If 6 GB is more than your card has, drop to Qwen3-VL-4B. It gives up only about two points of MMMU and one of DocVQA versus the 8B, and it fits in roughly 3.5 GB at Q4. For a 6 GB or 8 GB laptop GPU, the 4B is the sweet spot.

## The most compatible small VLM: Gemma 3 4B

Numbers are not everything. Gemma 3 4B loses to Qwen3-VL-4B on every vision benchmark here, but it wins on something the benchmarks do not show: runtime support. Gemma 3 shipped with day-one integration in Ollama, llama.cpp, and MLX, and it covers 140-plus languages of text. If you want a vision model that works the moment you type `ollama pull`, without hunting for a runtime build that supports the vision projector, Gemma 3 4B is the path of least resistance. It runs in about 3 GB at Q4 and takes text and image input with a 128K context window.

This is the recurring tension with local VLMs. The vision half of a model needs explicit runtime support, and that support lands later than text-only support does. A brand-new VLM can top the leaderboard and still be a pain to run for a few weeks until llama.cpp and Ollama catch up. Before you commit to a model, check that your runtime already handles its image path.

## The best for documents and OCR

Reading text out of images is the single most useful thing a small VLM does, and it is where Qwen3-VL pulls ahead hardest. Qwen3-VL-8B's 96.1 on DocVQA and Qwen3-VL-4B's 95.3 both clear every Gemma 3 size, including the 27B. If your workload is invoices, forms, receipts, or scanned reports, a Qwen3-VL model is the pick.

![Close-up of a camera lens showing optical elements](https://images.pexels.com/photos/8185667/pexels-photo-8185667.jpeg)
*OCR and document understanding are the highest-value tasks for a local VLM. Photo by John Rapone on Pexels (free to use).*

The other strong document model is MiniCPM-V 4.5 from OpenBMB, an 8.7B model that posts a 77.0 average on OpenCompass and is built specifically for OCR, high-resolution documents, and video. InternVL3.5-8B from OpenGVLab is worth a look too: it scores 73.4 on MMMU (val) and 840 on the 1000-point OCRBench. Both are viable if you want an alternative to the Qwen and Gemma duopoly, though runtime support is thinner.

## Grounding, detection, and pointing: Moondream 3

Most VLMs describe an image. Moondream 3 (Preview) points at things in it. It is a 9B mixture-of-experts model with only 2B active parameters, built for object detection, visual grounding, pointing, and structured output rather than long conversations. The sparse design (64 experts, 8 active per token) keeps it fast, and its 32K context is enough for few-shot prompts and multi-image tasks. If you are building something that needs coordinates back, like "find every price tag in this shelf photo and return bounding boxes," Moondream 3 is purpose-built for that in a way a general chat VLM is not.

## The top-end local pick

If you have a 24 GB card or a Mac with 32 GB or more of unified memory, the ceiling rises. Qwen3-VL-32B scores 76.0 on MMMU, the highest of any model here, and runs in roughly 20 GB at Q4. Gemma 3 27B is the more broadly supported 27B option at 64.9 MMMU and about 16 GB. Neither is a small model in the phone-in-your-pocket sense, but both stay inside the envelope of a single high-end consumer GPU, which is the line this blog cares about.

## Specialists worth knowing

General VLMs are not always the right tool. A few narrow models beat them inside their lane, and we have hands-on guides for the ones you can run locally:

### Medical imaging
[MedGemma 1.5](/posts/medgemma-1-5/) is a 4B model tuned for chest X-rays, dermatology, and pathology. It is not a general chat model, and you should not treat it as a diagnostician, but for parsing medical images it outperforms a general 4B VLM by a wide margin.

### Video and object detection
[Reka Edge](/posts/run-reka-edge-locally-vision-model-mac/) is a 7B vision-language model with strong video understanding and object detection, and our walkthrough covers running it on a Mac or PC.

### Vision plus tool use
[Ministral 3](/posts/run-ministral-3-locally/) ships vision across its 3B, 8B, and 14B sizes with native tool calling, which makes it a good base for agentic pipelines that mix images and function calls. For a 4B general option with on-demand reasoning, our [Qwen3.5-4B comparison](/posts/qwen3-5-4b-vs-phi-4-mini/) covers a model that folds vision into the same Qwen family.

## What is the best vision language model you can run locally?

For most people the best local vision language model in 2026 is Qwen3-VL-8B. It leads the small-model field on both multimodal reasoning (69.6 MMMU) and document reading (96.1 DocVQA), runs in about 6 GB at Q4, and ships under a permissive Apache 2.0 license. Step down to Qwen3-VL-4B if you have 6 to 8 GB of VRAM, or use Gemma 3 4B if you want the widest runtime support and the least setup friction.

## How much VRAM do you need to run a VLM locally?

A 4B VLM at Q4 needs about 3 to 4 GB for weights, so an 8 GB card runs it comfortably with room for image tokens. An 8B model wants 6 GB or so at Q4, which fits a 12 GB card. The 27B and 32B models need 16 to 20 GB, putting them on 24 GB cards or Macs with plenty of unified memory. Always leave extra headroom beyond these figures, because a vision model spends memory on its image encoder and on the tokens each image expands into, and a high-resolution photo can add up quickly. Our [GGUF quantization guide](/posts/gguf-quantization-levels-q4-q5-q8/) explains what you trade away at each level.

## Can you run a vision model on CPU?

Yes, but expect it to be slow. A 4B VLM will run on CPU through llama.cpp, and for a single image and a short answer you might wait several seconds rather than get an instant response. CPU inference is fine for batch jobs where latency does not matter, such as tagging a folder of photos overnight. For interactive use you want a GPU, or at least a modern laptop with fast unified memory. The vision encoder is the part that hurts most on CPU, since encoding a high-resolution image is compute-heavy before the language model even starts generating.

## The short version

Install Qwen3-VL-8B if you have 6 GB of VRAM or more, Qwen3-VL-4B if you are tighter than that, and Gemma 3 4B if you want the smoothest first-run experience. Reach for MiniCPM-V 4.5 or InternVL3.5 when you want an alternative, Moondream 3 when you need detection and pointing, and a specialist like MedGemma when your images are not the everyday kind. Then check your runtime supports the model's vision path before you download eight gigabytes of weights.

## Sources

- [Qwen3-VL GitHub repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-8B-Instruct model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen3-VL-4B-Instruct model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Qwen3-VL Technical Report (arXiv)](https://arxiv.org/abs/2511.21631)
- [Qwen3-VL-4B vs 8B benchmarks and VRAM guide (Codersera)](https://codersera.com/blog/qwen3-vl-4b-vs-qwen3-vl-8b-benchmarks-vram-guide/)
- [Gemma 3 Technical Report (arXiv)](https://arxiv.org/abs/2503.19786)
- [Gemma 3 announcement (Google Developers Blog)](https://blog.google/technology/developers/gemma-3/)
- [InternVL3.5 Technical Report (arXiv)](https://arxiv.org/abs/2508.18265)
- [MiniCPM-V 4.5 Technical Report (arXiv)](https://arxiv.org/abs/2509.18154)
- [Moondream 3 Preview announcement](https://moondream.ai/blog/moondream-3-preview)
