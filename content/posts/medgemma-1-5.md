---
title: "MedGemma 1.5: Google's 4B Medical Vision-Language Model You Can Run Locally"
date: 2026-05-24
draft: false
tags: ["small-models", "slm", "edge-ai", "medgemma", "gemma", "medical-ai", "multimodal", "ollama"]
categories: ["small-ai-models"]
description: "MedGemma 1.5 is Google's 4B medical multimodal model — Gemma 3 backbone, MedSigLIP encoder, 128K context. Runs in 3.3GB on Ollama. Benchmarks, deployment, and the limits."
slug: "medgemma-1-5"
---

MedGemma 1.5 is Google's 4B medical multimodal model, released on January 13, 2026 as part of the Health AI Developer Foundations (HAI-DEF) collection. It is a Gemma 3 backbone paired with a SigLIP image encoder pre-trained on de-identified medical imagery, fine-tuned for radiology, histopathology, ophthalmology, dermatology, and electronic health records. The Q4_K_M GGUF is 3.3 GB on Ollama — the same disk footprint as a generic 4B chat model, but with a vision tower that has actually seen chest X-rays.

This post covers what MedGemma 1.5 is, what changed from the original MedGemma, how it benchmarks, how to run it locally, and — importantly — what it is explicitly not built for.

{{< figure src="https://images.pexels.com/photos/4226139/pexels-photo-4226139.jpeg" alt="Medical professional examining a brain MRI scan on a backlit viewer" caption="MedGemma 1.5 4B can interpret 3D volumes from CT and MRI alongside text. Photo: [Anna Shvets](https://www.pexels.com/photo/4226139/), Pexels" >}}

## What MedGemma 1.5 is

MedGemma 1.5 is a decoder-only Transformer with grouped-query attention, built on the Gemma 3 4B architecture. The vision tower is a SigLIP encoder that Google further pre-trained on de-identified medical data — chest X-rays, dermatology, ophthalmology fundus images, and histopathology slides. Images are normalised to 896×896 and encoded to 256 tokens each. Input is text and vision, output is text only, up to 8192 tokens per generation. Context length is at least 128K tokens.

| Spec | Value |
|---|---|
| Parameters | 4B |
| Backbone | Gemma 3 (decoder-only, GQA) |
| Vision encoder | SigLIP, medically pre-trained |
| Image resolution | 896×896, 256 tokens per image |
| Context length | 128K tokens (minimum) |
| Max output | 8192 tokens |
| Input modalities | Text, vision |
| Output modality | Text only |
| License | Health AI Developer Foundations terms (free for research and commercial) |
| Released | January 13, 2026 |
| HuggingFace | google/medgemma-1.5-4b-it |
| Ollama | medgemma1.5 |

The first MedGemma release at Google I/O 2025 shipped three variants — a 4B multimodal, a 27B text-only, and a 27B multimodal. The 1.5 release updates only the 4B multimodal. The 27B checkpoints remain at 1.0 for now.

## What changed from MedGemma 1

MedGemma 1.5 is not a from-scratch rewrite. The architectural changes are the swap to a Gemma 3 backbone and the upgraded medically-trained SigLIP encoder. The headline additions are capability rather than parameter count:

- 3D volume interpretation for CT and MRI, where prior generations were limited to single 2D slices
- Whole-slide histopathology, evaluating multiple image patches in a single forward pass
- Longitudinal chest X-ray comparison against prior images for the same patient
- Anatomical localisation in chest X-rays via bounding boxes
- Stronger document understanding for lab reports and FHIR-based EHR data

The accuracy lift over MedGemma 1 is largest in the areas where the older model was weakest:

| Task | MedGemma 1 | MedGemma 1.5 | Delta |
|---|---|---|---|
| CT classification (7 conditions, macro acc.) | 58% | 61% | +3 |
| MRI classification (10 conditions, macro acc.) | 51% | 65% | +14 |
| Anatomical localisation (IoU on chest X-ray bounding boxes) | 3% | 38% | +35 |
| Chest X-ray time series (macro acc.) | 61% | 66% | +5 |
| MedQA (text knowledge) | 64% | 69% | +5 |
| EHRQA (electronic health records) | 68% | 90% | +22 |

The anatomical-localisation jump from 3% to 38% IoU reflects a capability that effectively did not exist in MedGemma 1, rather than a tuning improvement. The 22-point EHRQA gain is the most practically useful number in the table — it suggests MedGemma 1.5 can actually parse structured FHIR data well enough to answer queries against a patient record.

## Benchmark scorecard

Here are the published headline numbers from the MedGemma 1.5 model card, for context against general-purpose 4B models:

| Benchmark | Type | MedGemma 1.5 4B |
|---|---|---|
| MedQA (4-op) | Text knowledge | 69.1% |
| MedMCQA | Text knowledge | 59.8% |
| PubMedQA | Text knowledge | 68.2% |
| MMLU Med | Text knowledge | 69.6% |
| MIMIC-CXR (Macro F1, top 5) | Chest X-ray | 89.5% |
| EyePACS | Fundus imaging | 76.8% |
| PathMCQA | Histopathology | 70.0% |
| WSI-Path (ROUGE) | Whole-slide pathology | 49.4 |
| EHRQA | Records | 89.6% |
| EHRNoteQA | Clinical notes | 80.4% |

The WSI-Path ROUGE of 49.4 essentially matches the 0.498 (49.8) scored by PolyPath, the task-specific reference model — a 4B generalist holding its own against a specialised model on its own benchmark. For broader context on how 4B-class generalists perform without medical pre-training, the [Qwen3.5-4B vs Phi-4-mini comparison](/posts/qwen3-5-4b-vs-phi-4-mini/) covers the standard reasoning and code benchmarks that MedGemma 1.5 explicitly does not target.

## How to run MedGemma 1.5 locally

Ollama is the simplest path. The model is published as `medgemma1.5` with five tags:

| Tag | Quantization | Disk |
|---|---|---|
| medgemma1.5:latest | Q4_K_M | 3.3 GB |
| medgemma1.5:4b | Q4_K_M | 3.3 GB |
| medgemma1.5:4b-it-q4_K_M | Q4_K_M | 3.3 GB |
| medgemma1.5:4b-it-q8_0 | Q8_0 | 5.0 GB |
| medgemma1.5:4b-it-bf16 | BF16 (full precision) | 8.6 GB |

To pull and run the default Q4_K_M build:

```bash
ollama pull medgemma1.5
ollama run medgemma1.5
```

For an image-input prompt:

```bash
ollama run medgemma1.5 "Describe the findings in this chest X-ray." ./xray.png
```

If you want maximum fidelity for medical work, pull the BF16 tag instead. The 8.6 GB BF16 build still fits in 16 GB of VRAM with plenty of headroom for context, and aggressive quantisation can degrade fine-grained image reasoning — Google's own benchmarks were run at BF16. For a runtime-by-runtime breakdown of how Ollama compares to LM Studio and raw llama.cpp, see the [runtime comparison post](/posts/ollama-vs-lm-studio-vs-llama-cpp/).

For Python with Transformers, the model is `google/medgemma-1.5-4b-it` on HuggingFace, behind a gated access agreement (the Health AI Developer Foundations terms). Accept the terms on the model page, then:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

model_id = "google/medgemma-1.5-4b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

image = Image.open("xray.png")
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe the findings in this chest X-ray."}
    ]}
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=500)
print(processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Memory at BF16 is roughly 8–9 GB plus per-image overhead. On a 16 GB MacBook Pro, the [Apple Silicon setup guide](/posts/run-small-llms-apple-silicon/) covers how to route this through MLX for better throughput than the PyTorch MPS backend.

{{< figure src="https://images.pexels.com/photos/5723880/pexels-photo-5723880.jpeg" alt="High-resolution radiological images of the human spine displayed on a medical viewer" caption="MedGemma 1.5 handles single images well; multi-image and multi-turn workflows are explicitly out of scope. Photo: [cottonbro studio](https://www.pexels.com/photo/5723880/), Pexels" >}}

## Is MedGemma 1.5 safe to use clinically?

No. This is the most important section in the post and it has a one-word answer that Google repeats on every surface where the model is distributed.

The MedGemma 1.5 model card is explicit: outputs are "not intended to directly inform clinical diagnosis, patient management decisions, treatment recommendations, or any other direct clinical practice applications." The model is a developer foundation — a starting point for building healthcare applications that you then validate, fine-tune, and put through whatever regulatory process your jurisdiction requires.

The known limitations in the model card make the reasoning concrete:

- Multimodal evaluation was primarily on single-image tasks; multi-image workflows are not validated
- The model was not optimised for multi-turn conversation
- It is more sensitive to prompt formatting than the underlying Gemma 3
- Developers are expected to validate on data representative of their target population — age, sex, condition, imaging device

In other words: it is a strong base model for a research prototype, a triage assistant in a controlled setting, a query layer over an EHR, or a fine-tuning target for a narrower task. It is not a diagnostician.

## Who should actually use this

Three groups get genuine value from MedGemma 1.5 today:

Researchers prototyping medical NLP or vision tasks without paying GPT-4o per-call rates, and without sending PHI to a third party. A 4B model that runs on a workstation is also a 4B model that runs in an air-gapped lab environment.

Developers building healthcare-adjacent products that need a general medical-knowledge layer — a patient-facing question router, a documentation assistant, a literature summariser — without committing to a frontier model's API costs. The 10-point MedQA edge over a generic 4B is exactly the kind of base capability that domain pre-training is supposed to deliver.

Fine-tuning teams who want a medically pre-trained starting point. MedGemma 1.5 is positioned by Google as a foundation to specialise further — the released checkpoint outperforms the original MedGemma on every reported task, so it is a strictly better base for any downstream radiology or pathology fine-tune.

If your use case is "I want a small general-purpose model that happens to know some medicine," the answer is still probably Gemma 3 4B or Phi-4-mini fine-tuned on your own data. MedGemma 1.5 earns its weight when the workload is genuinely medical and the inputs include actual medical imagery.

## Sources

- [MedGemma 1.5 on Ollama](https://ollama.com/library/medgemma1.5)
- [Next-generation medical image interpretation with MedGemma 1.5 and MedASR (Google Research blog)](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
- [MedGemma 1.5 model card (Google for Developers)](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [google/medgemma-1.5-4b-it on HuggingFace](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma release collection on HuggingFace](https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4)
- [Health AI Developer Foundations terms of use](https://developers.google.com/health-ai-developer-foundations/terms)
