---
title: "How to Run a Small LLM in Your Browser with WebLLM (No Install, No API)"
date: 2026-05-25
draft: false
tags: ["webgpu", "edge-ai", "local-inference", "small-models"]
categories: ["guides"]
description: "Run a small open LLM (Qwen2.5, SmolLM2, Llama-3.2) entirely in your browser using WebLLM and WebGPU. How it works, browser support, model sizes, and a live demo you can try right now."
slug: "run-llm-in-browser-webllm"
---

You can run a real instruction-tuned language model in a browser tab. No Python, no Ollama, no API key, no server doing the inference. The model downloads once, compiles to GPU shaders the first time you load it, and from then on every token is generated on the same machine the page is rendered on. There is a live demo of exactly this at the end of the post: [tinyweights.dev/try](/try/).

This post covers how it works, what you actually need to make it work, and where the current ceiling is. The runtime that makes it possible is called WebLLM, and the browser feature that makes it fast is WebGPU.

{{< figure src="https://images.pexels.com/photos/270404/pexels-photo-270404.jpeg" alt="Laptop on a wooden desk with code on the screen, representing in-browser development" caption="WebLLM turns a browser tab into a local inference runtime. The same laptop runs the model and renders the page. (Photo: Pexels, free)" >}}

## What WebLLM is

WebLLM is an open-source JavaScript inference engine from MLC AI. It takes a quantized model in MLC's own format and runs it through WebGPU for GPU compute and WebAssembly for the parts that stay on CPU. The project is Apache-2.0 licensed and is currently at version 0.2.83 (April 2026).

The technical work behind it is described in the paper [*WebLLM: A High-Performance In-Browser LLM Inference Engine*](https://arxiv.org/abs/2412.15803) by Charlie F. Ruan, Tianqi Chen, and others. The headline result from that paper is that WebLLM retains up to 80% of native inference performance on the same hardware. The reason it can do that, despite running inside a sandbox, is that MLC-LLM and Apache TVM compile model-specific kernels ahead of time rather than relying on a general-purpose GPU library at runtime.

For an application developer the API looks exactly like the OpenAI client. You construct an engine, you call `chat.completions.create({ messages, stream: true })`, you read deltas off the stream. The local model is a drop-in for the wire protocol you already know.

## What WebGPU is and why it matters here

WebGPU is the browser API that exposes GPU compute to JavaScript. Without it, an in-browser LLM is unusable; CPU-only inference of even a 0.5B parameter model in WebAssembly is too slow to feel like a chat. WebGPU is the thing that makes a sub-2B model genuinely interactive on a laptop integrated GPU.

The current state of WebGPU support, per [caniuse.com](https://caniuse.com/webgpu):

| Browser | WebGPU status |
|---|---|
| Chrome / Edge | Shipping since v113 (Windows, macOS, ChromeOS). Android 12+ from v121. |
| Safari (macOS, iOS) | Shipping in Safari 26 (macOS Tahoe, iOS 26). |
| Firefox | Shipping on Windows since v141. Linux and macOS still in progress. |
| Chrome on Android | v121+ on devices with recent Qualcomm or ARM GPUs. |

Global support is around 82% of browsers as of mid-2026. That is high enough to ship a feature behind a capability check, but low enough that the fallback message matters. The [WebLLM /try demo on this site](/try/) feature-detects `navigator.gpu` and shows a graceful error for browsers without it instead of failing silently.

## How a model gets into your browser tab

The flow is shorter than it sounds. When the page loads WebLLM, it does four things:

1. Requests a WebGPU adapter from the browser and inspects its features (in particular, whether `shader-f16` is available, which controls whether fp16 activations are usable).
2. Downloads the quantized model weights, in 4-bit MLC format, into the browser's IndexedDB. This is the slow step; it is between roughly 200 MB and 850 MB depending on which model you picked.
3. Compiles the model's WebGPU shaders for your specific GPU. This happens once per model per machine.
4. Starts the inference loop. Tokens stream out of the engine the same way they would from a server-sent-events response.

On a second visit, IndexedDB already has the weights and shaders, so the model is resident in seconds rather than minutes. The download cost is paid once.

## Which models are small enough

A model that runs in a browser tab is bottlenecked by three things: download size, GPU memory, and how slow people are willing to tolerate for the first token. In practice that caps the usable range at around 1.5B parameters at 4-bit quantization on consumer hardware. The four models below are the ones the demo on this site uses, all of them prebuilt MLC quantizations:

| Model | Params | 4-bit download | Notes |
|---|---|---|---|
| SmolLM2-360M-Instruct | 360M | ~200 MB | HuggingFace's smallest. Fast, but coherence drops off quickly past one or two turns. |
| Qwen2.5-0.5B-Instruct | 500M | ~280 MB | Best balance of size and quality at the sub-1B scale. |
| Llama-3.2-1B-Instruct | 1B | ~670 MB | Meta's smallest Llama 3.2. Noticeably better instruction following. |
| Qwen2.5-1.5B-Instruct | 1.5B | ~840 MB | The most capable model that fits comfortably in a browser. Biggest download. |

Each model ships in two MLC builds, `q4f16_1` and `q4f32_1`. The fp16 build is smaller and faster but requires WebGPU's `shader-f16` feature; the fp32 build runs on every GPU. The right pattern is to probe for `shader-f16` at runtime, try the f16 build first, and fall back to f32 if compilation fails. Some drivers report `shader-f16` as supported and then error on actual kernel use, so the fallback should be on real kernel failure rather than capability flag alone.

This is not the same shortlist as the one in [Best Small Language Models in 2026](/posts/best-small-language-models-2026/). The browser ceiling is lower than what your local Ollama install can handle. A 3B or 4B model is comfortable on a recent laptop CPU via llama.cpp, but pushing those weights through a browser sandbox is currently a bad fit.

{{< figure src="https://images.pexels.com/photos/2582937/pexels-photo-2582937.jpeg" alt="Close-up of a computer GPU and circuit board, representing GPU compute via WebGPU" caption="WebGPU is the API that turns the integrated GPU in a laptop into something a browser tab can dispatch work to. Without it, an in-browser LLM is a CPU-only toy. (Photo: Pexels, free)" >}}

## How fast is it, really?

The paper claims WebLLM retains up to roughly 80% of the decoding throughput of native MLC-LLM on the same device. The published comparison runs on an Apple MacBook Pro M3 Max, with 4-bit quantized models in Chrome Canary against native Metal kernels. Two of the headline numbers from Table 1 of the paper: Llama-3.1-8B at 41.1 tok/s in the browser versus 57.7 tok/s native (71.2% retained), and Phi-3.5-mini at 71.1 tok/s in the browser versus 89.3 tok/s native (79.6% retained).

In practice what you feel on a typical laptop with the demo on this site:

- Time to first token on Qwen2.5-0.5B on a recent integrated GPU is in the half-second to two-second range.
- Decode throughput on the same model and hardware sits between 20 and 60 tokens per second.
- The 1.5B model on the same machine drops roughly in proportion to the parameter count, since the bottleneck is GPU memory bandwidth, not flops.

Those numbers are well past "interactive". For a 4-bit 0.5B parameter chat model, that is essentially native speed.

## Why this is worth caring about

Most browser AI demos you have used are a chat UI wrapped around a hosted API. They cost the operator money per request, they need a network round-trip per token, and they retain prompts. A model that runs in your tab inverts all three.

It costs the site that hosts it nothing in compute. The marginal cost of one user generating a million tokens is zero. It works on a plane. It works in a hospital. It works after the API provider raises the price tenfold or simply turns the endpoint off. The trade-off is capability; a sub-2B model is dramatically less smart than a frontier model. That is the same trade-off [the rest of this blog](/posts/what-can-you-do-with-local-small-llm/) is built on. The bet is that "less smart but runs anywhere, for free, forever" wins for more workloads than people currently think.

The privacy angle is also real, not marketing. There is no API call to read prompts on. There is no provider's privacy policy to update. Your prompts and the model's responses never leave the tab.

## Q: Do I need to know JavaScript to use a model in the browser?

No. If you only want to try one out, you do not need to write any code at all. There is a live demo on this site at [tinyweights.dev/try](/try/) that loads one of the four models in the table above and lets you chat with it. The demo runs entirely in your browser; nothing is sent to a server.

If you do want to build something with WebLLM yourself, the API call is short:

```javascript
import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.83";

const engine = await webllm.CreateMLCEngine(
  "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
  { initProgressCallback: (r) => console.log(r.text, r.progress) },
);

const stream = await engine.chat.completions.create({
  messages: [{ role: "user", content: "What is a small language model?" }],
  stream: true,
  temperature: 0.7,
  max_tokens: 400,
});

for await (const chunk of stream) {
  const delta = chunk.choices?.[0]?.delta?.content || "";
  process.stdout.write(delta);
}
```

That is the entire client-side integration. Pair it with a UI, route the model selection through a dropdown, and you have something resembling the demo page on this site. The whole runtime is included from a CDN; there is no build step required.

## Limits worth knowing before you ship

WebLLM does not free you from every constraint of running a model locally. The first load on a slow connection takes between 15 and 90 seconds depending on which model the user picked, and there is no way to make a 700 MB weights download instant. The IndexedDB cache helps repeat visits, but the cold start is a real cost.

Mobile support is uneven. iOS 26 and a recent Android Chrome on a flagship phone will work; older devices and budget Android phones often do not have enough free GPU memory for the 1B and 1.5B options.

And small models hallucinate. A 0.5B parameter chat model will confidently make up API names, library functions, and historical dates. The right framing for a browser-resident model is "a fast, private, free assistant for short tasks", not "ChatGPT in a tab".

## Try it now

The [live demo at tinyweights.dev/try](/try/) loads any of the four models above directly in your browser. Pick one from the dropdown, type a message, and the model loads on first send and streams a reply. Weights are cached in IndexedDB after the first download, so a return visit is instant. If you want a deeper read on the broader space these models live in, the [Best Small Language Models in 2026](/posts/best-small-language-models-2026/) post is the right next link.

If your hardware allows it, the same idea scales up. With Ollama or LM Studio you can run a 3B or 8B model on the same laptop. But there is something specific worth keeping about the in-browser version: the lowest-friction "first contact" anyone can have with a real language model is now a single click, no install, no key, no signup.

## Sources

- [WebLLM GitHub repository](https://github.com/mlc-ai/web-llm)
- [WebLLM: A High-Performance In-Browser LLM Inference Engine (Ruan et al., arXiv:2412.15803)](https://arxiv.org/abs/2412.15803)
- [WebLLM documentation](https://webllm.mlc.ai/docs/)
- [MLC AI prebuilt models on HuggingFace](https://huggingface.co/mlc-ai)
- [WebGPU browser support (caniuse.com)](https://caniuse.com/webgpu)
- [WebGPU API reference (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
- [WebGPU is now supported in major browsers (web.dev)](https://web.dev/blog/webgpu-supported-major-browsers)
