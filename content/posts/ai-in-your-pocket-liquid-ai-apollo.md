---
title: "AI in Your Pocket: How Liquid AI’s Apollo App Lets You Run Chatbots Completely Offline"
date: "2026-03-24"
draft: false
tags: ["small-models", "slm", "edge-ai", "LFM2", "Apollo"]
categories: ["small-ai-models"]
description: "Discover how Liquid AI's Apollo app and LEAP platform bring fully private, offline small language models (SLMs) straight to your smartphone."
slug: "ai-in-your-pocket-liquid-ai-apollo"
author: "Edwin"
---

For years, the phrase “AI chatbot” has carried the inherent assumption of a round-trip to the cloud. You type a prompt, it zips off to a server farm somewhere, crunches the data on massive GPUs, and beams the text back to your screen. But what happens when you’re on a Wi-Fi-less airplane? Or working offline in a remote area? Or discussing highly sensitive enterprise work you’d never want stored on a provider’s server?

Enter **Apollo**, an app originally independent but recently acquired and completely reimagined by Liquid AI. Designed to solve precisely these problems, Apollo allows you to run "Small Language Models" (SLMs) natively on the hardware in your pocket. Combined with Liquid’s new deployment platform, LEAP, Apollo is fundamentally changing the landscape of on-device AI. 

If you've been searching for an accessible way to test on-device inference without navigating complex GitHub repositories or running command-line scripts, Apollo is exactly what you need. Best of all? Following the acquisition, the app has gone entirely free on both the App Store and Google Play.

## The Problem: The Hidden Costs of Cloud-Based AI

Until the advent of hyper-efficient SLMs, running a capable chatbot meant relying on cloud APIs. A typical interaction with services like standard ChatGPT or major open-source inference endpoints requires an active, stable internet connection.

While this architecture has enabled the generative AI boom, it brings three major, often deal-breaking frictions for specific use-cases:

1.  **Latency and Speed Limits:** Even the fastest cloud models require network round-trips. Your text generation speed is irrevocably tied to your connection quality. If you are on a weak cellular signal, productivity grinds to a halt.
2.  **The Connectivity Requirement:** You are entirely dead in the water on a plane, subway, or in rural areas. Having an "intelligent assistant" is significantly less valuable if it clocks out the moment you lose LTE.
3.  **Privacy and Security Concerns:** In an era where data is currency, enterprise workers, developers, and privacy advocates logically hesitate to send sensitive source code, confidential HR drafts, or personal journaling to an external server. Once data leaves your device, you immediately lose a degree of control over how it might be logged, analyzed, or used for future model training by third-party data handlers.

## The Solution: Apollo and the Liquid Edge AI Platform (LEAP)

Liquid AI recognized this bottleneck. Their answer is two-fold: an underlying developer toolkit to make edge AI feasible, and an accessible, consumer-facing app to serve as the playground. 

Recently, Liquid AI officially launched **LEAP (the Liquid Edge AI Platform)**, an OS- and model-agnostic developer toolkit. LEAP ships with cross-platform Edge SDKs for both Android and iOS, meaning developers can now deploy SLMs directly into compiled mobile apps with just a few lines of code—essentially treating local inference with the same developer experience as calling a remote HTTP API. Hand-in-hand with this launch, Liquid brought **Apollo** fully into their ecosystem, providing native support for the entire LEAP library as a ready-to-use testing ground.

[![Mobile Device](https://upload.wikimedia.org/wikipedia/commons/0/02/Evolution_Directions_of_Mobile_Device.jpg)](https://commons.wikimedia.org/wiki/File:Evolution_Directions_of_Mobile_Device.jpg)
*Evolution Directions of Mobile Device. [Source Link](https://commons.wikimedia.org/wiki/File:Evolution_Directions_of_Mobile_Device.jpg) · Marcin Grygiel aka Marcin-prv*

### How Small Language Models Change the Math

Apollo leverages Small Language Models (SLMs)—specifically, models like Liquid's own new generation of explicitly designed **LFM2** models.

Large models (LLMs) are generalists. They are designed to do many things well, requiring hundreds of gigabytes of VRAM. Small models, by contrast, are either average at many tasks or excellent at a select few. Liquid’s architecture aggressively compresses state-of-the-art capabilities down to file sizes built explicitly for the edge. 

Instead of taking up massive storage or instantly draining your battery life, LEAP offers a curated library of checkpoints and advanced quantization options. This means developers can find the exact balance of precision and optimization for their specific on-device use case. Models optimized for LEAP can be compressed down to as small as 300MB. Furthermore, they run smoothly on modern mobile devices equipped with 4GB+ RAM, natively leveraging internal compute resources to maximize hardware efficiency while minimizing power consumption.

Once you download the model weights in Apollo (ideally via Wi-Fi), the inferencing happens physically on your smartphone’s processor. No server is required, and latency is effectively zero. This continues the 2026 trend toward efficiency, proving that you do not always need massive compute clusters to run practical AI.

## Everyday Uses for the Offline AI Era

Having capable AI on-device completely alters when, where, and how you decide to use it. When connectivity and privacy barriers are removed, the utility of language models skyrockets. 

### 1. Drafting Without Connection
Next time you’re on a long flight without costly Wi-Fi access, you don't have to pause your workflow. You can still outline emails, structure complex blog posts, organize meeting notes, or brainstorm project names right inside the Apollo app. By the time you touch down, your drafts are ready to send.

### 2. The 100% Private Sounding Board
We all have moments where we need a second opinion on a sensitive HR email, an unannounced product launch, or a highly confidential coding problem. Because the Apollo AI runs purely locally on your phone, there are absolutely zero outbound data transmissions during inference. Your prompts and the model's outputs never leave the secure enclave of your device. 

### 3. A Sandbox for Developers and Tinkerers
For the developer community, Apollo isn't just a consumer chat app—it is an accessible playground. As Liquid AI themselves point out, developers can use Apollo to "vibe-check" new small models instantly. You can stress-test a model's tone, inferencing speed (tokens-per-second), and efficiency live right on your target hardware. 

Additionally, recent updates to Apollo introduced "Nano app experiences," utilizing specifically trained LFM Nano models for precise, programmatic tasks. For instance, developers can test extracting unstructured data into strictly formatted JSON, or run bidirectional Japanese/English translation without exposing data to the cloud. Furthermore, support for Vision models (LFM2-VL) allows you to send images and interact with the visual world entirely offline.

[![Artificial Intelligence](https://upload.wikimedia.org/wikipedia/commons/6/64/Dall-e_3_%28jan_%2724%29_artificial_intelligence_icon.png)](https://commons.wikimedia.org/wiki/File:Dall-e_3_(jan_%2724)_artificial_intelligence_icon.png)
*Artificial Intelligence. [Source Link](https://commons.wikimedia.org/wiki/File:Dall-e_3_(jan_%2724)_artificial_intelligence_icon.png) · JPxG*

## Bring Your Own Backend

Apollo’s sheer simplicity is its superpower, but it refuses to hamstring power users. 

While the app champions offline privacy, it also serves as an incredibly robust "bring your own key" (BYOK) client. If you want top-tier cloud power when you *are* connected, Apollo allows you to plug in your OpenRouter API key. This suddenly opens the door to chatting with nearly every open and closed-source model available on the web (ranging from Meta Llama 3 to OpenAI's GPT-4), securely through Apollo's unified interface. 

Furthermore, Apollo provides explicit **Custom Backend support**. For the true tinkerers hosting massive local LLMs via tools like LM Studio, Ollama, or vLLM on a home desktop (or a customized cheap VPS), you can directly point Apollo to your server's endpoint. 

```bash
# Example: If running an Ollama or LM Studio backend locally 
# You can point your Apollo custom backend to your network IP
http://192.168.1.50:11434/v1
```

This effectively turns Apollo into a sleek remote client mobile app that securely queries your heavily-parameterized models from anywhere, acting as your personal, self-hosted version of ChatGPT.

## Embracing the Future of Local Generative AI

The move to make Apollo fully free upon acquisition by Liquid AI removes the final barrier to entry for local AI usage on mobile platforms. 

Whether you are frustrated by the constant cloud API subscriptions, blocked by the lack of connectivity during travel, or mandated by strict privacy requirements, Apollo bridges the gap beautifully. It delivers practical, reliable generative AI without compromising your data or your battery life. We are entering an era where AI is less a service we connect to, and more a foundational utility running silently and powerfully alongside the rest of our device's native hardware.

## Sources
* [Liquid AI Apollo Product Page](https://www.liquid.ai/apollo)
* [Liquid AI Blog: LEAP and Apollo Announcement](https://www.liquid.ai/blog/liquid-ai-launches-leap-and-apollo-bringing-edge-ai-to-every-developer)
* [VentureBeat: Liquid AI's LEAP](https://venturebeat.com/ai/finally-a-dev-kit-for-designing-on-device-mobile-ai-apps-is-here-liquid-ais-leap)
* [Google Play Store: Apollo - Powered by Liquid](https://play.google.com/store/apps/details?id=ai.liquid.chatapp&hl=en_US)
* [Apple App Store: Apollo - Powered by Liquid](https://apps.apple.com/us/app/apollo-powered-by-liquid/id6448019325)
* [Reddit r/macapps: Apollo has gone free](https://www.reddit.com/r/macapps/comments/1m2tqo7/apollo_the_native_local_llm_app_has_gone_free/)