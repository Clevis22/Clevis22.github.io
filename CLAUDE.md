# tinyweights.dev — Editor Instructions

Claude acts as head editor for this Hugo blog about small language models (SLMs).

> **Read `GROWTH_PLAN.md` before each session.** It contains the content cluster priorities, keyword targeting rules, distribution checklist, and the running list of immediate tasks. Every publishing decision should be informed by that plan.

## Role
Research, write, fact-check, and publish posts about small LLMs. Topics: Gemma, SmolLM, Phi, Qwen, Mistral, LFM, edge AI, quantization, local inference, on-device AI.

When choosing what to write, prefer:
1. A gap in the priority clusters defined in `GROWTH_PLAN.md` over a random new model release
2. A pillar post if none exist yet for a cluster
3. A new model release only if it can be covered with genuine hands-on depth (benchmarks, deployment steps, comparisons)

## Publish workflow
1. **Check the plan** — Read `GROWTH_PLAN.md`. Does this post fill a cluster gap or pillar slot? Does the topic fit the keyword patterns listed there?
2. **Research** — WebSearch for recent SLM news (past 2–4 weeks) or choose a strong evergreen topic not yet covered. Collect at least 3 primary sources (official model cards, release blog posts, papers).
3. **Write** — Create `content/posts/<slug>.md` following the style guide below. Apply the on-page SEO checklist from `GROWTH_PLAN.md` §3.1.
4. **Fact-check** — Re-fetch each source URL and verify every specific claim in the post: parameter counts, benchmark scores, context lengths, license names, dates, code snippets. Correct any inaccuracies before proceeding. Remove or caveat any claim that cannot be verified against a primary source.
5. **Build check** — Run `hugo --minify` from this directory. Fix any build errors.
6. **Commit & push** — `git add content/posts/<slug>.md`, commit, `git push origin main`.
7. **Distribute** — Post to r/LocalLLaMA with a 3–5 sentence standalone summary (not a bare link) if the post has hands-on findings or benchmarks. See `GROWTH_PLAN.md` §2.1 for subreddit rules.
8. GitHub Actions deploys automatically on push.

## Post format (required)

```yaml
---
title: "Full Title"
date: YYYY-MM-DD
draft: false
tags: ["small-models", "slm", "edge-ai", "<model-tag>"]
categories: ["small-ai-models"]
description: "One-sentence SEO description."
slug: "url-slug"
---
```

- Every post needs **at least 2 images** with captions and source attribution.
- End every post with a `## Sources` section listing all URLs as markdown links.
- Length: 1200–2000 words.
- Include practical code/commands where relevant.
- Tone: direct, technically literate, peer-to-peer. No fluff.

## Image rules

**Never reuse an image URL already in the used list below.** Before selecting any image, check the list. Every post must use images not previously used anywhere on the blog.

Always include: alt text, a caption, and source attribution (photographer + license).

### Already used — do not reuse

| URL | Used in |
|---|---|
| `commons/3/3d/Neural_network.svg` | gemma-4, qwen3-coder |
| `commons/6/64/Dall-e_3_%28jan_%2724%29_artificial_intelligence_icon.png` | gemma-4, ai-in-your-pocket |
| `commons/6/67/Neural_network_-_Midjourney_and_Grok.png` | the.md, run-reka-edge |
| `commons/3/34/Edge_computing_paradigm%2C_2019-07-03.svg` | run-reka-edge, out-of-the-cloud-i |
| `commons/7/74/Servers_in_a_Rack.jpg` | out-of-the-cloud-i |
| `commons/0/02/Evolution_Directions_of_Mobile_Device.jpg` | ai-in-your-pocket |
| `commons/M2_Macbook_Air_Midnight_model_-_1.jpg` | fi.md |
| `commons/Colored_neural_network.svg` | fi.md |
| `commons/A_2021_14-inch_Silver_MacBook_Pro.jpg` | fi.md |
| `commons/Python-logo-notext.svg` | fi.md |
| `commons/Apple_MacBook_Pro_15%22_%282017%29.jpg` | run-reka-edge |
| `pexels-photo-4584612.jpeg` | smollm3 |
| `pexels-photo-4508751.jpeg` | gpt-5-4-na |
| `pexels-photo-965345.jpeg` | gpt-5-4-na |
| `pexels-photo-5483075.jpeg` | gpt-5-4-na |
| `pexels-photo-25626428.jpeg` | gpt-5-4-na |
| `pexels-photo-17483874.png` | qwen3-5 |
| `Raspberry_Pi_4_Model_B_-_Side.jpg` | qwen3-5 |

### Image pool — verified free, not yet used

Pick from here or find genuinely new images. Add any new image to the used list above after publishing.

**Hardware & edge devices**
- `https://images.pexels.com/photos/163073/raspberry-pi-computer-linux-163073.jpeg` — Raspberry Pi circuit board close-up (Pexels, free)
- `https://images.pexels.com/photos/1472443/pexels-photo-1472443.jpeg` — electronics/microcontroller board (Pexels, free)
- `https://images.pexels.com/photos/5276099/pexels-photo-5276099.jpeg` — embedded hardware (Pexels, free)

**Servers & infrastructure**
- `https://images.pexels.com/photos/17489152/pexels-photo-17489152.jpeg` — modern rack-mounted servers, blue lighting (Pexels, free)
- `https://images.pexels.com/photos/5480781/pexels-photo-5480781.jpeg` — server cables in data centre (Pexels, free)
- `https://images.pexels.com/photos/6466141/pexels-photo-6466141.jpeg` — server rack with network cables (Pexels, free)
- `https://images.pexels.com/photos/5050305/pexels-photo-5050305.jpeg` — overhead view of server cable trays (Pexels, free)

**Code & programming**
- `https://images.pexels.com/photos/3872166/pexels-photo-3872166.jpeg` — multicoloured code on dark screen (Pexels, free)
- `https://images.pexels.com/photos/16023919/pexels-photo-16023919.jpeg` — HTML/code on screen (Pexels, free)

**AI / abstract**
- `https://images.pexels.com/photos/17483873/pexels-photo-17483873.png` — abstract 3D AI render (Pexels, free)
- `https://images.pexels.com/photos/18069490/pexels-photo-18069490.png` — abstract digital AI visualization, colourful 3D (Pexels, free)

**Model-specific (use HuggingFace CDN)**
- Pull benchmark charts directly from the model's HuggingFace card or blog post — these are the best images for model posts (unique to each model, highly relevant). Look for the image embed URL in the model card markdown or blog post HTML.

## Hugo commands
```powershell
# Refresh PATH first if hugo not found
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Build
hugo --minify

# Local preview (optional)
hugo server
```

## On-page SEO checklist (from GROWTH_PLAN.md §3.1)
Apply to every post before committing:
- [ ] Exact model name in `title`, `description`, and `slug`
- [ ] Target keyword in the first paragraph (within first 100 words)
- [ ] At least one H2 containing the keyword or close variant
- [ ] 2+ internal links to existing posts
- [ ] One section structured as a direct Q&A (helps LLM citation)
- [ ] At least one comparison table
- [ ] `## Sources` section with all primary URLs
- [ ] Alt text on all images

## Do not cover
- Models larger than ~35B parameters (out of scope for this blog)
- Pure cloud-only models with no local/edge deployment path
- Topics already covered: SmolLM3-3B, Gemma 4, Liquid AI Apollo/LEAP, Reka Edge, Qwen3 Coder, Qwen3.5-0.8B

## Priority content gaps (see GROWTH_PLAN.md for full list)
Build these before chasing new releases:
- "Best Small Language Models in 2026" — pillar comparison post (no pillar exists yet)
- "Ollama vs LM Studio vs llama.cpp" — runtime comparison pillar
- Hardware guides: Raspberry Pi, Apple Silicon, low-VRAM Windows GPU
- Quantization explainer: GGUF vs MLX vs ONNX
