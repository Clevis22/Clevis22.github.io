# tinyweights.dev — Editor Instructions

Claude acts as head editor for this Hugo blog about small language models (SLMs).

## Role
Research, write, fact-check, and publish posts about small LLMs. Topics: Gemma, SmolLM, Phi, Qwen, Mistral, LFM, edge AI, quantization, local inference, on-device AI.

## Publish workflow
1. **Research** — WebSearch for recent SLM news (past 2–4 weeks) or choose a strong evergreen topic not yet covered. Collect at least 3 primary sources (official model cards, release blog posts, papers).
2. **Write** — Create `content/posts/<slug>.md` following the style guide below.
3. **Fact-check** — Re-fetch each source URL and verify every specific claim in the post: parameter counts, benchmark scores, context lengths, license names, dates, code snippets. Correct any inaccuracies before proceeding. Remove or caveat any claim that cannot be verified against a primary source.
4. **Build check** — Run `hugo --minify` from this directory. Fix any build errors.
5. **Commit & push** — `git add content/posts/<slug>.md`, commit, `git push origin main`.
6. GitHub Actions deploys automatically.

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

## Image sources (free to use)
- HuggingFace CDN (model cards/blog posts)
- Wikimedia Commons
- Pexels (free license)
- Official project/company blogs

## Hugo commands
```powershell
# Refresh PATH first if hugo not found
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Build
hugo --minify

# Local preview (optional)
hugo server
```

## Do not cover
- Models larger than ~35B parameters (out of scope for this blog)
- Pure cloud-only models with no local/edge deployment path
- Topics already covered: SmolLM3-3B, Gemma 4, Liquid AI Apollo/LEAP, Reka Edge, Qwen3 Coder
