# tinyweights.dev — SEO & Traffic Growth Plan

**Last updated:** 2026-05-13  
**Current state:** ~9 published posts, Hugo + PaperMod, custom domain, GitHub Pages  
**Target audience:** Developers, tinkerers, indie hackers who run LLMs locally

---

## The Big Picture

tinyweights.dev has one genuine advantage: it is the only blog with a clear, narrow focus on small language models. That specificity is the entire strategy. Do not drift into general AI coverage. Every post should be answerable to the question: *"Is this something someone who runs models locally would search for and trust?"*

The goal for the first 12 months is **topical authority in the SLM niche** — owning the topic cluster so thoroughly that when someone searches "best 3B model for local inference", Google and AI search engines default to this site.

---

## Part 1: Content Strategy

### 1.1 — Topical Clusters (The Core SEO Structure)

Rather than a flat list of posts, build interconnected clusters. Each cluster has one **pillar post** (comprehensive, 2000+ words) and several **satellite posts** (specific, 1000–1500 words) that link back to it.

**Cluster 1: Hardware-specific deployment**
- Pillar: "The Complete Guide to Running Small LLMs on Apple Silicon (M-series)"
- Satellites: "Running Ollama on M4 MacBook Pro: Setup and Benchmarks", "Best Quantization for MLX on Mac", "How Much RAM Do You Actually Need for Local LLMs"

**Cluster 2: Model size comparisons**
- Pillar: "The Best Small Language Models in 2026: A Practical Comparison"
- Satellites: Individual model reviews (which we're already doing), head-to-head posts ("Phi-4-mini vs SmolLM3: Which 3B Model Wins for Coding?"), benchmark breakdowns

**Cluster 3: Quantization and formats**
- Pillar: "GGUF vs ONNX vs MLX: Which Format Should You Use for Local Inference?"
- Satellites: "Q4 vs Q5 vs Q8 Quants: What You're Actually Giving Up", "How to Quantize a Model with llama.cpp", "GGUF File Sizes by Model: What Fits on Your Hardware"

**Cluster 4: Use cases**
- Pillar: "What Can You Actually Do With a Local Small LLM? A Practical Guide"
- Satellites: "Local LLM for Private Document Q&A", "Using a Small Model as a Coding Assistant", "On-Device Translation with Small Multilingual Models"

**Cluster 5: Tools and runtimes**
- Pillar: "Ollama vs LM Studio vs llama.cpp: Which Local AI Runtime Is Right for You?"
- Satellites: Individual tool walkthroughs, vLLM setup guides, LlamaEdge deployment

### 1.2 — Keyword Targeting

**Priority keyword patterns (high intent, low competition):**

| Pattern | Example | Why It Works |
|---|---|---|
| `run [model] locally` | "run Qwen3.5 locally" | High purchase intent, specific |
| `[model] ollama guide` | "SmolLM3 Ollama setup" | Tutorial intent, easy to rank |
| `[model] benchmark` | "Phi-4-mini benchmark results" | Research intent, niche |
| `best [size]B model for [task]` | "best 3B model for coding" | Comparison intent, converts |
| `[model] vs [model]` | "SmolLM3 vs Phi-4-mini" | Very low competition, high click-through |
| `local LLM for [use case]` | "local LLM for document summarization" | Evergreen, growing search volume |
| `[model] GGUF` | "Qwen3.5 GGUF download" | High intent, underserved |

**Rule of thumb:** Always include the exact model name (e.g., "Qwen3.5-0.8B" not just "Qwen"). People searching for model-specific information want the exact name in the post title, URL slug, and first paragraph.

### 1.3 — Content Cadence

- **Target: 2 posts per week** (currently 1 autonomous post possible via scheduler)
- Mix: 60% news/new releases, 40% evergreen guides
- Every evergreen guide should link to at least 3 existing posts
- Every new model post should link to the relevant cluster pillar once it exists

### 1.4 — Post Types to Prioritize

**"How to run X locally"** — The single highest-traffic post type for this niche. Always include exact commands, expected RAM usage, and a note on quantization. These rank because people search for this immediately after hearing about a model.

**"X vs Y" comparisons** — Almost no one else writes these rigorously for small models. Pick two models at the same parameter scale and benchmark them on the same tasks. These get shared heavily in r/LocalLLaMA.

**Model release posts** — What we're already doing. Publish within 24–48 hours of a major release while competition is lowest. Use the model name exactly as it appears on HuggingFace in the title.

**Hardware guides** — "What can a Raspberry Pi 5 run in 2026?" These attract people who don't know what model to pick, which is a large audience. They rank for long-tail hardware + AI queries.

---

## Part 2: Distribution (The Part Most Blogs Skip)

SEO takes months. Distribution gives traffic on day one. Do both.

### 2.1 — Reddit

**r/LocalLLaMA (717K members)** — This is the single most important distribution channel. The audience is exactly the people who read this blog: developers running models locally. Rules:

- Never post a bare link. Write a 3–5 sentence summary of what the post covers, then link. Treat the Reddit post as a standalone value piece.
- Best performing post types here: hands-on benchmarks, "I tested X, here's what happened", surprising findings. Avoid posts that are purely summaries of press releases.
- Post when a model releases — the community is actively looking for coverage and hands-on notes.
- Engage in comments. Answer technical questions. Build recognition as a contributor, not just a promoter.

**Other relevant subreddits:**
- r/MachineLearning — more academic, good for architecture-focused posts
- r/learnmachinelearning — good for beginner-friendly tutorials and explainers
- r/selfhosted — good for privacy/local inference angle posts
- r/homelab — hardware guides and on-device AI posts

**Posting rhythm:** Post to Reddit within 2 hours of a new post going live. Don't post every article — choose the ones with a genuine finding or a hands-on angle.

### 2.2 — Hacker News

**Show HN / Tell HN** — HN rewards depth and originality over timeliness. The best posts to submit here are the architecture deep-dives (like the Gated DeltaNet post), evergreen guides, and comparison posts with real benchmark data.

- Submit in the morning US time (9–11am ET) for maximum visibility
- HN front page can drive 10,000–50,000 visits in a single day
- The audience skews infrastructure/backend — technical depth is rewarded

### 2.3 — X / Twitter (@TinyWeightsAI)

The account already exists. Use it as a signal amplifier, not a content platform.

**What to post:**
- A thread (3–5 tweets) summarizing each new post, with the key finding in tweet 1 (no link until tweet 2+)
- Benchmark screenshots and comparison tables as images
- Quick takes on model releases before the full post goes live ("Just downloaded Qwen3.5-0.8B. First thoughts: [finding]. Full post coming.")

**Who to engage with:**
- HuggingFace, Ollama, LM Studio, Unsloth — they often retweet coverage of models they support
- Model authors (tag them when covering their model — many share coverage)
- Other SLM-focused accounts

### 2.4 — HuggingFace Discussions

HuggingFace model cards have a Community tab. When a post covers a specific model, leave a comment on that model's HuggingFace discussion linking to the post as a "deployment guide" or "hands-on review." This is an underused traffic source — people browsing model cards are exactly the right audience.

### 2.5 — Newsletter (Phase 2 — once at 50+ posts)

A weekly "SLM digest" email — 5 bullet points about what moved in small models that week, with links to relevant posts. Substack or Buttondown. Do not start this until there is enough archived content to justify new subscriber trust.

---

## Part 3: Technical SEO

### 3.1 — On-Page Checklist (apply to every post)

- [ ] Target keyword in the `title`, `description`, and `slug`
- [ ] Target keyword in the first paragraph (within the first 100 words)
- [ ] Model name used *exactly* as it appears on HuggingFace in the title (e.g., "Qwen3.5-0.8B" not "Qwen 3.5")
- [ ] At least one H2 containing the keyword or a close variant
- [ ] 2+ internal links to existing posts (preferably to the relevant cluster pillar)
- [ ] `## Sources` section with all primary URLs
- [ ] Alt text on all images

### 3.2 — Schema Markup

Add Article schema to improve AI search engine visibility. In `layouts/partials/extend_head.html`, add structured data for each post. Priority schemas:

- `Article` — on every post page (author, date, headline, image)
- `FAQPage` — on pillar/guide posts with Q&A sections
- `HowTo` — on deployment guide posts with step-by-step instructions

This is what gets tinyweights.dev cited in ChatGPT and Gemini answers rather than just ranked in Google.

### 3.3 — Internal Linking Rules

- Every new post should link to at least 2 older posts
- Every model post should link to the relevant "how to run locally" guide when it exists
- Pillar posts must be updated to link to every new satellite post in their cluster
- Use descriptive anchor text: not "click here" but "our Qwen3 Coder breakdown"

### 3.4 — Hugo Config Fix (Minor)

The `hugo.toml` uses `languageCode` which is deprecated since Hugo v0.158.0. Replace with `locale` to eliminate build warnings. Low priority but clean it up when touching the config.

---

## Part 4: LLM SEO (Being Cited by AI Assistants)

In 2026, a significant portion of research happens through ChatGPT, Claude, and Perplexity rather than Google. These systems cite sources they've indexed. To get cited:

- **Write in answer format.** Structure posts to directly answer questions: "What is the best 3B model for local inference?" should appear as a literal heading with a direct answer below it.
- **Include comparison tables.** AI systems love extracting structured data. Every model post should have at least one comparison table.
- **Cite primary sources correctly.** Posts that cite official HuggingFace model cards, papers, and official blogs are treated as more authoritative and more likely to be re-cited by AI search.
- **Keep facts verifiable and specific.** Vague claims ("this model is fast") don't get cited. Specific claims ("achieves 7.6 decode tokens/s on Raspberry Pi 5 CPU") do.

---

## Part 5: Metrics to Track

Track these monthly:

| Metric | Tool | Target (Month 6) | Target (Month 12) |
|---|---|---|---|
| Organic search impressions | Google Search Console | 10K/mo | 50K/mo |
| Organic clicks | Google Search Console | 500/mo | 5K/mo |
| r/LocalLLaMA posts | Manual | 2/mo | 4/mo |
| Indexed pages | Google Search Console | 30+ | 80+ |
| Backlinks (referring domains) | Ahrefs free / Moz | 10+ | 50+ |

**Set up Google Search Console immediately** — connect the property for tinyweights.dev to get keyword impression data. This is free and essential.

---

## Part 6: Immediate Action List

Things to do in the next 30 days that will compound over time:

1. **Submit tinyweights.dev to Google Search Console** and verify via DNS or HTML tag
2. **Write the first pillar post** — "Best Small Language Models in 2026: A Practical Comparison" (this becomes the hub for all model review posts to link back to)
3. **Add Article schema** to `layouts/partials/extend_head.html`
4. **Post the Qwen3.5 post to r/LocalLLaMA** with a hands-on framing
5. **Start "run locally" posts** — one for each model already covered: SmolLM3 deployment deep-dive, Gemma 4 E2B on Raspberry Pi
6. **Fix internal linking** — go back through existing posts and add links where they're missing

---

## What Not To Do

- **Don't chase every model release** if you can't add hands-on value. A post that's just a rewrite of the HuggingFace blog post will not rank and will not get shared.
- **Don't post to Reddit before the site has real content** — the community will see through it. Build genuine posts first.
- **Don't go broad.** One post about GPT-5 or a frontier model will confuse the topical signal and dilute the niche authority we're building.
- **Don't neglect internal links.** The site currently has ~9 posts none of which strongly link to each other. Fix this before publishing more.

---

## The 3-Sentence Summary

Build topical authority by covering every angle of the small LLM space — model reviews, hardware guides, quantization explainers, and comparison posts — with real hands-on specificity. Distribute every strong post to r/LocalLLaMA and occasionally HN; tag model authors on X. The niche is small enough that 50 well-targeted posts with genuine technical depth can make this the default reference site for anyone running small models locally.
