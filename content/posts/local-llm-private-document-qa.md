---
title: "Local LLM for Private Document Q&A: Build an Offline RAG Pipeline"
date: 2026-07-07
draft: false
tags: ["rag", "use-cases", "local-inference", "small-models"]
categories: ["guides"]
description: "How to build a fully private, offline document Q&A system with a local small LLM and an embedding model: exact Ollama commands, a runnable Python RAG script, and hardware requirements."
slug: "local-llm-private-document-qa"
---

A local LLM for private document Q&A is the clearest case where running a model on your own hardware wins outright against any cloud API. Your contracts, medical records, source code, and internal notes never touch a network. This guide builds a fully offline retrieval-augmented generation (RAG) pipeline: a small embedding model turns your documents into vectors, a local vector store finds the passages that answer a question, and a small LLM writes the answer from those passages. The whole thing runs on a laptop with 8 to 16 GB of RAM, with no API key and no per-token bill.

The pieces are all open source and all run through [Ollama](/posts/ollama-vs-lm-studio-vs-llama-cpp/), so if you have already used a local model this will feel familiar. If you have not, the [what-can-you-do overview](/posts/what-can-you-do-with-local-small-llm/) covers the basics of getting a model running first.

{{< figure src="https://images.pexels.com/photos/17846111/pexels-photo-17846111.jpeg" alt="Close-up of a large pile of old paper document files stacked in folders" caption="Document Q&A is the use case where 'nothing leaves the machine' is a hard requirement, not a nice-to-have. (Photo: Adil Khan Marwat, Pexels)" >}}

## Why run document Q&A locally?

Privacy here is categorical, not incremental. When the model and the index both run on your machine, no provider's data-retention policy applies to you because your data never reached a provider. For a confidential settlement agreement, a folder of patient files, or a private codebase, that is the difference between a tool you are allowed to use and one you are not.

There is a performance angle too. A cloud RAG service charges per token on every query and every re-index. Local embeddings and local generation cost nothing after the one-time model download, so you can re-index a 500-page manual as often as you like and run hundreds of queries a day without watching a meter.

## How the RAG pipeline works

Retrieval-augmented generation splits the problem into two jobs. First, indexing: you break each document into chunks, run every chunk through an embedding model to get a vector, and store those vectors. Second, querying: you embed the question with the same model, find the handful of chunks whose vectors sit closest to the question's vector, and paste those chunks into the LLM's prompt so it answers from real text instead of memory.

The reason you retrieve instead of pasting the entire document is partly context length and partly accuracy. Even models with enormous context windows get less reliable as the prompt grows, and attention over 100k tokens is slow on a laptop. Retrieval keeps the prompt short and focused, which makes a 3B or 4B model punch well above its weight.

There is one exception worth knowing. If you only have a single medium-length document and you query it rarely, you can skip retrieval entirely. Qwen3-4B-Instruct-2507 ships with a native 262,144-token context window, which is roughly 200,000 words, so a whole book fits in one prompt. For a folder of documents, or for the same file queried many times, RAG is faster and more accurate than re-reading everything on every question.

## Choosing an embedding model

The embedding model decides what "relevant" means, so a weak one quietly caps the quality of the whole system no matter how good the generator is. For local RAG the practical choices are small, fast, and open. Here is how the common options compare.

| Model | Params | Dimensions | Max context | MTEB (English) | License |
|---|---|---|---|---|---|
| EmbeddingGemma | 308M | 768 (→128 via MRL) | 2,048 tokens | 69.67 | Gemma |
| nomic-embed-text-v1.5 | 137M | 768 (→64 via MRL) | 8,192 tokens | 62.28 | Apache 2.0 |
| mxbai-embed-large | 334M | 1,024 | 512 tokens | n/a | Apache 2.0 |
| all-minilm | 23M | 384 | 256 tokens | n/a | Apache 2.0 |

EmbeddingGemma is the strongest of the group and was built for exactly this job. Google describes it as the highest-ranking open multilingual text embedding model under 500M parameters on the Massive Text Embedding Benchmark (MTEB), it covers 100+ languages, and quantized it runs in under 200 MB of RAM with sub-15ms inference on an EdgeTPU for a 256-token input. Its one constraint is a 2,048-token context, so your chunks have to stay under that.

nomic-embed-text is the pragmatic default when your documents have long, dense sections. Its 8,192-token window lets you use larger chunks (fewer, more complete passages per document), it is a tiny 137M parameters, and it is Apache 2.0. Both models support Matryoshka Representation Learning, which means you can truncate the 768-dimensional vector to 512, 256, or 128 dims to shrink your index at a small accuracy cost.

{{< figure src="https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/EmbeddingGemma_Chart01_RD3-V01.original.png" alt="Bar chart comparing EmbeddingGemma MTEB scores against other embedding models under 500M parameters" caption="EmbeddingGemma's MTEB score against other sub-500M embedding models. (Chart: Google, from the EmbeddingGemma announcement)" >}}

**Which embedding model should I use for local RAG?**

Start with nomic-embed-text if your documents are mostly English and have long sections, because its 8k context lets you chunk less aggressively. Switch to EmbeddingGemma if you need multilingual retrieval or the best raw quality and your chunks are short enough to fit its 2k window. Reach for all-minilm only when you are indexing millions of chunks on very weak hardware and speed matters more than accuracy.

## Setting up Ollama

Pull one embedding model and one generation model. The generation model does the actual answering, so pick a small instruct model that follows grounding instructions well; Qwen3-4B is a good default, and IBM's [Granite 4.1](/posts/run-ibm-granite-4-1-locally/) is tuned specifically for retrieval workloads if you want an alternative.

```bash
ollama pull nomic-embed-text     # 137M embedding model
ollama pull qwen3:4b             # 4B generation model
```

Confirm the embedding endpoint works before writing any code:

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "nomic-embed-text",
  "input": "Llamas are members of the camelid family"
}'
```

That returns a JSON object with an `embeddings` array. If you get a vector back, both halves of the pipeline are ready.

## The RAG script

This is a complete, dependency-light pipeline using Ollama for both embedding and generation and [Chroma](https://docs.trychroma.com/) as the local vector store. Chroma is Apache 2.0, runs in-process, and persists to a folder on disk, so nothing about it reaches the network.

```python
import ollama
import chromadb

EMBED_MODEL = "nomic-embed-text"
GEN_MODEL   = "qwen3:4b"

db = chromadb.PersistentClient(path="./doc_db")
col = db.get_or_create_collection("docs")

def chunk(text, words=350, overlap=50):
    w = text.split()
    step = words - overlap
    return [" ".join(w[i:i + words]) for i in range(0, len(w), step)]

def index(doc_text):
    chunks = chunk(doc_text)
    vectors = ollama.embed(model=EMBED_MODEL, input=chunks)["embeddings"]
    col.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=chunks,
    )

def ask(question, k=4):
    q_vec = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    hits = col.query(query_embeddings=[q_vec], n_results=k)
    context = "\n\n---\n\n".join(hits["documents"][0])
    reply = ollama.chat(model=GEN_MODEL, messages=[
        {"role": "system", "content":
            "Answer only using the context below. "
            "If the answer is not in the context, say you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ])
    return reply["message"]["content"]

index(open("contract.txt", encoding="utf-8").read())
print(ask("What are the payment terms and the notice period?"))
```

Chunk size is the setting people get wrong most often. Keep each chunk under the embedding model's context window (350 words is roughly 450 to 500 tokens, comfortably inside both nomic's 8k and EmbeddingGemma's 2k), and keep a small overlap so a sentence split across a chunk boundary still appears whole in one of them. Retrieve four to six chunks per query to start, then tune: too few and you miss the answer, too many and you bury the model in noise.

**Do I need a vector database for document Q&A?**

For a few dozen chunks you do not; a plain Python list of vectors and a cosine-similarity loop works fine and has zero dependencies. A vector store like Chroma, Qdrant, or LanceDB earns its place once you are indexing thousands of chunks, because it keeps the index on disk (so you do not re-embed on every launch) and searches it with an approximate-nearest-neighbour algorithm instead of scanning everything.

## Keeping a small model honest

Small models hallucinate more readily than frontier models, so the grounding instructions do real work. Two rules make the difference. Tell the model, in the system prompt, to answer only from the supplied context and to admit when the answer is not there. And return the retrieved passages to the user alongside the answer so a person can check the model against its own sources. For an instruct build like Qwen3-4B-Instruct-2507, answers come back concise and grounded; if you use a reasoning variant, expect longer output with visible thinking, which you may want to strip before display.

## Hardware requirements

**How much RAM do I need to run local document Q&A?**

Very little. The embedding model is the cheap part: nomic-embed-text is 137M parameters and EmbeddingGemma runs in under 200 MB quantized. The generation model dominates the footprint, and a 4B model at a q4 quant needs roughly 3 GB. That means an 8 GB laptop handles the whole pipeline, and 16 GB gives you room to jump to a 7B or 8B generator for harder questions. The full sizing breakdown is in [How Much RAM Do You Actually Need for Local LLMs](/posts/how-much-ram-for-local-llms/), and if you are on a Mac the [Apple Silicon guide](/posts/run-small-llms-apple-silicon/) covers unified-memory specifics. Quantization trade-offs are covered in [our Q4 vs Q5 vs Q8 explainer](/posts/gguf-quantization-levels-q4-q5-q8/).

## Is a local RAG pipeline actually private?

Yes, with one honest caveat. Ollama serves models from localhost, Chroma writes its index to a local folder, and the embedding and generation steps are plain function calls on your own CPU or GPU. Pull the two models once, and after that you can disconnect from the network entirely and the pipeline keeps working. The caveat is the initial `ollama pull`, which downloads weights from a registry. Once those files are on disk, no part of indexing or querying phones home, and you can verify that by running the whole flow with your network adapter switched off.

## When to size up

A 4B model with good retrieval answers most factual "what does this document say" questions well. It struggles on two things: multi-hop questions that require stitching facts from several documents, and large corpora where the right chunk is easy to miss. For the first, a 7B or 8B generator helps. For the second, add a reranker (a small cross-encoder such as a bge-reranker model) that re-scores the top 20 retrieved chunks and keeps the best 4, which sharply improves which passages reach the model. Both upgrades slot into the same script without changing its shape: swap the generation model, or add a rerank step between `query` and `chat`.

## Sources

- [EmbeddingGemma model card (Hugging Face)](https://huggingface.co/google/embeddinggemma-300m)
- [Introducing EmbeddingGemma (Google Developers Blog)](https://developers.googleblog.com/en/introducing-embeddinggemma/)
- [nomic-embed-text-v1.5 model card (Hugging Face)](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [mxbai-embed-large-v1 model card (Hugging Face)](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- [all-MiniLM-L6-v2 model card (Hugging Face)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Embedding models (Ollama Blog)](https://ollama.com/blog/embedding-models)
- [Qwen3-4B-Instruct-2507 model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Chroma documentation](https://docs.trychroma.com/)
