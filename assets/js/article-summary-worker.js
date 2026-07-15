import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const BATCH_SIZE = 8;
const MAX_CANDIDATES = 56;

env.allowLocalModels = false;
env.useBrowserCache = true;
env.useWasmCache = true;
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

let extractor = null;
const runtime = "MiniLM · WASM q8";

const post = (id, type, data = {}) => self.postMessage({ id, type, ...data });

const progressCallback = (id) => (report) => {
  if (report.status !== "progress" && report.status !== "progress_total") return;
  post(id, "progress", {
    file: report.file || report.name || "model files",
    progress: Math.max(0, Math.min(1, (Number(report.progress) || 0) / 100)),
  });
};

const loadExtractor = async (id) => {
  if (extractor) return extractor;
  post(id, "stage", { message: "initializing memory-safe MiniLM", current: 0, total: 1 });
  extractor = await pipeline("feature-extraction", MODEL_ID, {
    device: "wasm",
    dtype: "q8",
    progress_callback: progressCallback(id),
  });
  return extractor;
};

const sentenceList = (text) => {
  const segments = "Segmenter" in Intl
    ? [...new Intl.Segmenter("en", { granularity: "sentence" }).segment(text)].map((item) => item.segment)
    : (text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || []);
  return segments
    .map((sentence) => sentence.replace(/\s+/g, " ").trim())
    .filter((sentence) => sentence.length >= 35 && sentence.length <= 360)
    .filter((sentence) => /[.!?]["')\]]?$/.test(sentence))
    .filter((sentence) => !sentence.includes(" | "))
    .filter((sentence) => !/\b(this post|this blog|posts? on this blog|see the .* (guide|comparison|post))\b/i.test(sentence));
};

const evenlySample = (items, limit) => {
  if (items.length <= limit) return items;
  return Array.from({ length: limit }, (_, index) =>
    items[Math.round(index * (items.length - 1) / (limit - 1))],
  );
};

const embedTexts = async (id, texts) => {
  const vectors = [];
  const total = Math.ceil(texts.length / BATCH_SIZE);
  for (let offset = 0; offset < texts.length; offset += BATCH_SIZE) {
    const current = Math.floor(offset / BATCH_SIZE);
    post(id, "stage", {
      message: `reading article section ${current + 1} of ${total}`,
      current,
      total,
    });
    const output = await extractor(texts.slice(offset, offset + BATCH_SIZE), {
      pooling: "mean",
      normalize: true,
      truncation: true,
    });
    vectors.push(...output.tolist());
  }
  return vectors;
};

const dot = (left, right) => {
  let value = 0;
  for (let index = 0; index < left.length; index += 1) value += left[index] * right[index];
  return value;
};

const rankSentences = (sentences, vectors, titleVector) => sentences
  .map((sentence, index) => {
    const neighbours = vectors
      .map((vector, other) => (other === index ? -1 : dot(vectors[index], vector)))
      .sort((a, b) => b - a)
      .slice(0, 5);
    const centrality = neighbours.reduce((sum, value) => sum + Math.max(0, value), 0)
      / Math.max(1, neighbours.length);
    const titleSimilarity = Math.max(0, dot(vectors[index], titleVector));
    const earlyPosition = 1 - (index / Math.max(1, sentences.length - 1));
    return {
      sentence,
      index,
      vector: vectors[index],
      score: (centrality * 0.62) + (titleSimilarity * 0.3) + (earlyPosition * 0.08),
    };
  })
  .sort((a, b) => b.score - a.score);

const pickTakeaways = (ranked) => {
  const selected = [];
  for (const candidate of ranked) {
    const isDuplicate = selected.some((item) => dot(candidate.vector, item.vector) > 0.82);
    if (!isDuplicate) selected.push(candidate);
    if (selected.length === 4) break;
  }
  if (selected.length < 4) {
    for (const candidate of ranked) {
      if (!selected.includes(candidate)) selected.push(candidate);
      if (selected.length === 4) break;
    }
  }
  return selected.sort((a, b) => a.index - b.index).map((item) => item.sentence);
};

const pickBottomLine = (sentences) => {
  const ending = sentences.slice(Math.max(0, sentences.length - 12));
  const ranked = ending.map((sentence, index) => {
    const imperative = /^(run|choose|use|pick|start|avoid|prefer|try)\b/i.test(sentence) ? 2 : 0;
    const decision = /\b(better|best|choice|tiebreaker|recommend|for most|should)\b/i.test(sentence) ? 1 : 0;
    return { sentence, score: imperative + decision + (index / Math.max(1, ending.length)) };
  }).sort((a, b) => b.score - a.score);
  return ranked[0]?.sentence || sentences.at(-1) || "";
};

const summarizeArticle = async (id, title, article) => {
  await loadExtractor(id);
  post(id, "model-ready", { runtime });

  const allSentences = sentenceList(article);
  if (!allSentences.length) throw new Error("No article text was available to summarize.");

  const overview = allSentences.slice(0, 2).join(" ");
  const candidates = evenlySample(allSentences.slice(2), MAX_CANDIDATES);
  const vectors = await embedTexts(id, [title, ...candidates]);
  const titleVector = vectors.shift();
  post(id, "stage", { message: "selecting key points", current: 1, total: 1 });
  const ranked = rankSentences(candidates, vectors, titleVector);

  return {
    summary: overview,
    takeaways: pickTakeaways(ranked),
    bottom_line: pickBottomLine(allSentences),
    runtime,
  };
};

self.addEventListener("message", async (event) => {
  const { type, id, title, article } = event.data || {};
  if (type !== "summarize" || !id) return;
  try {
    const summary = await summarizeArticle(id, title || "Article", article || "");
    post(id, "result", { summary });
  } catch (error) {
    post(id, "error", { message: error?.message || String(error) || "Local summarization failed." });
  }
});
