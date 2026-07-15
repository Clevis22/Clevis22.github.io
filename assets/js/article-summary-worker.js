import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "onnx-community/text_summarization-ONNX";
const CHUNK_CHARS = 1400;

env.allowLocalModels = false;
env.useBrowserCache = true;

let summarizer = null;
let runtime = "";

const post = (id, type, data = {}) => self.postMessage({ id, type, ...data });

const progressCallback = (id) => (report) => {
  if (report.status !== "progress") return;
  post(id, "progress", {
    file: report.file || report.name || "model file",
    progress: Math.max(0, Math.min(1, (Number(report.progress) || 0) / 100)),
  });
};

const loadSummarizer = async (id) => {
  if (summarizer) return summarizer;

  const attempts = [{ device: "wasm", dtype: "q8", label: "WASM q8" }];

  let lastError;
  for (const attempt of attempts) {
    try {
      post(id, "stage", { message: `initializing ${attempt.label}`, current: 0, total: 1 });
      summarizer = await pipeline("summarization", MODEL_ID, {
        device: attempt.device,
        dtype: attempt.dtype,
        progress_callback: progressCallback(id),
      });
      runtime = attempt.label;
      return summarizer;
    } catch (error) {
      lastError = error;
      summarizer = null;
      post(id, "stage", {
        message: `${attempt.label} failed: ${error?.message || String(error)}`,
        current: 0,
        total: 1,
      });
    }
  }
  throw lastError || new Error("The T5 summarizer could not be loaded.");
};

const splitLongText = (text, limit = CHUNK_CHARS) => {
  const chunks = [];
  let current = "";

  const pushCurrent = () => {
    if (current.trim()) chunks.push(current.trim());
    current = "";
  };

  const append = (part) => {
    const clean = part.replace(/\s+/g, " ").trim();
    if (!clean) return;
    if (clean.length > limit) {
      pushCurrent();
      const words = clean.split(" ");
      let piece = "";
      for (const word of words) {
        if (piece && `${piece} ${word}`.length > limit) {
          chunks.push(piece);
          piece = word;
        } else {
          piece = piece ? `${piece} ${word}` : word;
        }
      }
      if (piece) chunks.push(piece);
      return;
    }
    if (current && `${current} ${clean}`.length > limit) pushCurrent();
    current = current ? `${current} ${clean}` : clean;
  };

  for (const paragraph of text.split(/\n+/)) append(paragraph);
  pushCurrent();
  return chunks;
};

const summarizeText = async (text, maxNewTokens = 72) => {
  const result = await summarizer(text, {
    truncation: true,
    max_new_tokens: maxNewTokens,
    min_new_tokens: 16,
    do_sample: false,
    num_beams: 1,
    no_repeat_ngram_size: 3,
    length_penalty: 1.0,
  });
  const summary = result?.[0]?.summary_text?.replace(/\s+/g, " ").trim();
  if (!summary) throw new Error("T5 returned an empty summary.");
  return summary;
};

const sentenceList = (text) => {
  const segments = "Segmenter" in Intl
    ? [...new Intl.Segmenter("en", { granularity: "sentence" }).segment(text)].map((item) => item.segment)
    : (text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || []);
  return segments
    .map((sentence) => sentence.replace(/\s+/g, " ").trim())
    .filter((sentence) => sentence.length >= 35);
};

const wordSet = (text) => new Set(
  text.toLowerCase().match(/[a-z0-9][a-z0-9.+-]*/g)?.filter((word) => word.length > 2) || [],
);

const rankSourceSentences = (text, generatedSummary) => {
  const summaryWords = wordSet(generatedSummary);
  return sentenceList(text)
    .filter((sentence) => sentence.length <= 360)
    .map((sentence, index) => {
      const words = wordSet(sentence);
      let overlap = 0;
      for (const word of words) if (summaryWords.has(word)) overlap += 1;
      const score = overlap / Math.max(1, Math.sqrt(words.size * summaryWords.size));
      return { sentence, score, index };
    })
    .sort((a, b) => b.score - a.score);
};

const bestSourceSentence = (chunk, generatedSummary) => {
  const ranked = rankSourceSentences(chunk, generatedSummary);
  return ranked[0]?.sentence || chunk.slice(0, 280).trim();
};

const pickOverview = (article) => sentenceList(article).slice(0, 2).join(" ");

const pickTakeaways = (chunks, sectionSummaries) => {
  const candidates = chunks.map((chunk, index) => bestSourceSentence(chunk, sectionSummaries[index]));
  if (candidates.length <= 4) return candidates;
  return [0, 1, 2, 3].map((index) => candidates[Math.round(index * (candidates.length - 1) / 3)]);
};

const pickBottomLine = (chunk, generatedDigest) => {
  const ranked = rankSourceSentences(chunk, generatedDigest)
    .map((item) => {
      const imperative = /^(run|choose|use|pick|start|avoid|prefer|try)\b/i.test(item.sentence) ? 1 : 0;
      const decision = /\b(better|best|choice|tiebreaker|recommend|for most|should)\b/i.test(item.sentence) ? 0.35 : 0;
      return { ...item, decisionScore: item.score + imperative + decision };
    })
    .sort((a, b) => b.decisionScore - a.decisionScore);
  return ranked[0]?.sentence || bestSourceSentence(chunk, generatedDigest);
};

const summarizeArticle = async (id, title, article) => {
  await loadSummarizer(id);
  post(id, "model-ready", { runtime });

  const chunks = splitLongText(article);
  if (!chunks.length) throw new Error("No article text was available to summarize.");

  const sectionSummaries = [];
  for (let index = 0; index < chunks.length; index += 1) {
    post(id, "stage", {
      message: `summarizing section ${index + 1} of ${chunks.length}`,
      current: index,
      total: chunks.length,
    });
    sectionSummaries.push(await summarizeText(`Title: ${title}\n${chunks[index]}`));
  }

  const generatedDigest = `${title} ${sectionSummaries.join(" ")}`;
  const finalSummary = pickOverview(article);
  const takeaways = pickTakeaways(chunks, sectionSummaries);
  const bottomLine = pickBottomLine(chunks.at(-1), generatedDigest);
  return {
    summary: finalSummary,
    takeaways,
    bottom_line: bottomLine,
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
    post(id, "error", {
      message: error?.message || String(error) || "Local summarization failed.",
    });
  }
});
