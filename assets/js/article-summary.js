const CACHE_PREFIX = "tw-article-summary-v2";
const MAX_ARTICLE_CHARS = 20000;
const INFERENCE_TIMEOUT_MS = 90000;

const root = document.querySelector("[data-article-summary]");

if (root) {
  const ui = {
    run: root.querySelector("[data-summary-run]"),
    note: root.querySelector("[data-summary-note]"),
    output: root.querySelector("[data-summary-output]"),
    status: root.querySelector("[data-summary-status]"),
    statusText: root.querySelector("[data-summary-status-text]"),
    statusMark: root.querySelector(".tw-summary-status-mark"),
    progress: root.querySelector("[data-summary-progress]"),
    progressBar: root.querySelector("[data-summary-progress-bar]"),
    result: root.querySelector("[data-summary-result]"),
    lead: root.querySelector("[data-summary-lead]"),
    pointsWrap: root.querySelector("[data-summary-points-wrap]"),
    points: root.querySelector("[data-summary-points]"),
    bottomWrap: root.querySelector("[data-summary-bottom-wrap]"),
    bottom: root.querySelector("[data-summary-bottom]"),
    model: root.querySelector("[data-summary-model]"),
    regenerate: root.querySelector("[data-summary-regenerate]"),
    hide: root.querySelector("[data-summary-hide]"),
  };

  const cacheKey = `${CACHE_PREFIX}:${root.dataset.slug}:${root.dataset.version}`;
  let summaryWorker = null;
  let running = false;
  let runId = 0;
  let startedAt = 0;
  let statusBase = "";
  let elapsedTimer = null;
  let inferenceTimer = null;

  const elapsedSeconds = () => Math.max(0, Math.round((performance.now() - startedAt) / 1000));

  const paintStatusText = () => {
    ui.statusText.textContent = running ? `${statusBase} · ${elapsedSeconds()}s` : statusBase;
  };

  const setStatus = (message, kind = "loading", mark = "[....]") => {
    statusBase = message;
    ui.output.hidden = false;
    ui.status.hidden = false;
    ui.status.classList.toggle("is-ready", kind === "ready");
    ui.status.classList.toggle("is-error", kind === "error");
    ui.statusMark.textContent = mark;
    paintStatusText();
  };

  const setProgress = (value) => {
    const percent = Math.max(0, Math.min(100, Math.round(value * 100)));
    ui.progress.hidden = percent <= 0 || percent >= 100;
    ui.progressBar.style.width = `${percent}%`;
  };

  const setRunning = (value) => {
    running = value;
    ui.run.textContent = value ? "stop" : "summarize";
    ui.regenerate.disabled = value;
    clearInterval(elapsedTimer);
    elapsedTimer = value ? setInterval(paintStatusText, 1000) : null;
  };

  const clearInferenceTimer = () => {
    clearTimeout(inferenceTimer);
    inferenceTimer = null;
  };

  const normalizeSummary = (value) => {
    const summary = typeof value?.summary === "string" ? value.summary.trim() : "";
    const takeaways = Array.isArray(value?.takeaways)
      ? value.takeaways
        .filter((item) => typeof item === "string" && item.trim())
        .slice(0, 5)
        .map((item) => item.trim())
      : [];
    const bottomLine = typeof value?.bottom_line === "string" ? value.bottom_line.trim() : "";
    const runtime = typeof value?.runtime === "string" ? value.runtime.trim() : "";
    if (!summary) throw new Error("The model returned an empty summary.");
    return { summary, takeaways, bottom_line: bottomLine, runtime };
  };

  const renderSummary = (summary) => {
    ui.lead.textContent = summary.summary;
    ui.points.replaceChildren();
    for (const point of summary.takeaways) {
      const item = document.createElement("li");
      item.textContent = point;
      ui.points.append(item);
    }
    ui.pointsWrap.hidden = summary.takeaways.length === 0;
    ui.bottom.textContent = summary.bottom_line;
    ui.bottomWrap.hidden = !summary.bottom_line;
    ui.model.textContent = summary.runtime
      ? `generated locally · T5-small · ${summary.runtime}`
      : "generated locally · T5-small";
    ui.result.hidden = false;
    ui.status.hidden = true;
    ui.progress.hidden = true;
    ui.output.hidden = false;
  };

  const readCache = () => {
    try {
      const cached = JSON.parse(localStorage.getItem(cacheKey));
      return cached?.summary ? normalizeSummary(cached.summary) : null;
    } catch {
      return null;
    }
  };

  const writeCache = (summary) => {
    try {
      localStorage.setItem(cacheKey, JSON.stringify({ summary, savedAt: Date.now() }));
    } catch {
      // A disabled or full localStorage should not prevent summarization.
    }
  };

  const extractArticle = () => {
    const content = document.querySelector(".post-content");
    if (!content) throw new Error("Article text could not be found.");

    const clone = content.cloneNode(true);
    clone.querySelectorAll("figure, pre, script, style, noscript, .highlight, .copy-code").forEach((node) => node.remove());

    const sources = [...clone.querySelectorAll("h2")]
      .find((heading) => heading.textContent.trim().toLowerCase() === "sources");
    if (sources) {
      let node = sources;
      while (node) {
        const next = node.nextSibling;
        node.remove();
        node = next;
      }
    }

    const blocks = [...clone.querySelectorAll("h2, h3, p, li, tr")]
      .filter((node) => !(node.matches("p") && node.closest("li")))
      .map((node) => {
        if (node.matches("tr")) {
          return [...node.querySelectorAll("th, td")]
            .map((cell) => cell.textContent.trim())
            .filter(Boolean)
            .join(" | ");
        }
        return node.textContent.replace(/\s+/g, " ").trim();
      })
      .filter(Boolean)
      .join("\n");

    if (!blocks) throw new Error("Article text could not be read.");
    return blocks.length > MAX_ARTICLE_CHARS ? blocks.slice(0, MAX_ARTICLE_CHARS) : blocks;
  };

  const destroyWorker = () => {
    summaryWorker?.terminate();
    summaryWorker = null;
  };

  const finishRun = () => {
    clearInferenceTimer();
    setRunning(false);
  };

  const failRun = (message, stopped = false) => {
    finishRun();
    setStatus(message, stopped ? "ready" : "error", stopped ? "[STOP]" : "[FAIL]");
    ui.result.hidden = true;
    ui.progress.hidden = true;
  };

  const armInferenceTimeout = () => {
    clearInferenceTimer();
    inferenceTimer = setTimeout(() => {
      destroyWorker();
      failRun("local inference timed out after 90 seconds");
    }, INFERENCE_TIMEOUT_MS);
  };

  const handleWorkerMessage = (event) => {
    const message = event.data || {};
    if (message.id !== runId) return;

    if (message.type === "progress") {
      const filename = (message.file || "model file").split("/").pop();
      setStatus(`loading ${filename}`, "loading", "[LOAD]");
      setProgress(Number(message.progress) || 0);
      return;
    }

    if (message.type === "model-ready") {
      setStatus(`model ready on ${message.runtime}; starting summary`, "loading", "[AI  ]");
      setProgress(0);
      armInferenceTimeout();
      return;
    }

    if (message.type === "stage") {
      setStatus(message.message || "summarizing article", "loading", "[AI  ]");
      setProgress(message.total ? message.current / message.total : 0);
      return;
    }

    if (message.type === "result") {
      try {
        const summary = normalizeSummary(message.summary);
        writeCache(summary);
        renderSummary(summary);
        finishRun();
      } catch (error) {
        failRun(error.message || "The summary could not be displayed.");
      }
      return;
    }

    if (message.type === "error") {
      destroyWorker();
      failRun(message.message || "Local summarization failed.");
    }
  };

  const getWorker = () => {
    if (summaryWorker) return summaryWorker;
    summaryWorker = new Worker(root.dataset.workerUrl, { type: "module" });
    summaryWorker.addEventListener("message", handleWorkerMessage);
    summaryWorker.addEventListener("error", (event) => {
      destroyWorker();
      failRun(event.message || "The local summarizer worker failed to start.");
    });
    return summaryWorker;
  };

  const generate = () => {
    if (running) {
      runId += 1;
      destroyWorker();
      failRun("summary stopped", true);
      return;
    }

    try {
      const article = extractArticle();
      runId += 1;
      startedAt = performance.now();
      ui.result.hidden = true;
      setProgress(0);
      setRunning(true);
      setStatus("starting local summarizer", "loading", "[LOAD]");
      getWorker().postMessage({
        type: "summarize",
        id: runId,
        title: root.dataset.title,
        article,
      });
    } catch (error) {
      failRun(error.message || "Article text could not be prepared.");
    }
  };

  ui.run.addEventListener("click", generate);
  ui.regenerate.addEventListener("click", generate);
  ui.hide.addEventListener("click", () => {
    ui.output.hidden = true;
  });

  if (!("Worker" in window)) {
    ui.run.disabled = true;
    ui.note.textContent = "This browser does not support the worker required for local summarization.";
  } else {
    ui.note.textContent = "T5-small · private · runs in a worker · about 78 MB on first use";
    const cached = readCache();
    if (cached) renderSummary(cached);
  }
}
