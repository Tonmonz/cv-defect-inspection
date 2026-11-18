// Visual Inspection demo page. Plain JavaScript talking to the FastAPI backend in app.py.
const $ = (selector, root = document) => root.querySelector(selector);
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const state = { models: [], summary: null, product: null, filter: "all", sample: null, result: null, seen: {}, example: {} };

// ---------- small helpers ----------
function el(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "style") Object.assign(node.style, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const kid of kids.flat()) if (kid != null) node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return node;
}
function svg(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
const num = (value, decimals = 0) => Number(value).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
const pct = (value, decimals = 0) => `${num(value * 100, decimals)}%`;
const nice = (kind) => kind.replace(/_/g, " ");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const current = () => state.models.find((m) => m.name === state.product);
const ICONS = {
  idle: "M5 12h14",
  busy: "M21 12a9 9 0 1 1-6.2-8.6",
  pass: "M20 6 9 17l-5-5",
  defect: "M12 8v5M12 16.5h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  error: "M18 6 6 18M6 6l12 12",
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return response.json();
}
function setStatus(kind, text) {
  $("#status").dataset.state = kind;
  $("#status-text").textContent = text;
}
function loadImage(img, src) {
  return new Promise((resolve) => {
    img.onload = resolve;
    img.onerror = resolve;
    img.src = src;
  });
}

// ---------- tabs ----------
function setupTabs() {
  const tabs = [...document.querySelectorAll(".tab")];
  const select = (tab) => {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    }
    if (tab.id === "tab-results") showResults();
    if (tab.id === "tab-how") buildHow();
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (e) => {
      const move = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!move) return;
      const next = tabs[(i + move + tabs.length) % tabs.length];
      next.focus();
      select(next);
    });
  });
  state.selectTab = (id) => select(document.getElementById(id));
}

// ---------- products ----------
function buildProducts() {
  $("#products").replaceChildren(...state.models.map((m) => el("button", {
    class: "product", type: "button", "aria-pressed": "false", "data-name": m.name, onclick: () => selectProduct(m.name),
  },
  el("img", { src: m.samples[0]?.url, alt: "" }),
  el("div", {}, el("b", { text: nice(m.name) }),
    el("span", { text: `AUROC ${num(m.metrics.image_auroc, 3)} · ${m.public ? "MVTec AD" : "your images"}` })))));
}
function selectProduct(name, autoInspect = true) {
  state.product = name;
  state.filter = "all";
  for (const b of $("#products").children) b.setAttribute("aria-pressed", b.dataset.name === name);
  const m = current();
  const s = m.metrics;
  const stats = [
    ["Image AUROC", num(s.image_auroc, 3)],
    ["Pixel AUROC", s.pixel_auroc == null ? "-" : num(s.pixel_auroc, 3)],
    ["Defects caught", `${s.true_positives}/${s.true_positives + s.false_negatives}`],
    ["Good parts flagged", `${s.false_positives}/${s.false_positives + s.true_negatives}`],
  ];
  $("#model-stats").replaceChildren(...stats.map(([k, v]) => el("div", { class: "stat" }, el("p", { text: `${nice(m.name)} · ${k}` }), el("b", { text: v }))));
  $("#threshold-note").textContent = `Learned from ${m.learned_from} good images only. The threshold is the highest score among ${m.calibration_images} other good images kept aside, so no defect was used to set it.`;
  renderFilters();
  renderTray();
  resetStage();
  const firstDefect = m.samples.find((x) => x.kind !== "good");
  if (autoInspect && firstDefect) inspectSample(firstDefect.index);
}
function renderFilters() {
  const kinds = ["all", ...new Set(current().samples.map((x) => x.kind))];
  $("#filters").replaceChildren(...kinds.map((k) => el("button", {
    class: "filter", type: "button", "aria-pressed": state.filter === k, text: k === "all" ? "All" : nice(k),
    onclick: () => { state.filter = k; renderFilters(); renderTray(); },
  })));
}
function renderTray() {
  const m = current();
  const shown = m.samples.filter((x) => state.filter === "all" || x.kind === state.filter);
  $("#tray").replaceChildren(...shown.map((x) => {
    const seen = state.seen[`${m.name}:${x.index}`];
    return el("button", {
      class: "thumb", type: "button", "aria-current": state.sample === x.index,
      "aria-label": `Inspect test image ${x.index + 1}, true label ${nice(x.kind)}`, onclick: () => inspectSample(x.index),
    },
    el("img", { src: x.url, alt: "", loading: "lazy" }),
    el("span", { class: `kind ${x.kind === "good" ? "good" : "bad"}`, text: nice(x.kind) }),
    seen ? el("span", { class: "seen", "data-correct": seen.correct, text: seen.correct ? "RIGHT" : "WRONG" }) : null);
  }));
}

// ---------- inspection ----------
function setVerdict(kind, text, sub) {
  const box = $("#verdict");
  box.dataset.state = kind;
  $("#verdict-icon").setAttribute("d", ICONS[kind]);
  $("#verdict-text").textContent = text;
  if (sub != null) $("#verdict-sub").textContent = sub;
}
function resetStage() {
  const stage = $("#stage");
  stage.dataset.ready = "false";
  stage.dataset.verdict = "";
  state.result = null;
  setVerdict("idle", "Waiting", "Choose a test image to start.");
  for (const id of ["fact-score", "fact-ms", "fact-truth", "fact-correct"]) {
    $(`#${id}`).textContent = "-";
    $(`#${id}`).removeAttribute("data-tone");
  }
  $("#meter-mark").style.left = "0%";
}
async function run(request) {
  const stage = $("#stage");
  stage.dataset.busy = "true";
  setVerdict("busy", "Scanning", "Comparing every patch with the memory of good parts…");
  try {
    const [result] = await Promise.all([request, new Promise((r) => setTimeout(r, reducedMotion ? 0 : 700))]);
    await Promise.all([loadImage($("#base"), result.image), loadImage($("#heat"), result.heat)]);
    showResult(result);
  } catch (err) {
    setVerdict("error", "Error", err.message);
  } finally {
    stage.dataset.busy = "false";
  }
}
function inspectSample(index) {
  state.sample = index;
  renderTray();
  $("#base").alt = `Test image of a ${nice(state.product)}, true label ${nice(current().samples[index].kind)}`;
  return run(api(`/api/inspect/${encodeURIComponent(state.product)}/sample/${index}`, { method: "POST" }));
}
function inspectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setVerdict("error", "Not an image", "Drop or paste a PNG, JPG or BMP picture.");
    return;
  }
  state.selectTab("tab-inspect");
  state.sample = null;
  renderTray();
  $("#base").alt = `Your image: ${file.name || "pasted picture"}`;
  const form = new FormData();
  form.append("file", file, file.name || "pasted.png");
  return run(api(`/api/inspect/${encodeURIComponent(state.product)}`, { method: "POST", body: form }));
}
function showResult(r) {
  state.result = r;
  const stage = $("#stage");
  stage.dataset.ready = "true";
  stage.dataset.verdict = r.defect ? "defect" : "pass";
  const reticle = $("#reticle");
  reticle.style.left = `${r.hotspot.x * 100}%`;
  reticle.style.top = `${r.hotspot.y * 100}%`;
  reticle.classList.toggle("flip-x", r.hotspot.x > 0.62);
  reticle.classList.toggle("flip-y", r.hotspot.y < 0.18);
  if (r.truth != null && r.truth !== "good" && r.defect) state.example[r.product] = r;
  $("#reticle-label").textContent = `hotspot ${num(r.score, 2)}x`;
  const ratio = num(r.score, 2);
  setVerdict(r.defect ? "defect" : "pass", r.defect ? "DEFECT" : "PASS", r.defect
    ? `The most unusual spot scores ${ratio}x the threshold: nothing the model remembers from good parts looks like it.`
    : `Every spot is close to something the model remembers from good parts (${ratio}x the threshold).`);
  $("#meter-mark").style.left = `${clamp(r.score / 2, 0, 1) * 100}%`;
  $("#fact-score").textContent = `${ratio}x`;
  $("#fact-ms").textContent = `${Math.round(r.ms)} ms`;
  const truth = $("#fact-truth");
  const correct = $("#fact-correct");
  if (r.truth != null) {
    truth.textContent = nice(r.truth);
    correct.textContent = r.correct ? "right" : "wrong";
    correct.dataset.tone = r.correct ? "good" : "bad";
    state.seen[`${r.product}:${state.sample}`] = { correct: r.correct };
    renderTray();
  } else {
    truth.textContent = "your image";
    correct.textContent = "-";
    correct.removeAttribute("data-tone");
  }
}
function setupStage() {
  const stage = $("#stage");
  for (const radio of document.querySelectorAll('input[name="mode"]')) {
    radio.addEventListener("change", () => { stage.dataset.mode = radio.value; });
  }
  const opacity = $("#opacity");
  opacity.addEventListener("input", () => {
    stage.style.setProperty("--heat-opacity", opacity.value / 100);
    opacity.style.setProperty("--fill", `${opacity.value}%`);
  });
  const compare = $("#compare");
  compare.addEventListener("input", () => stage.style.setProperty("--split", `${compare.value}%`));
  $("#file").addEventListener("change", (e) => inspectFile(e.target.files[0]));
  let depth = 0;
  document.addEventListener("dragenter", (e) => { if ([...e.dataTransfer.types].includes("Files")) { depth++; document.body.classList.add("dragging"); } });
  document.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) document.body.classList.remove("dragging"); });
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    document.body.classList.remove("dragging");
    if (e.dataTransfer.files[0]) inspectFile(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (e) => {
    const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
    if (file) inspectFile(file);
  });
}

// ---------- results ----------
function barRow(label, value, text, tone) {
  return el("div", { class: "bar-row", "data-tone": tone },
    el("span", { class: "label", text: label }), el("span", { class: "num", text }),
    el("span", { class: "track", "aria-hidden": "true" }, el("i", { "data-w": `${clamp(value, 0, 1) * 100}%` })));
}
function buildResults() {
  $("#result-cards").replaceChildren(...state.models.map((m) => {
    const s = m.metrics;
    return el("section", { class: "card result-card" },
      el("img", { src: m.samples.find((x) => x.kind !== "good")?.url || m.samples[0]?.url, alt: "" }),
      el("div", {}, el("h3", { text: nice(m.name) }), el("p", { class: "big" }, num(s.image_auroc, 3), el("small", { text: "image AUROC" }))),
      el("ul", {},
        el("li", {}, "Pixel AUROC ", el("b", { text: s.pixel_auroc == null ? "-" : num(s.pixel_auroc, 3) })),
        el("li", {}, "Defects caught ", el("b", { text: `${s.true_positives}/${s.true_positives + s.false_negatives}` })),
        el("li", {}, "Good parts flagged ", el("b", { text: `${s.false_positives}/${s.false_positives + s.true_negatives}` })),
        el("li", {}, "Learned from ", el("b", { text: `${m.learned_from} good images` })),
        el("li", {}, "Speed ", el("b", { text: `${Math.round(s.ms_per_image)} ms / image` }))));
  }));
  $("#auroc").replaceChildren(...state.models.map((m) => el("div", { class: "auroc-group" },
    el("h3", { text: nice(m.name) }),
    barRow("image AUROC", (m.metrics.image_auroc - 0.5) / 0.5, num(m.metrics.image_auroc, 3), "cyan"),
    m.metrics.pixel_auroc == null ? null : barRow("pixel AUROC", (m.metrics.pixel_auroc - 0.5) / 0.5, num(m.metrics.pixel_auroc, 3), "pixel"))));
  $("#product-reports").replaceChildren(...state.models.map((m) => el("section", { class: "card product-report" },
    el("div", { class: "card-head" }, el("div", {}, el("h2", { text: `${nice(m.name)}: result per type` }),
      el("p", { class: "muted small", text: `${m.test_images} test images · memory bank ${num(m.memory_bank)} of ${num(m.patches_seen)} good patches · ${m.source}` }))),
    el("div", { class: "grid" },
      el("div", { class: "auroc" }, ...m.metrics.per_kind.map((k) => {
        const good = k.kind === "good";
        const rate = good ? 1 - k.flagged / k.count : k.flagged / k.count;
        return barRow(good ? "good (passed)" : `${nice(k.kind)} (caught)`, rate, `${good ? k.count - k.flagged : k.flagged}/${k.count}`, rate >= 0.8 ? "pass" : "defect");
      })),
      el("div", { class: "gallery" }, ...m.figures.map((f) => el("button", { class: "fig", type: "button", onclick: () => openLightbox(f) },
        el("img", { src: f.url, alt: f.caption, loading: "lazy" }), el("span", { text: f.caption }))))))));
}
function showResults() {
  requestAnimationFrame(() => {
    for (const bar of document.querySelectorAll("#panel-results .track i")) bar.style.width = bar.dataset.w;
  });
}
function openLightbox(figure) {
  $("#lightbox-img").src = figure.url;
  $("#lightbox-img").alt = figure.caption;
  $("#lightbox-caption").textContent = figure.caption;
  $("#lightbox").showModal();
}

// ---------- how it works ----------
function buildHow() {
  const m = current();
  const good = m.samples.find((x) => x.kind === "good");
  const dots = svg("svg", { viewBox: "0 0 100 100", "aria-hidden": "true" });
  let seed = 7;
  const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 220; i++) {
    const keep = i % 12 === 0;
    dots.append(svg("circle", { cx: 8 + rand() * 84, cy: 8 + rand() * 84, r: keep ? 2.2 : 1.1, fill: keep ? "#22d3ee" : "rgba(148,166,196,0.35)" }));
  }
  // A correctly caught test defect of this product, so the heatmap step always shows a real example.
  const compare = el("div", { class: "visual" });
  const showExample = (r) => compare.replaceChildren(el("img", { src: r.image, alt: "" }), el("img", { src: r.heat, alt: "" }));
  const defects = m.samples.filter((x) => x.kind !== "good");
  if (state.example[m.name]) showExample(state.example[m.name]);
  else if (defects.length) {
    compare.append(el("img", { src: defects[0].url, alt: "" }));
    (async () => {
      for (const sample of defects) {
        const r = await api(`/api/inspect/${encodeURIComponent(m.name)}/sample/${sample.index}`, { method: "POST" });
        if (r.defect) { state.example[m.name] = r; showExample(r); return; }
      }
    })().catch(() => {});
  }
  const meter = svg("svg", { viewBox: "0 0 100 100", "aria-hidden": "true" });
  const bars = [6, 14, 26, 18, 9, 4, 2, 1];
  bars.forEach((h, i) => meter.append(svg("rect", { x: 10 + i * 8, y: 70 - h * 2, width: 6, height: h * 2, rx: 1.5, fill: "#34d399" })));
  [2, 4, 7, 5, 3].forEach((h, i) => meter.append(svg("rect", { x: 58 + i * 7, y: 70 - h * 3, width: 5, height: h * 3, rx: 1.5, fill: "#fb7185" })));
  meter.append(svg("line", { x1: 57, y1: 14, x2: 57, y2: 76, stroke: "#e8f1ff", "stroke-width": 1.2, "stroke-dasharray": "3 2" }));
  const steps = [
    ["See", "28 x 28 patches", "A Wide ResNet-50 pretrained on ImageNet, never retrained, describes every patch of the image with 1,536 numbers.",
      el("div", { class: "visual" }, el("img", { src: good?.url, alt: "" }), el("div", { class: "grid28" }))],
    ["Remember", `${num(m.patches_seen)} → ${num(m.memory_bank)}`, `All patches of ${m.learned_from} good images go into a memory bank. A greedy coreset keeps 1% that still cover the rest.`,
      el("div", { class: "visual" }, dots)],
    ["Compare", "nearest good patch", "Each patch of a new image is matched to its closest remembered good patch. Far away means unusual. The distances, smoothed, are the heatmap.",
      compare],
    ["Decide", "threshold = max(good)", `The threshold is the highest score among ${m.calibration_images} good images kept aside. Anything above it is called a defect.`,
      el("div", { class: "visual" }, meter)],
  ];
  $("#steps").replaceChildren(...steps.map(([title, big, text, visual]) => el("li", { class: "step" },
    visual, el("h3", { text: title }), el("span", { class: "big", text: big }), el("p", { text }))));
}

// ---------- start ----------
async function init() {
  setupTabs();
  setupStage();
  try {
    const data = await api("/api/models");
    state.models = data.models;
    state.summary = data.summary;
  } catch (err) {
    setStatus("error", `Models not reachable: ${err.message}`);
    return;
  }
  setStatus("online", `Model ready · ${state.models.length} product${state.models.length === 1 ? "" : "s"}`);
  buildProducts();
  buildResults();
  $("#lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") e.target.close(); });
  selectProduct(state.models[0].name);
}
init();
