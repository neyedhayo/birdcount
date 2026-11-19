"use strict";

const get = id => document.getElementById(id);
const html = (tag, className, label) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (label !== undefined) node.textContent = label;
  return node;
};

function svg(tag, attributes, label) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  return node;
}

function wavePath(samples, x, middle, width, amplitude) {
  const points = (side, position) => {
    const px = x + width * position / (samples.length - 1);
    const py = middle - samples[position][side] * amplitude;
    return `${px.toFixed(1)} ${py.toFixed(1)}`;
  };
  const top = samples.map((_, index) => points(1, index));
  const bottom = samples.map((_, index) => points(0, index)).reverse();
  return svg("path", { class: "wave", d: `M${top.join(" L")} L${bottom.join(" L")} Z` });
}

function waveform(samples, color) {
  const box = svg("svg", { viewBox: "0 0 160 66", preserveAspectRatio: "none", "aria-hidden": true });
  box.style.color = color || "var(--green)";
  box.append(wavePath(samples, 0, 33, 160, 30));
  return box;
}

function makeAudioButton(className, title, path, label = title) {
  const button = html("button", className, title);
  button.type = "button";
  button.dataset.audio = path;
  button.setAttribute("aria-label", label);
  return button;
}

function drawChallenge() {
  const calls = birdcountData.annotatedCalls;
  const examples = [
    { label: "Songs repeat", tracks: [[0, 15], [2, 248], [0, 493]] },
    { label: "Songs overlap", tracks: [[0, 185], [1, 247]] },
    { label: "Some birds stay quiet", tracks: [] },
  ];
  for (const { label, tracks } of examples) {
    const row = html("div", "challenge-row");
    const view = svg("svg", { viewBox: "0 0 740 110", "aria-hidden": true });
    view.append(svg("line", {
      x1: 0, y1: 55, x2: 740, y2: 55, stroke: "var(--line)", "stroke-width": 1.4,
      "stroke-dasharray": tracks.length ? "0" : "4 8",
    }));
    tracks.forEach(([call, start], index) => {
      const wave = wavePath(calls[call].wave, start, 55, 196, 47);
      wave.style.color = call === 1 ? "var(--rust)" : "var(--green)";
      wave.style.opacity = index === 1 && label === "Songs overlap" ? ".77" : "1";
      view.append(wave);
    });
    row.append(html("b", "", label), view);
    get("challenge-figure").append(row);
  }
}

function colorEmbedding(container, features) {
  const strongest = Math.max(...features.map(Math.abs), 1);
  for (const value of features) {
    const band = html("i");
    const weight = Math.round(14 + 86 * Math.abs(value) / strongest);
    const hue = value < 0 ? "var(--rust)" : "var(--green)";
    band.style.setProperty("--tone", `color-mix(in srgb, ${hue} ${weight}%, #f3f7ef)`);
    container.append(band);
  }
}

function drawAudio() {
  const { fieldWave, detectedCalls } = birdcountData;
  const plot = get("field-wave");
  const width = 1128;
  const timeX = seconds => 4 + seconds / 60 * width;
  const mini = get("lineage-wave");

  for (const [index, call] of detectedCalls.entries()) {
    const startX = timeX(call.start);
    plot.append(svg("rect", {
      x: startX, y: 0, width: timeX(call.start + 2) - startX, height: 126, fill: "var(--lime)",
    }));
    mini.append(svg("rect", { x: call.start / 60 * 280, y: 7, width: 9.33, height: 89, fill: "var(--lime)" }));
    const label = `${call.start.toFixed(1)} s`;
    const button = makeAudioButton("clip-button", "", call.audio, `Play detected clip ${index + 1}, starting at ${label}`);
    button.append(waveform(call.wave), html("span", "", label));
    get("detected-clips").append(button);
  }

  plot.append(wavePath(fieldWave, 4, 63, width, 62));
  plot.append(svg("line", { x1: 4, x2: 1132, y1: 130, y2: 130, class: "plot-tick" }));
  for (const seconds of [0, 10, 20, 30, 40, 50, 60]) {
    const x = timeX(seconds);
    plot.append(svg("line", { x1: x, x2: x, y1: 130, y2: 138, class: "plot-tick" }));
    plot.append(svg("text", {
      x, y: 162, class: "plot-text", "text-anchor": seconds === 0 ? "start" : seconds === 60 ? "end" : "middle",
    }, `${seconds} s`));
  }
  plot.append(svg("line", {
    id: "audio-playhead", x1: 4, x2: 4, y1: 0, y2: 126, stroke: "var(--rust)", "stroke-width": 2, visibility: "hidden",
  }));
  mini.style.color = "var(--green)";
  mini.append(wavePath(fieldWave, 0, 52, 280, 42));
}

function drawEmbeddings() {
  for (const [index, call] of birdcountData.annotatedCalls.entries()) {
    const row = html("div", "embedding-row");
    const source = html("div", "embedding-input");
    const arrow = html("span", "encoding-arrow", "⟶");
    const features = html("div", "barcode");
    const number = index + 1;
    source.append(makeAudioButton("song-button", number, call.audio, `Play annotated song ${number}`), waveform(call.wave));
    arrow.setAttribute("aria-hidden", "true");
    features.setAttribute("role", "img");
    features.setAttribute("aria-label", `Sample of embedding values for song ${number}`);
    colorEmbedding(features, call.features);
    row.append(source, arrow, features);
    get("embedding-diagram").append(row);
  }
  colorEmbedding(get("lineage-embedding"), birdcountData.annotatedCalls[0].features);
}

function drawGroups() {
  const projection = birdcountData.projection;
  const xs = projection.map(point => point[0]);
  const ys = projection.map(point => point[1]);
  const mapRange = (value, range) => (value - Math.min(...range)) / (Math.max(...range) - Math.min(...range));
  const points = projection.map(([x, y]) => ({
    x: 22 + 56 * mapRange(x, xs), y: 72 - 44 * mapRange(y, ys),
  }));
  const groups = birdcountData.merges.slice(0, 2).map(([first, second]) => [first, second]);
  const colors = ["cluster-a", "cluster-b"];

  function addPoints(container, grouped) {
    container.setAttribute("role", "img");
    if (grouped) {
      groups.forEach((group, index) => {
        const positions = group.map(call => points[call]);
        const left = Math.min(...positions.map(point => point.x));
        const right = Math.max(...positions.map(point => point.x));
        const top = Math.min(...positions.map(point => point.y));
        const bottom = Math.max(...positions.map(point => point.y));
        const ring = html("div", `point-ring ${colors[index]}`);
        Object.assign(ring.style, {
          left: `calc(${left}% - 36px)`, top: `calc(${top}% - 36px)`,
          width: `calc(${right - left}% + 72px)`, height: `calc(${bottom - top}% + 72px)`,
        });
        container.append(ring);
      });
    }
    points.forEach(({ x, y }, index) => {
      const group = groups.findIndex(members => members.includes(index));
      const dot = html("div", grouped ? `point ${colors[group]}` : "point", index + 1);
      Object.assign(dot.style, { left: `${x}%`, top: `${y}%` });
      container.append(dot);
    });
  }

  addPoints(get("call-points"), false);
  addPoints(get("group-points"), true);
}

function drawSweep() {
  const data = birdcountData.sweep;
  const chart = get("sweep-chart");
  const thresholdX = threshold => 74 + (threshold - 0.05) / 0.45 * 640;
  const scoreY = score => 309 - (score - 0.6) / 0.3 * 278;
  for (const tick of [0.6, 0.7, 0.8, 0.9]) {
    const y = scoreY(tick);
    chart.append(svg("line", { x1: 74, x2: 726, y1: y, y2: y, class: "plot-tick" }));
    chart.append(svg("text", { x: 57, y: y + 6, "text-anchor": "end", class: "plot-text" }, `${Math.round(tick * 100)}%`));
  }
  for (const tick of [0.05, 0.15, 0.25, 0.35, 0.45, 0.50]) {
    chart.append(svg("text", { x: thresholdX(tick), y: 343, "text-anchor": "middle", class: "plot-text" }, tick.toFixed(2)));
  }
  const cut = thresholdX(0.25);
  chart.append(svg("line", { x1: cut, x2: cut, y1: 29, y2: 309, class: "plot-selected" }));
  const coordinates = data.map(([threshold, score]) => [thresholdX(threshold), scoreY(score)]);
  chart.append(svg("polyline", { points: coordinates.map(pair => pair.join(",")).join(" "), class: "plot-line" }));
  coordinates.forEach(([cx, cy], index) => {
    chart.append(svg("circle", { cx, cy, r: data[index][0] === 0.25 ? 7 : 4, class: "plot-dot" }));
  });
  chart.append(svg("text", { x: 394, y: 380, "text-anchor": "middle", class: "plot-text" }, "Cosine-distance threshold"));
}

function renderDendrogram(stage) {
  const chart = get("dendrogram");
  const cut = 0.25;
  const yAt = distance => 354 - distance / .35 * 314;
  const leaves = [0, 2, 1, 3];
  const leafX = [155, 290, 488, 623];
  const clusters = [];
  chart.replaceChildren();

  chart.append(svg("text", { x: 14, y: 20, class: "plot-text" }, "Cosine distance"));
  for (const tick of [0, 0.1, 0.2, 0.3]) {
    const y = yAt(tick);
    chart.append(svg("line", { x1: 90, x2: 715, y1: y, y2: y, class: "plot-tick" }));
    chart.append(svg("text", { x: 72, y: y + 6, class: "plot-text", "text-anchor": "end" }, tick.toFixed(2)));
  }
  leaves.forEach((call, index) => {
    clusters[call] = [leafX[index], yAt(0)];
  });

  birdcountData.merges.forEach(([first, second, distance], index) => {
    const [leftX, leftY] = clusters[first];
    const [rightX, rightY] = clusters[second];
    const y = yAt(distance);
    clusters.push([(leftX + rightX) / 2, y]);

    if (index >= stage) return;
    const stroke = ["var(--green)", "var(--rust)", "#9ca9a0"][index];
    chart.append(svg("path", {
      d: `M${leftX} ${leftY} V${y} H${rightX} V${rightY}`,
      stroke, "stroke-width": index === 2 ? 2.5 : 4,
      "stroke-dasharray": index === 2 ? "5 5" : "none", fill: "none",
    }));
  });
  if (stage === 3) {
    const cutY = yAt(cut);
    chart.append(svg("line", { x1: 90, x2: 715, y1: cutY, y2: cutY, class: "plot-selected" }));
    chart.append(svg("text", { x: 714, y: cutY - 9, class: "plot-text", "text-anchor": "end" }, "0.25 cut"));
  }
  leaves.forEach((call, index) => {
    const joined = index < 2 ? stage > 0 : stage > 1;
    const color = index < 2 ? "var(--green)" : "var(--rust)";
    const x = leafX[index];
    chart.append(svg("circle", { cx: x, cy: yAt(0), r: 8, fill: joined ? color : "white", stroke: joined ? color : "var(--green)", "stroke-width": 2.5 }));
    chart.append(svg("text", { x, y: 390, class: "plot-text", "text-anchor": "middle" }, `Song ${call + 1}`));
  });
  get("merge-count").textContent = 4 - Math.min(stage, 2);
  for (const button of document.querySelectorAll("[data-merge]")) {
    button.setAttribute("aria-pressed", Number(button.dataset.merge) === stage);
  }
}

function drawMatrix() {
  const matrix = birdcountData.countMatrix;
  const table = html("table");
  table.append(html("caption", "", "Predicted count"));
  const head = table.createTHead().insertRow();
  head.append(html("th", "row-axis", "True count"));
  matrix.forEach((_, index) => {
    const cell = html("th", "", index + 1);
    cell.scope = "col";
    head.append(cell);
  });
  const rows = table.createTBody();
  matrix.forEach((values, row) => {
    const line = rows.insertRow();
    const label = html("th", "", row + 1);
    label.scope = "row";
    line.append(label);
    values.forEach((value, column) => {
      const cell = line.insertCell();
      if (row === column) cell.className = value > 15 ? "cell-strong" : "cell-correct";
      else if (value) cell.className = "cell-error";
      cell.textContent = value;
    });
  });
  get("count-matrix").append(table);
}

drawChallenge();
drawAudio();
drawEmbeddings();
drawGroups();
drawSweep();
renderDendrogram(0);
drawMatrix();

const slides = [...document.querySelectorAll(".slide")];
const overview = get("overview");
const notes = get("speaker-notes");
const player = get("audio-player");
const audioStatus = get("audio-status");
let currentSlide = 0;
let activeAudio = null;

function stopAudio() {
  player.pause();
  activeAudio?.classList.remove("playing");
  activeAudio = null;
  get("audio-playhead").setAttribute("visibility", "hidden");
}

function updateNotes() {
  const slide = slides[currentSlide];
  const source = slide.querySelector(".notes");
  const heading = slide.querySelector("h1, h2");
  notes.replaceChildren(html("strong", "", heading.textContent));
  if (source) notes.append(...Array.from(source.childNodes, node => node.cloneNode(true)));
}

const overviewButtons = slides.map((slide, index) => {
  const row = html("li");
  const button = html("button");
  const heading = slide.querySelector("h1, h2");
  button.type = "button";
  button.append(html("span", "", String(index + 1).padStart(2, "0")), heading.innerText.replace(/\s+/g, " "));
  button.addEventListener("click", () => {
    overview.close();
    showSlide(index);
  });
  row.append(button);
  get("slide-list").append(row);
  return button;
});

function showSlide(index, updateHash = true) {
  const next = Math.max(0, Math.min(index, slides.length - 1));
  if (next !== currentSlide) {
    stopAudio();
    audioStatus.hidden = true;
  }
  const focusWasOnSlide = slides[currentSlide].contains(document.activeElement);
  currentSlide = next;
  slides.forEach((slide, position) => { slide.hidden = position !== next; });
  overviewButtons.forEach((button, position) => button.setAttribute("aria-current", position === next));
  get("slide-number").textContent = `${next + 1} / ${slides.length}`;
  get("progress").style.width = `${(next + 1) / slides.length * 100}%`;
  get("previous").disabled = next === 0;
  get("next").disabled = next === slides.length - 1;
  if (focusWasOnSlide) {
    const heading = slides[next].querySelector("h1, h2");
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
  }
  if (!notes.hidden) updateNotes();
  if (updateHash) {
    const hash = `#${slides[next].id}`;
    if (location.hash !== hash) {
      try { history.replaceState(null, "", hash); }
      catch { location.hash = hash; }
    }
  }
}

function hashIndex() {
  const fragment = location.hash.slice(1);
  const numbered = /^slide-(\d+)$/.exec(fragment);
  if (numbered) return Number(numbered[1]) - 1;
  return Math.max(0, slides.findIndex(slide => slide.id === fragment));
}

function toggleOverview() {
  if (overview.open) overview.close();
  else {
    overview.showModal();
    overviewButtons[currentSlide].focus();
  }
}

function toggleNotes() {
  notes.hidden = !notes.hidden;
  get("notes-button").setAttribute("aria-expanded", !notes.hidden);
  if (!notes.hidden) updateNotes();
}

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch {
    get("fullscreen-button").hidden = true;
  }
}

function fitStage() {
  const viewport = get("viewport");
  document.documentElement.style.setProperty("--scale", Math.min(viewport.clientWidth / 1280, viewport.clientHeight / 720));
}

get("previous").addEventListener("click", () => showSlide(currentSlide - 1));
get("next").addEventListener("click", () => showSlide(currentSlide + 1));
get("overview-button").addEventListener("click", toggleOverview);
get("notes-button").addEventListener("click", toggleNotes);
get("fullscreen-button").addEventListener("click", toggleFullscreen);
get("fullscreen-button").hidden = !document.fullscreenEnabled;
document.addEventListener("fullscreenchange", () => {
  get("fullscreen-button").textContent = document.fullscreenElement ? "Exit full screen" : "Fullscreen";
  fitStage();
});
window.addEventListener("hashchange", () => showSlide(hashIndex(), false));
window.addEventListener("resize", fitStage);

document.addEventListener("keydown", event => {
  if (overview.open || event.metaKey || event.ctrlKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (event.target.closest?.("button, a, input, textarea, select") && [" ", "enter"].includes(key)) return;
  const commands = {
    arrowright: () => showSlide(currentSlide + 1),
    pagedown: () => showSlide(currentSlide + 1),
    " ": () => showSlide(currentSlide + 1),
    arrowleft: () => showSlide(currentSlide - 1),
    pageup: () => showSlide(currentSlide - 1),
    home: () => showSlide(0),
    end: () => showSlide(slides.length - 1),
    o: toggleOverview,
    n: toggleNotes,
    f: toggleFullscreen,
  };
  if (!Object.hasOwn(commands, key)) return;
  event.preventDefault();
  commands[key]();
});

for (const button of document.querySelectorAll("[data-merge]")) {
  button.addEventListener("click", () => renderDendrogram(Number(button.dataset.merge)));
}

for (const button of document.querySelectorAll("[data-audio]")) {
  button.addEventListener("click", async () => {
    const restart = activeAudio !== button;
    stopAudio();
    audioStatus.hidden = true;
    if (!restart) return;
    activeAudio = button;
    button.classList.add("playing");
    player.src = `media/${button.dataset.audio}`;
    try {
      await player.play();
    } catch {
      if (activeAudio !== button) return;
      stopAudio();
      audioStatus.textContent = "The audio file could not be played.";
      audioStatus.hidden = false;
    }
  });
}
player.addEventListener("ended", stopAudio);
player.addEventListener("timeupdate", () => {
  if (activeAudio?.dataset.audio !== "field-recording.mp3") return;
  const line = get("audio-playhead");
  const x = 4 + Math.min(60, player.currentTime) / 60 * 1128;
  line.setAttribute("x1", x);
  line.setAttribute("x2", x);
  line.setAttribute("visibility", "visible");
});

fitStage();
showSlide(hashIndex(), false);
