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

function drawPairwiseMLP() {
  const { features, pairs, cutoff, labels } = birdcountData.pairwiseMLP;
  const graph = get("pairwise-graph");
  const score = get("pairwise-score");
  const pairList = get("pairwise-pairs");
  const links = pairs.filter(([, , probability]) => probability >= cutoff);
  const positions = [42, 231, 127, 316];
  const graphLinks = links.map(([a, b]) => `${a + 1} and ${b + 1}`).join(", ");
  graph.setAttribute("aria-label", `Song graph with links between songs ${graphLinks}`);
  get("pairwise-cutoff").textContent = `Link when score ≥ ${cutoff.toFixed(2)}`;
  get("pairwise-count").textContent = new Set(labels).size;

  function showGraph(left, right, linked) {
    graph.replaceChildren();
    for (const [a, b] of links) {
      graph.append(svg("line", {
        x1: positions[a], x2: positions[b], y1: 45, y2: 45,
        class: a === left && b === right ? "pairwise-edge selected" : "pairwise-edge",
      }));
    }
    const emphasis = linked ? " selected" : " rejected";
    positions.forEach((x, song) => {
      const state = song === left || song === right ? emphasis : "";
      graph.append(
        svg("circle", { cx: x, cy: 45, r: 19, class: `pairwise-node${state}` }),
        svg("text", {
          x, y: 46, "text-anchor": "middle", "dominant-baseline": "middle",
          class: `pairwise-node-label${state}`,
        }, song + 1),
      );
    });
  }

  function select(index) {
    const [left, right, probability] = pairs[index];
    const a = features[left];
    const b = features[right];
    const linked = probability >= cutoff;

    get("pairwise-first-label").textContent = `Song ${left + 1}`;
    get("pairwise-second-label").textContent = `Song ${right + 1}`;
    const vectors = [
      ["pairwise-first", a],
      ["pairwise-second", b],
      ["pairwise-difference", a.map((value, i) => Math.abs(value - b[i]))],
      ["pairwise-product", a.map((value, i) => value * b[i])],
    ];
    for (const [id, values] of vectors) {
      const strip = get(id);
      strip.replaceChildren();
      colorEmbedding(strip, values);
    }

    score.querySelector("strong").textContent = probability.toFixed(3);
    score.querySelector("span").textContent = linked ? "Link these songs" : "No direct link";
    score.classList.toggle("rejected", !linked);
    buttons.forEach((button, position) => button.setAttribute("aria-pressed", position === index));
    showGraph(left, right, linked);
  }

  const buttons = pairs.map(([left, right], index) => {
    const button = html("button", "", `${left + 1} + ${right + 1}`);
    button.type = "button";
    button.setAttribute("aria-label", `Compare songs ${left + 1} and ${right + 1}`);
    button.addEventListener("click", () => select(index));
    pairList.append(button);
    return button;
  });
  select(pairs.findIndex(([left, right]) => left === 0 && right === 2));
}

function drawLinkage() {
  const cutoff = 0.25;
  const { songDistances, merges } = birdcountData;
  const clusters = songDistances.map((_, song) => [song]);
  let active = clusters.map((_, index) => index);
  const steps = [{ title: `Start with ${active.length}`, before: active, after: active }];
  const name = members => members.map(song => song + 1).join(" + ");

  for (const [left, right, distance] of merges) {
    const before = active;
    const blocked = distance >= cutoff;
    if (!blocked) {
      const members = [...clusters[left], ...clusters[right]].sort((a, b) => a - b);
      active = [...before.filter(cluster => cluster !== left && cluster !== right), clusters.length];
      clusters.push(members);
      active.sort((a, b) => clusters[a][0] - clusters[b][0]);
    }
    const title = blocked
      ? `Stop with ${active.length}`
      : `Merge ${name(clusters[left])} + ${name(clusters[right])}`;
    steps.push({ title, before, after: active, pair: [left, right], distance, blocked });
    if (blocked) break;
  }

  function meanDistance(left, right) {
    let sum = 0;
    for (const a of left) {
      for (const b of right) sum += songDistances[a][b];
    }
    return sum / (left.length * right.length);
  }

  function makeDistanceTable(step) {
    const crossCluster = step.blocked;
    const precision = crossCluster ? 4 : 3;
    const rowGroups = crossCluster
      ? clusters[step.pair[0]].map(song => [song])
      : step.before.map(cluster => clusters[cluster]);
    const columnGroups = crossCluster ? clusters[step.pair[1]].map(song => [song]) : rowGroups;
    const table = html("table", "merge-matrix");
    table.setAttribute("aria-label", crossCluster ? "Cross-cluster song distances" : "Candidate cluster distances");
    const header = table.createTHead().insertRow();
    header.append(html("th", "", crossCluster ? "Song" : "Cluster"));
    columnGroups.forEach(group => {
      const label = html("th", crossCluster ? "cluster-b" : "", name(group));
      label.scope = "col";
      header.append(label);
    });

    const body = table.createTBody();
    rowGroups.forEach((group, row) => {
      const line = body.insertRow();
      const label = html("th", crossCluster ? "cluster-a" : "", name(group));
      label.scope = "row";
      line.append(label);
      columnGroups.forEach((other, column) => {
        const cell = line.insertCell();
        if (!crossCluster && column <= row) return;
        cell.textContent = meanDistance(group, other).toFixed(precision);
        if (!crossCluster && step.pair?.includes(step.before[row]) && step.pair.includes(step.before[column])) {
          cell.className = "nearest";
        }
      });
    });
    return table;
  }

  function showStep(step) {
    const detail = get("merge-detail");
    const caption = html("figcaption");
    const heading = step.blocked ? "Why the last merge is skipped" : step.pair ? "Closest pair before merging" : "Starting distances";
    const subtitle = step.blocked ? "Song-to-song cosine distances" : "Cosine distance averaged across each pair of clusters";
    caption.append(html("h3", "merge-detail-title", heading), html("p", "small-label", subtitle));
    detail.replaceChildren(caption, makeDistanceTable(step));
    if (!step.pair) return;

    const compare = html("div", step.blocked ? "merge-comparison blocked" : "merge-comparison");
    const distance = html("div");
    const threshold = html("div");
    const pairCount = clusters[step.pair[0]].length * clusters[step.pair[1]].length;
    const distanceLabel = step.blocked ? `Mean of ${pairCount} distances` : "Closest distance";
    distance.append(html("strong", "", step.distance.toFixed(3)), html("small", "", distanceLabel));
    threshold.append(html("strong", "", cutoff.toFixed(3)), html("small", "", "Cutoff"));
    compare.append(distance, html("span", "", step.blocked ? ">" : "<"), threshold);
    detail.append(compare);
  }

  const list = get("merge-steps");
  const buttons = steps.map(step => {
    const row = html("li");
    const button = html("button", step.blocked ? "merge-row blocked" : "merge-row");
    button.type = "button";
    button.setAttribute("aria-controls", "merge-detail");

    const groups = html("span", "merge-groups");
    const groupNames = step.after.map(index => name(clusters[index]));
    groups.setAttribute("aria-label", `Clusters: ${groupNames.join("; ")}`);
    groups.setAttribute("role", "img");
    for (const index of step.after) {
      const members = clusters[index];
      const color = index % 2 === 0 ? "cluster-a" : "cluster-b";
      const group = html("span", members.length === 1 ? "merge-group" : `merge-group ${color}`);
      for (const song of members) group.append(html("span", "merge-song", song + 1));
      groups.append(group);
    }
    button.append(html("span", "", step.title), groups, html("span", "merge-distance", step.distance?.toFixed(3) ?? "—"));
    row.append(button);
    list.append(row);
    return button;
  });

  function select(index) {
    buttons.forEach((button, position) => button.setAttribute("aria-pressed", position === index));
    showStep(steps[index]);
  }

  buttons.forEach((button, index) => button.addEventListener("click", () => select(index)));
  select(steps.length - 1);
}

function drawMethodAccuracy() {
  const ranking = get("accuracy-ranking");
  const methods = birdcountData.methodAccuracy;
  const sampleSizes = new Set(methods.map(method => method.recordings));
  const sharedSample = sampleSizes.size === 1;

  for (const { label, accuracy, recordings } of methods) {
    const row = html("li");
    const track = html("span", "accuracy-track");
    const fill = html("span", "accuracy-fill");
    const percentage = accuracy * 100;
    fill.style.width = `${percentage}%`;
    track.setAttribute("aria-hidden", "true");
    track.append(fill);
    row.append(
      html("span", "accuracy-method", sharedSample ? label : `${label} (n = ${recordings})`),
      track,
      html("span", "accuracy-value", `${percentage.toFixed(1)}%`),
    );
    ranking.append(row);
  }
  get("accuracy-sample-size").textContent = sharedSample
    ? `${methods[0].recordings} held-out recordings`
    : "Held-out recordings";
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
drawPairwiseMLP();
drawLinkage();
drawMethodAccuracy();
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
