/* /how: the debt walkthrough, diagrams, and the measurement charts. */
(() => {
  if ("scrollRestoration" in history) history.scrollRestoration = "manual"
  const NS = "http://www.w3.org/2000/svg"
  const el = (tag, attrs = {}, text) => {
    const n = document.createElementNS(NS, tag)
    for (const k in attrs) n.setAttribute(k, attrs[k])
    if (text != null) n.textContent = text
    return n
  }
  const tip = document.getElementById("tip")
  const showTip = (e, html) => {
    tip.innerHTML = html
    tip.style.opacity = 1
    tip.style.left = Math.min(e.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px"
    tip.style.top = e.clientY + 14 + "px"
  }
  const hideTip = () => { tip.style.opacity = 0 }
  const hover = (node, html, on, off) => {
    node.setAttribute("data-hit", "")
    node.addEventListener("mousemove", (e) => { showTip(e, html()); on && on() })
    node.addEventListener("mouseleave", () => { hideTip(); off && off() })
  }
  const mount = (id, title, svg, caption) => {
    const fig = document.getElementById(id)
    if (title) {
      fig.appendChild(Object.assign(document.createElement("div"), { className: "label", textContent: title }))
      fig.lastChild.style.marginBottom = ".6rem"
    }
    fig.appendChild(svg)
    fig.appendChild(Object.assign(document.createElement("figcaption"), { textContent: caption }))
  }
  const svgOf = (W, H, label) => el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label })

  /* ---- letter tape ---------------------------------------------------- */
  ;(() => {
    const L = ["No", "will", "I", "hate", "me", "here"], R = ["Ere", "he", "met", "a", "hill", "I", "won"]
    const letters = (L.join("") + R.join("")).toLowerCase(), N = letters.length, half = N / 2
    const W = 640, H = 178, pad = 14, gap = 12, base = 128
    const cw = (W - 2 * pad - gap) / N
    const cx = (i) => pad + (i + 0.5) * cw + (i >= half ? gap : 0)
    const svg = svgOf(W, H, "The letters of a palindrome mirrored across its centre; the two halves split the same letters into different words.")
    const arcs = [], cells = []
    for (let i = 0; i < half; i++) {
      const a = cx(i), b = cx(N - 1 - i), r = (b - a) / 2
      arcs.push(svg.appendChild(el("path", { d: `M${a},${base - 16}A${r},${Math.min(r * 0.32, 92)} 0 0 1 ${b},${base - 16}`,
        fill: "none", stroke: "var(--haze)", "stroke-width": 1.2 })))
    }
    svg.appendChild(el("line", { x1: W / 2, x2: W / 2, y1: 20, y2: base + 6, stroke: "var(--ink)", "stroke-dasharray": "2 3" }))
    for (let i = 0; i < N; i++) cells.push(svg.appendChild(el("text", { x: cx(i), y: base, "text-anchor": "middle", class: "m" }, letters[i])))
    const pair = (i, on) => {
      const j = N - 1 - i, k = Math.min(i, j)
      arcs[k].setAttribute("stroke", on ? "var(--signal)" : "var(--haze)")
      arcs[k].setAttribute("stroke-width", on ? 2 : 1.2)
      for (const n of [i, j]) cells[n].style.fill = on ? "var(--signal)" : ""
    }
    for (let i = 0; i < N; i++) {
      const hit = svg.appendChild(el("rect", { x: cx(i) - cw / 2, y: base - 16, width: cw, height: 22, fill: "transparent" }))
      hover(hit, () => `“${letters[i]}”: letter ${i + 1} and letter ${N - i} from the left`, () => pair(i, true), () => pair(i, false))
    }
    let at = 0
    for (const w of [...L, ...R]) {
      const a = cx(at) - cw / 2 + 2, b = cx(at + w.length - 1) + cw / 2 - 2
      svg.appendChild(el("line", { x1: a, x2: b, y1: base + 10, y2: base + 10, stroke: "var(--grey)", "stroke-width": 1.5 }))
      svg.appendChild(el("text", { x: (a + b) / 2, y: base + 28, "text-anchor": "middle" }, w))
      at += w.length
    }
    mount("tape", null, svg, "Each letter is tied to its mirror across the centre; hover one to see its partner. The halves share letters, not words: “no will I hate me here” against “ere he met a hill I won”.")
  })()

  /* ---- walkthrough ---------------------------------------------------- */
  const STEPS = [
    { l: ["hate", "me", "here"], r: [], owed: ["right", "erehemetah"], n: 3, side: "l",
      note: "Write “hate me here” on the left. The right half now owes those ten letters, reversed." },
    { l: ["hate", "me", "here"], r: ["ere", "he", "met"], owed: ["right", "ah"], n: 3, side: "r",
      note: "The right pays with real words. “ere he met” covers eight of the ten letters." },
    { l: ["hate", "me", "here"], r: ["ere", "he", "met", "a"], owed: ["right", "h"], n: 1, side: "r",
      note: "“a” pays one more. One letter is still owed." },
    { l: ["hate", "me", "here"], r: ["ere", "he", "met", "a", "hill."], owed: ["left", "lli"], n: 1, side: "r",
      note: "“hill” pays the h and overshoots by “ill”. The surplus flips: now the LEFT owes, and must start with letters that reverse to “ill”." },
    { l: ["Will", "I", "hate", "me", "here?"], r: ["Ere", "he", "met", "a", "hill."], owed: ["right", "iw"], n: 2, side: "l",
      note: "“Will I” pays “lli” and overshoots by “wi”. The debt crosses back to the right." },
    { l: ["Will", "I", "hate", "me", "here?"], r: ["Ere", "he", "met", "a", "hill.", "I", "won."], owed: ["left", "no"], n: 2, side: "r",
      note: "“I won” pays “iw” and leaves “on” over. The left owes once more." },
    { l: ["No,", "will", "I", "hate", "me", "here?"], r: ["Ere", "he", "met", "a", "hill.", "I", "won."], owed: null, n: 1, side: "l",
      note: "“No” settles it. Nothing is owed, so the paragraph is a palindrome: nowillihatemehere | erehemetahilliwon." },
  ]
  let at = 0
  const onStep = []
  const words = (ws, fresh, from) => ws.map((w, i) => {
    const isNew = from === "end" ? i >= ws.length - fresh : i < fresh
    return isNew ? `<span class="new">${w}</span>` : w
  }).join(" ")
  const dots = document.getElementById("w-dots")
  STEPS.forEach(() => dots.appendChild(document.createElement("span")))
  const draw = () => {
    const s = STEPS[at]
    document.getElementById("w-left").innerHTML = words(s.l, s.side === "l" ? s.n : 0, "start") || "&nbsp;"
    document.getElementById("w-right").innerHTML = words(s.r, s.side === "r" ? s.n : 0, "end") || "&nbsp;"
    document.getElementById("w-owed").innerHTML = s.owed
      ? `${s.owed[0]} owes <span class="debt">${s.owed[1]}</span>` : "nothing"
    document.getElementById("w-note").textContent = s.note
    ;[...dots.children].forEach((d, i) => d.classList.toggle("on", i === at))
    document.getElementById("w-back").disabled = at === 0
    document.getElementById("w-next").textContent = at === STEPS.length - 1 ? "Again" : "Next"
    onStep.forEach((f) => f(at))
  }
  /* ---- debt per step ------------------------------------------------- */
  ;(() => {
    // Diverging: the right half's debt rises above the centre line, the left's hangs below it.
    const W = 640, H = 230, left = 70, mid = 100, reach = 70, max = Math.max(...STEPS.map((s) => s.owed ? s.owed[1].length : 0))
    const bw = (W - left) / STEPS.length
    const svg = svgOf(W, H, "Letters owed after each step of the walkthrough: above the line when the right half owes, below when the left does.")
    svg.appendChild(el("line", { x1: left, x2: W, y1: mid, y2: mid, stroke: "var(--ink)" }))
    svg.appendChild(el("text", { x: left - 10, y: mid - reach / 2, "text-anchor": "end" }, "right owes"))
    svg.appendChild(el("text", { x: left - 10, y: mid + reach / 2 + 4, "text-anchor": "end" }, "left owes"))
    const groups = STEPS.map((s, i) => {
      const n = s.owed ? s.owed[1].length : 0, h = (reach * n) / max, x = left + i * bw + bw * 0.3
      const up = !s.owed || s.owed[0] === "right"
      const g = svg.appendChild(el("g", {}))
      g.appendChild(el("rect", { x, y: up ? mid - Math.max(h, 1.5) : mid, width: bw * 0.4, height: Math.max(h, 1.5), rx: 3,
        fill: up ? "var(--signal)" : "var(--ink)" }))
      g.appendChild(el("text", { x: x + bw * 0.2, y: up ? mid - h - 6 : mid + h + 14, "text-anchor": "middle", class: "val" }, n))
      g.appendChild(el("text", { x: x + bw * 0.2, y: H - 8, "text-anchor": "middle" }, `step ${i + 1}`))
      const hit = svg.appendChild(el("rect", { x: left + i * bw, y: 0, width: bw, height: H, fill: "transparent" }))
      hit.style.cursor = "pointer"
      hover(hit, () => s.owed ? `<b>Step ${i + 1}</b><br>${s.owed[0]} owes “${s.owed[1]}”` : `<b>Step ${i + 1}</b><br>nothing owed`)
      hit.addEventListener("click", () => { at = i; draw() })
      return g
    })
    onStep.push((k) => groups.forEach((g, i) => g.setAttribute("opacity", i === k ? 1 : 0.35)))
    mount("debt", null, svg, "Letters owed at each step of the walkthrough. Every overpayment flips the debt across the line to the other half; it closes at zero. Click a step to show it above.")
  })()

  document.getElementById("w-back").onclick = () => { at = Math.max(0, at - 1); draw() }
  document.getElementById("w-next").onclick = () => { at = at === STEPS.length - 1 ? 0 : at + 1; draw() }
  draw()

  /* ---- beam diagram --------------------------------------------------- */
  ;(() => {
    const W = 640, H = 230, cols = 4, per = 8, keep = 3
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Each round, every kept candidate is extended; the candidates are scored and only the best few survive." })
    const x = (c) => 40 + c * ((W - 80) / (cols - 1))
    const y = (i) => 18 + i * ((H - 60) / (per - 1))
    // Deterministic "scores": the kept ones are the top `keep` of each round.
    for (let c = 0; c < cols; c++) {
      if (c > 0) for (let k = 0; k < keep; k++) for (let i = 0; i < per; i++) {
        if ((i + k) % 3) continue
        svg.appendChild(el("line", { x1: x(c - 1) + 6, y1: y(k * 2 + 1), x2: x(c) - 6, y2: y(i),
          stroke: "var(--haze)", "stroke-width": 1.5 }))
      }
      for (let i = 0; i < per; i++) {
        const kept = c === cols - 1 ? i === 1 : [1, 3, 5].includes(i)
        svg.appendChild(el("circle", { cx: x(c), cy: y(i), r: 6,
          fill: kept ? "var(--signal)" : "var(--paper)", stroke: kept ? "var(--signal)" : "var(--grey)", "stroke-width": 2 }))
      }
      svg.appendChild(el("text", { x: x(c), y: H - 14, "text-anchor": "middle" },
        c === cols - 1 ? "closed" : `round ${c + 1}`))
    }
    const fig = document.getElementById("beam")
    fig.appendChild(svg)
    fig.appendChild(Object.assign(document.createElement("figcaption"), {
      textContent: "Beam search, schematic. Every kept candidate (filled) is extended by every legal word; all of them are scored and only the best survive to the next round. The real beam holds hundreds per round." }))
  })()

  /* ---- readability scale ------------------------------------------- */
  ;(() => {
    const W = 640, H = 170
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "The verifier's readable probability rises from repetitive garbage to readable sentences." })
    const x0 = 30, x1 = W - 30, base = 110
    let d = ""
    for (let i = 0; i <= 100; i++) {
      const t = i / 100, v = 1 / (1 + Math.exp(-(t - 0.5) * 12))
      d += `${i ? "L" : "M"}${x0 + t * (x1 - x0)},${base - v * 80}`
    }
    svg.appendChild(el("path", { d: d + `L${x1},${base}L${x0},${base}Z`, fill: "var(--signal-soft)" }))
    svg.appendChild(el("path", { d, fill: "none", stroke: "var(--signal)", "stroke-width": 2 }))
    svg.appendChild(el("line", { x1: x0, x2: x1, y1: base, y2: base, stroke: "var(--ink)", "stroke-width": 1 }))
    // Two real sentences from the live generator, placed at the probability the verifier gave them.
    for (const [p, s] of [[0.242, "Still act on did I wonder."], [0.908, "So help me to stay a lion, till I know."]]) {
      const t = 0.5 + Math.log(p / (1 - p)) / 12, x = x0 + t * (x1 - x0), y = base - p * 80
      svg.appendChild(el("circle", { cx: x, cy: y, r: 5, fill: "var(--paper)", stroke: "var(--signal)", "stroke-width": 2 }))
      svg.appendChild(el("text", { x, y: base + 24, "text-anchor": "middle", class: "val" }, s))
      svg.appendChild(el("text", { x: x + 10, y: y + 4 }, p.toFixed(2)))
    }
    svg.appendChild(el("text", { x: x0, y: base + 46 }, "repetitive garbage"))
    svg.appendChild(el("text", { x: x1, y: base + 46, "text-anchor": "end" }, "readable English"))
    svg.appendChild(el("text", { x: x0, y: 18 }, "probability readable"))
    const fig = document.getElementById("spectrum")
    fig.appendChild(svg)
    fig.appendChild(Object.assign(document.createElement("figcaption"), {
      textContent: "The verifier turns counts into a probability that a sentence is readable. Both sentences come from the live generator; the beam rewards the second and punishes the first." }))
  })()

  /* ---- nested shells ---------------------------------------------------- */
  ;(() => {
    const parts = ["L3", "L2", "L1", "racecar", "R1", "R2", "R3"]
    const W = 640, H = 150, bw = 70, seamW = 104, gap = 10, y = 30, bh = 34
    const total = 6 * bw + seamW + 6 * gap
    let cur = (W - total) / 2
    const box = parts.map((p) => { const w = p === "racecar" ? seamW : bw, b = { p, x: cur, w }; cur += w + gap; return b })
    const svg = svgOf(W, H, "Nested shells L3 L2 L1, a seam insert, then R1 R2 R3; each Lk mirrors Rk.")
    const arcs = {}, rects = {}
    for (let k = 1; k <= 3; k++) {
      const a = box[3 - k], b = box[3 + k], ax = a.x + a.w / 2, bx = b.x + b.w / 2, r = (bx - ax) / 2
      arcs[k] = svg.appendChild(el("path", { d: `M${ax},${y + bh + 4}A${r},${18 + k * 14} 0 0 0 ${bx},${y + bh + 4}`,
        fill: "none", stroke: "var(--grey)", "stroke-width": 1.2 }))
    }
    box.forEach((b) => {
      const seam = b.p === "racecar"
      rects[b.p] = svg.appendChild(el("rect", { x: b.x, y, width: b.w, height: bh, rx: 6,
        fill: seam ? "var(--signal-soft)" : "var(--paper)", stroke: seam ? "var(--signal)" : "var(--grey)", "stroke-width": 1.5 }))
      svg.appendChild(el("text", { x: b.x + b.w / 2, y: y + 22, "text-anchor": "middle", class: seam ? "m" : "val" }, b.p))
      if (seam) svg.lastChild.style.fill = "var(--signal)"
    })
    svg.appendChild(el("text", { x: W / 2, y: y - 10, "text-anchor": "middle" }, "seam"))
    const light = (k, on) => {
      arcs[k].setAttribute("stroke", on ? "var(--signal)" : "var(--grey)")
      for (const p of ["L" + k, "R" + k]) rects[p].setAttribute("stroke", on ? "var(--signal)" : "var(--grey)")
    }
    box.forEach((b) => {
      const hit = svg.appendChild(el("rect", { x: b.x, y: 0, width: b.w, height: H, fill: "transparent" }))
      const k = +b.p[1]
      if (b.p === "racecar") hover(hit, () => "A palindrome at the centre seam keeps the whole exact.")
      else hover(hit, () => `${b.p} is ${b.p[0] === "L" ? "R" : "L"}${k} reversed, letter for letter`, () => light(k, true), () => light(k, false))
    })
    mount("shells", null, svg, "Schematic of v3 and v4. Each shell pair mirrors letter for letter around one centre, so any palindrome dropped into the centre seam, here racecar, leaves the whole paragraph exact.")
  })()

  /* ---- measurement charts --------------------------------------------- */
  const LABELS = {
    brown: "English prose (Brown)", classics: "Classic palindromes", v1: "v1 generator",
    v5_live: "This generator, 300 letters", v5_sophia: "This generator, 600–1,000 letters",
  }
  const ORDER = ["brown", "classics", "v1", "v5_live", "v5_sophia"]
  const bars = (id, title, key, fmt, caption, data) => {
    const rows = ORDER.filter((k) => data[k] && data[k][key] != null)
    const W = 640, rowH = 34, left = 210, right = 70, H = rows.length * rowH + 8
    const max = Math.max(...rows.map((k) => data[k][key])) || 1
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title })
    rows.forEach((k, i) => {
      const v = data[k][key], w = ((W - left - right) * v) / max, yy = i * rowH + 6
      const ours = k.startsWith("v5")
      svg.appendChild(el("text", { x: left - 10, y: yy + 16, "text-anchor": "end" }, LABELS[k]))
      const bar = el("rect", { x: left, y: yy + 4, width: Math.max(2, w), height: 16, rx: 4,
        fill: ours ? "var(--signal)" : "var(--grey)" })
      svg.appendChild(bar)
      svg.appendChild(el("text", { x: left + Math.max(2, w) + 8, y: yy + 16, class: "val" }, fmt(v)))
      const hit = el("rect", { x: 0, y: yy, width: W, height: rowH, fill: "transparent" })
      hit.addEventListener("mousemove", (e) => showTip(e, `<b>${LABELS[k]}</b><br>${fmt(v)} · ${data[k].n} texts`))
      hit.addEventListener("mouseleave", hideTip)
      svg.appendChild(hit)
    })
    const fig = document.getElementById(id)
    fig.appendChild(Object.assign(document.createElement("div"), { className: "label", textContent: title }))
    fig.lastChild.style.marginBottom = ".6rem"
    fig.appendChild(svg)
    fig.appendChild(Object.assign(document.createElement("figcaption"), { textContent: caption }))
  }
  const pct = (v) => `${Math.round(v * 100)}%`

  /* ---- distributions -------------------------------------------------- */
  const strip = (fig, data) => {
    const keys = ["brown", "v1", "v5_live", "v5_sophia"].filter((k) => fig.gpt2[k])
    const W = 640, rowH = 34, left = 210, right = 20, top = 8, H = keys.length * rowH + top + 30
    const lo = Math.log(20), hi = Math.log(1500), x = (v) => left + ((Math.log(v) - lo) / (hi - lo)) * (W - left - right)
    const svg = svgOf(W, H, "GPT-2 perplexity of every text, per corpus, on a log scale.")
    for (const t of [20, 50, 100, 200, 500, 1000]) {
      svg.appendChild(el("line", { x1: x(t), x2: x(t), y1: top, y2: H - 26, stroke: "var(--haze)" }))
      svg.appendChild(el("text", { x: x(t), y: H - 8, "text-anchor": "middle" }, t.toLocaleString()))
    }
    keys.forEach((k, r) => {
      const vs = fig.gpt2[k], yy = top + r * rowH + rowH / 2, ours = k.startsWith("v5")
      svg.appendChild(el("text", { x: left - 10, y: yy + 4, "text-anchor": "end" }, LABELS[k]))
      vs.forEach((v, i) => svg.appendChild(el("circle", { cx: x(v), cy: yy + (((i * 0.618) % 1) - 0.5) * 16, r: 2.6,
        fill: ours ? "var(--signal)" : "var(--grey)", "fill-opacity": 0.55 })))
      const m = data[k].gpt2_ppl_median
      svg.appendChild(el("line", { x1: x(m), x2: x(m), y1: yy - 12, y2: yy + 12, stroke: "var(--ink)", "stroke-width": 2 }))
      const hit = svg.appendChild(el("rect", { x: 0, y: yy - rowH / 2, width: W, height: rowH, fill: "transparent" }))
      hover(hit, () => `<b>${LABELS[k]}</b><br>median ${Math.round(m).toLocaleString()} · range ${Math.round(vs[0]).toLocaleString()}–${Math.round(vs[vs.length - 1]).toLocaleString()} · ${vs.length} texts`)
    })
    mount("d-gpt2", "GPT-2 perplexity, every text (log scale)", svg,
      "One dot per text, the black tick at the median. The ranges do not overlap: the worst English paragraph reads better to GPT-2 than the best generated one, and the worst generated one better than any v1 paragraph. v1’s dots stack because it repeats itself.")
  }
  const hist = (id, title, label, caption, series, bins, binLabel) => {
    const W = 640, H = 200, left = 34, top = 14, base = 160
    const counts = series.map((s) => bins.map(([a, b]) => s.vs.filter((v) => v >= a && v < b).length / (s.share ? s.vs.length : 1)))
    const max = Math.max(...counts.flat())
    const bw = (W - left) / bins.length, sw = (bw * 0.8) / series.length
    const svg = svgOf(W, H, label)
    svg.appendChild(el("line", { x1: left, x2: W, y1: base, y2: base, stroke: "var(--ink)" }))
    bins.forEach((bin, i) => {
      series.forEach((s, j) => {
        const c = counts[j][i], h = ((base - top) * c) / max
        svg.appendChild(el("rect", { x: left + i * bw + bw * 0.1 + j * sw, y: base - h, width: Math.max(sw - 1, 1), height: h, rx: 1.5, fill: s.color, "fill-opacity": s.opacity || 1 }))
      })
      if (i % Math.ceil(bins.length / 10) === 0) svg.appendChild(el("text", { x: left + i * bw + bw / 2, y: base + 18, "text-anchor": "middle" }, binLabel(bin, true)))
      const hit = svg.appendChild(el("rect", { x: left + i * bw, y: 0, width: bw, height: H, fill: "transparent" }))
      hover(hit, () => `<b>${binLabel(bin)}</b><br>` + series.map((s, j) => `${s.name}: ${s.share ? pct(counts[j][i]) : counts[j][i]}`).join("<br>"))
    })
    svg.appendChild(el("text", { x: left, y: 8 }, series[0].share ? "share" : "texts"))
    mount(id, title, svg, caption)
  }

  Promise.all([fetch("/how/data.json").then((r) => r.json()), fetch("/how/figures.json").then((r) => r.json())]).then(([data, fig]) => {
    document.getElementById("n-v5").textContent = data.v5_live.n + data.v5_sophia.n
    bars("c-readable", "Sentences the verifier rates readable", "readable_share", pct,
      "Share of sentences scoring at or above the verifier’s readable threshold. v1 never ends a sentence, so its one sentence per paragraph fails.", data)
    bars("c-gpt2", "GPT-2 perplexity, median (lower reads more like English)", "gpt2_ppl_median",
      (v) => Math.round(v).toLocaleString(),
      "An independent check: GPT-2 never takes part in the search, so the generator cannot have been tuned against it.", data)
    bars("c-short", "Words of one or two letters", "short_share", pct,
      "Short words are the cheapest way to pay a letter debt. This generator uses more of them than v1, which is its worst result.", data)
    bars("c-sentence", "Words per sentence", "sentence_words", (v) => v.toFixed(1),
      "Free punctuation lets the generator end sentences; v1 wrote each paragraph as one run-on.", data)
    const cols = [["readable_share", "readable", pct], ["gpt2_ppl_median", "GPT-2 ppl", (v) => Math.round(v)],
      ["short_share", "≤2-letter words", pct], ["sentence_words", "words/sentence", (v) => v.toFixed(1)],
      ["letters", "letters", (v) => Math.round(v)], ["exact", "exact", pct], ["n", "n", (v) => v]]
    const t = document.createElement("table")
    t.innerHTML = `<tr><th></th>${cols.map((c) => `<th>${c[1]}</th>`).join("")}</tr>` +
      ORDER.filter((k) => data[k]).map((k) =>
        `<tr><td>${LABELS[k]}</td>${cols.map((c) => `<td>${data[k][c[0]] == null ? "–" : c[2](data[k][c[0]])}</td>`).join("")}</tr>`).join("")
    document.getElementById("table").appendChild(t)

    strip(fig, data)
    const sl = fig.sentence_lengths
    hist("d-sentence", "Sentence length in words", "Distribution of sentence lengths, this generator against English prose.",
      `Every sentence pooled: ${sl.v5.length.toLocaleString()} generated (blue) against ${sl.brown.length} from Brown (grey). 68% of generated sentences run ten words or fewer, against 37% of English; 2% reach twenty, against 33%.`,
      [{ name: "This generator", vs: sl.v5, share: true, color: "var(--signal)" }, { name: "Brown", vs: sl.brown, share: true, color: "var(--grey)" }],
      [...Array(30)].map((_, i) => (i < 29 ? [i + 1, i + 2] : [30, Infinity])),
      ([a, b], short) => (b === Infinity ? (short ? "30+" : "30 or more words") : short ? String(a) : `${a} word${a > 1 ? "s" : ""}`))
    const lv = fig.letters
    hist("d-letters", "Letters per paragraph", "Distribution of paragraph lengths in letters for both generator runs.",
      "Length is a target the search hits: paragraphs stop at or just past it, 300 letters live (dark), 600, 700 or 1,000 in the long runs (light).",
      [{ name: "300-letter target", vs: lv.v5_live, color: "var(--signal)" }, { name: "600–1,000 target", vs: lv.v5_sophia, color: "var(--signal)", opacity: 0.4 }],
      [...Array(32)].map((_, i) => [300 + i * 25, 325 + i * 25]),
      ([a, b], short) => (short ? String(a) : `${a}–${b - 1} letters`))

    /* ---- versions timeline ------------------------------------------ */
    ;(() => {
      const W = 640, H = 150, pad = 60, y = 40, vs = fig.versions
      const x = (i) => pad + (i * (W - 2 * pad)) / (vs.length - 1)
      const svg = svgOf(W, H, "Five versions of the generator, one key number each.")
      svg.appendChild(el("line", { x1: x(0), x2: x(vs.length - 1), y1: y, y2: y, stroke: "var(--grey)" }))
      vs.forEach(([v, n, what], i) => {
        const ours = i === vs.length - 1
        svg.appendChild(el("circle", { cx: x(i), cy: y, r: 6, fill: ours ? "var(--signal)" : "var(--paper)", stroke: ours ? "var(--signal)" : "var(--grey)", "stroke-width": 2 }))
        svg.appendChild(el("text", { x: x(i), y: y - 16, "text-anchor": "middle", class: "val" }, v))
        svg.appendChild(el("text", { x: x(i), y: y + 30, "text-anchor": "middle", class: "val" }, n))
        const ws = what.split(" "), cut = Math.ceil(ws.length / 2)
        ;[ws.slice(0, cut), ws.slice(cut)].forEach((line, j) =>
          svg.appendChild(el("text", { x: x(i), y: y + 48 + j * 15, "text-anchor": "middle" }, line.join(" "))))
        const hit = svg.appendChild(el("rect", { x: x(i) - 50, y: 0, width: 100, height: H, fill: "transparent" }))
        hover(hit, () => `<b>${v}</b><br>${n} ${what}`)
      })
      mount("timeline", null, svg, "One number per version. Each failed on a different axis, so no single measure fits all five.")
    })()

    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView()
  })
})()
