/* /how: the debt walkthrough, two diagrams, and the measurement charts. */
(() => {
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
  }
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
    svg.appendChild(el("text", { x: x0, y: base + 24, class: "val" }, "no it a no it a"))
    svg.appendChild(el("text", { x: x1, y: base + 24, "text-anchor": "end", class: "val" }, "Ere he met a hill."))
    svg.appendChild(el("text", { x: x0, y: base + 46 }, "repetitive garbage"))
    svg.appendChild(el("text", { x: x1, y: base + 46, "text-anchor": "end" }, "readable English"))
    svg.appendChild(el("text", { x: x0, y: 18 }, "probability readable"))
    const fig = document.getElementById("spectrum")
    fig.appendChild(svg)
    fig.appendChild(Object.assign(document.createElement("figcaption"), {
      textContent: "Schematic. The verifier turns counts into a probability that a sentence is readable; the beam pays for sentences on the right and punishes those on the left." }))
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

  fetch("/how/data.json").then((r) => r.json()).then((data) => {
    document.getElementById("n-v5").textContent = data.v5_live.n + data.v5_sophia.n
    bars("c-readable", "Sentences the verifier rates readable", "readable_share", pct,
      "Share of sentences scoring at or above the verifier’s readable threshold. v1 never ends a sentence, so its one sentence per paragraph fails.", data)
    bars("c-gpt2", "GPT-2 perplexity, median (lower reads more like English)", "gpt2_ppl_median",
      (v) => Math.round(v).toLocaleString(),
      "An independent check: GPT-2 never takes part in the search, so the generator cannot have been tuned against it.", data)
    bars("c-short", "Words of one or two letters", "short_share", pct,
      "The price of exactness. Short words are the cheapest way to pay a letter debt, and every palindrome method leans on them.", data)
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
  })
})()
