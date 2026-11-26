import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import type { MouseEvent } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import About from "./About"
import Tutorial from "./Tutorial"

/* THE POSTER, in one paragraph.
 *
 * The page breaks its own lines. Words are placed on a character grid and each
 * one is positioned absolutely at the cell it landed in, because the cell a word
 * lands in then depends only on the words between it and the mirror — so a
 * longer draft re-uses every position the shorter one had, and the text grows
 * outward at its two edges instead of reflowing. Letting the browser wrap
 * instead would have been fine for a finished palindrome and useless for one
 * still being written: the search grows the left half by PREPENDING, and
 * prepending into a flowing paragraph moves every word after it. The font is
 * monospace, so a cell is a fixed size and none of this needs measuring, beyond
 * one reading of the character advance at startup.
 *
 * The camera falls out of the same grid rather than being a mechanism of its
 * own. The mirror sits at a fixed cell — column c0, the middle of row 0 — and
 * the block is translated by exactly that point and scaled about it, so the
 * caret is nailed to the centre of the screen for the whole run and the scale is
 * the only thing that ever changes. An earlier version pinned the block by its
 * own bounding centre, which drifts off the mirror as the two halves take
 * different numbers of lines.
 *
 * Nothing sets will-change. Promoting the block to its own layer made the
 * browser rasterise it whole; at the starting scale that is far past the
 * maximum texture size, and the failed raster was the black/white flicker.
 */

/* The mirror is a property of the LETTERS, so the server cuts the word list
 * where the letters actually turn around. `center` is the word the mirror runs
 * through, empty when it falls in a gap, and `pivot` indexes into
 * `centerDisplay` — the middle letter itself when `pivotOdd`, otherwise the gap
 * in front of it. */
type Shape = {
  center: string
  centerDisplay: string
  pivot: number
  pivotOdd: boolean
  promptCenter: boolean
  left: string[]
  right: string[]
  letters: number
  words: number
}

type Result = Shape & {
  lm: number | null
  coherence: number
  seconds: number
  // v2 only. It places whole sentences mined from Wikipedia, so it has to say
  // which spans it quoted rather than composed.
  sentences?: { text: string; quoted: boolean }[]
  quoted?: string[]
  quotedCount?: number
  attribution?: { source: string; license: string; url: string; note: string }
  // v3 only. The poster draws letters, because the mirror is a property of the
  // characters and only a monospace grid shows that. `written` is the same
  // letters with the case and punctuation put back, which is what "read it"
  // should mean once there is a version that has any.
  written?: string
  yours?: boolean
}

/* Which generator this page talks to.
 *
 * This poster serves two: `/` is v1, which composes from single words, and
 * `/v2` places whole attested sentences. They share the frame contract, so the
 * poster below draws either one without knowing which it has — only the
 * endpoint and the credit line differ.
 *
 * `/v2` was `/dev` until v3 took that path. The route moved rather than the
 * page being deleted: it still works, and it is what v3 has to be compared
 * against. `IS_V2` is the old `IS_DEV` under a name that says what it selects.
 *
 * v3 differs in one way that matters here: it does not stream, because it does
 * not search — the whole composition arrives in one response. It is fed to the
 * same pen in frames anyway (`runV3` below), because the writing was never a
 * progress bar. It is how a reader sees that two halves are being built
 * outward from one mirror, and that is as true of a composition as of a
 * search. */
const PATH = typeof window !== "undefined"
  ? window.location.pathname.replace(/\/+$/, "") : ""
const IS_V2 = PATH === "/v2"
/* The root is the streaming search again, now on the v5 engine. v3 keeps
   `/v3`, and `/dev` because that URL has been shared. */
const IS_V3 = PATH === "/v3" || PATH === "/dev"
const API = IS_V2 ? "/api/v2/generate" : "/api/generate"

/* The length dial, which only v3 has: v1 and v2 stop when their budget runs
 * out, so their length is a result rather than a request.
 *
 * Logarithmic. The bank holds about 14,500 letters and every interesting choice
 * is in the first tenth of that — 40 to 400 letters is the difference between a
 * line and a paragraph; 13,000 to 14,500 is not a difference at all. A linear
 * track spends nine tenths of its travel on the range nobody wants. */
const LEN_MIN = 40
const LEN_STEPS = 1000
const fromSlider = (t: number, cap: number) =>
  Math.round(LEN_MIN * Math.pow(cap / LEN_MIN, t / LEN_STEPS))
const toSlider = (n: number, cap: number) =>
  Math.round(LEN_STEPS * Math.log(n / LEN_MIN) / Math.log(cap / LEN_MIN))

/* How the mocked stream is paced. Enough frames that the pen never jumps a
   whole line at once, few enough that a long composition does not take longer
   to draw than a real search takes to find one. */
const V3_FRAMES = 14
const V3_GAP = 300

/* The paragraph. Letter-level: the LETTERS mirror, at paragraph length.
 *
 * This used to serve a word-order palindrome, where the sentence sequence
 * mirrors and the letters do not — a different and much easier constraint,
 * since it pays nothing per letter. It was the default while the letter-level
 * paragraph was still a list of two-word fragments.
 *
 * What changed is the unit. The mirror cost of 3.3 bits per free letter forces
 * units to be short, and short units carry no subject, which is why every
 * attempt at a through-line failed. The canon's self-palindromic SENTENCES pay
 * the same cost and do carry subjects, so a paragraph nested out of the ones
 * that share one is about something and mirrors letter for letter. */
type Paragraph = {
  mode: string; text: string; words: number; bank?: string
  units?: string[]
  /* What it turned out to be about, and whether the visitor asked. A prompt
   * matching nothing still returns a paragraph — a request has to be answered
   * — and without these two fields that looked exactly like a prompt that
   * worked. `theme` is null precisely when the asking failed. */
  theme?: string | null; prompted?: boolean
  wordPalindrome: boolean; letterPalindrome: boolean; note: string
}

type Phase = "idle" | "searching" | "done" | "error"

const FONT_PX = 42          // one unit of the block's own coordinate system
const LINE = 1.42           // line height, in those units
const TOP_CHROME = 108      // space kept clear for the prompt
const CARD_CHROME = 260     // the open "What is this?" card, so it covers no words
const MAX_SCALE = 3         // scaling a large block UP is what strains the rasteriser
const EXPECT_LETTERS = 1200 // fallback if the search does not announce its size
const CHARS_PER_LETTER = 1.3  // what the spaces between words add
const DRAW_MS = 6000       // the whole paragraph is written in this long
/* The camera's two numbers, both taken from the deployed poster.
 *
 * OPEN_W is a WIDTH, not a multiple of the finished fit: the screen shows about
 * this many block-units across at the open, whatever the palindrome turns out to
 * be. That is what lets the start run off the edges — a fit near 0.45 against an
 * open of 3 is close to seven times too big — and a multiple of the fit, which
 * is what this was, never can be: it shrinks with the very thing it is supposed
 * to overflow, so 1.5x of a small poster is still a poster that fits.
 *
 * EASE is what keeps it there. (1-p)^2.2 is still 2.2x too big at the half way
 * mark and 1.3x at three quarters, so the text stays cut off for most of the
 * write and the frame only closes on it at the end. A linear ramp spends the
 * whole overflow in the opening seconds and then has nothing left to do. */
const OPEN_W = 462          // block-units across the screen at the open
const EASE = 2.2            // how long the frame stays behind the text

/* The caret is a sized BAR, not a "|" glyph.
 *
 * A glyph's ink sits off-centre inside its character box — side bearings on one
 * axis, baseline and leading on the other — so rotating the box makes the ink
 * orbit rather than spin. Drawing the bar as an element makes the box and the
 * ink the same rectangle, and transform-origin: 50% 50% is then exactly the
 * visual middle by construction, with nothing to measure and nothing to drift.
 *
 * Every cursor blinks, including while the search runs. A spinning caret said
 * "working" at a time when nothing else on the page moved; now the text itself
 * is being written word by word, and that says it better. */
const BAR_W = 0.085 * FONT_PX
const BAR_H = 0.82 * FONT_PX

/* Everything is measured FROM THE MIRROR, in half-letters.
 *
 * Counting from the start of the text instead would renumber every word on the
 * left each time the search prepends one, and those numbers are what the React
 * keys and the click pairing are built on. Distance from the mirror does not
 * move: the two halves always hold the same number of letters, so growth adds
 * the same amount to each side and leaves everything already placed where it
 * was. Halves are the unit because an odd palindrome mirrors through the MIDDLE
 * of a letter, and halves keep that an integer.
 *
 * A letter n places out therefore spans [2n, 2n+2), its midpoint is odd, and the
 * mirror image of any span is simply its negation. */
type Cell = {
  text: string       // what to print
  row: number        // 0 is the mirror's line; the left half runs negative
  col: number
  rel: number        // half-letters from the mirror to this cell's first letter
}

type Ext = { minRow: number; maxRow: number; minCol: number; maxCol: number }

type Grid = {
  left: Cell[]       // ordered FROM the mirror outward, which is drawing order
  right: Cell[]
  center: Cell | null
  caretCol: number   // in character units; .5 means between two cells
  leftEnd: number    // the cell the next left word would end in
  rightStart: number
  // Running extents: leftExt[i] covers the first i left words. The camera reads
  // these so it can frame what has been WRITTEN rather than what has arrived.
  leftExt: Ext[]
  rightExt: Ext[]
}

const stretch = (e: Ext, row: number, col: number, w: number): Ext => ({
  minRow: Math.min(e.minRow, row),
  maxRow: Math.max(e.maxRow, row),
  minCol: Math.min(e.minCol, col),
  maxCol: Math.max(e.maxCol, col + w),
})

const union = (a: Ext, b: Ext): Ext => ({
  minRow: Math.min(a.minRow, b.minRow),
  maxRow: Math.max(a.maxRow, b.maxRow),
  minCol: Math.min(a.minCol, b.minCol),
  maxCol: Math.max(a.maxCol, b.maxCol),
})

/* Every cursor on the page is this: a blinking bar, sized in the block's own
   units so it scales with the text and centred on the point it marks. */
function Bar({ x, y }: { x: number; y: number }) {
  return (
    <span
      aria-hidden="true"
      className="caret absolute bg-signal"
      style={{ left: x - BAR_W / 2, top: y - BAR_H / 2, width: BAR_W, height: BAR_H }}
    />
  )
}

/* Which letter each printed character is, so that a centre echoed with the
   visitor's own spacing still pairs up letter for letter. */
/** Letters only: a word's punctuation takes space on screen but no place in the mirror. */
const letterCount = (w: string) => (w.match(/[a-z]/gi) ?? []).length

function letterAt(text: string): number[] {
  const out: number[] = []
  let n = 0
  for (const ch of text) out.push(/[a-z]/i.test(ch) ? n++ : -1)
  return out
}

/* Placement, and the reason the poster holds still.
 *
 * Both halves are laid out FROM the mirror outward, so the nth word out is
 * placed by the n−1 words between it and the centre and by nothing else. Two
 * drafts that share a prefix therefore share every position in it, and the
 * longer one differs only by cells the shorter one never used. */
function place(s: Shape, cols: number): Grid {
  const c0 = Math.floor(cols / 2)
  const display = s.centerDisplay || s.center

  // A word at the mirror takes a space on each side; a bare gap IS the space.
  const cStart = display ? c0 - s.pivot : c0
  const leftEnd = display ? cStart - 2 : c0 - 1
  const rightStart = display ? cStart + display.length + 1 : c0 + 1

  /* The mirror cuts the centre word after its first P letters, wherever that
     falls: in the gap before letter P when the count is even, through letter P
     itself when it is odd. Its letters start at −2P half-letters (−2P−1 when the
     mirror sits on a letter) and run 2C on from there. */
  const before = letterCount(display.slice(0, s.pivot))
  const cRel = display ? -2 * before - (s.pivotOdd ? 1 : 0) : 0
  const cEnd = display ? cRel + 2 * letterCount(s.center) : 0
  const center: Cell | null = display
    ? { text: display, row: 0, col: cStart, rel: cRel }
    : null

  let base: Ext = { minRow: 0, maxRow: 0, minCol: c0, maxCol: c0 }
  if (center) base = stretch(base, 0, cStart, display.length)

  // Right half: appended, so it fills rightward and wraps downward.
  const right: Cell[] = []
  const rightExt: Ext[] = [base]
  let row = 0
  let col = rightStart
  let rel = cEnd
  for (const w of s.right) {
    if (col + w.length > cols) { row += 1; col = 0 }
    right.push({ text: w, row, col, rel })
    rightExt.push(stretch(rightExt[rightExt.length - 1], row, col, w.length))
    col += w.length + 1
    rel += 2 * letterCount(w)
  }

  // Left half: prepended, so it fills leftward and wraps UPWARD, each line
  // anchored at its right end. That anchor is what keeps a line still while
  // words are still being added to its left. Walking the array backwards puts
  // these in drawing order too — nearest the mirror first.
  const left: Cell[] = []
  const leftExt: Ext[] = [base]
  row = 0
  let end = leftEnd
  rel = cRel
  for (let i = s.left.length - 1; i >= 0; i--) {
    const w = s.left[i]
    if (end - w.length + 1 < 0) { row -= 1; end = cols - 1 }
    const at = Math.max(0, end - w.length + 1)
    rel -= 2 * letterCount(w)
    left.push({ text: w, row, col: at, rel })
    leftExt.push(stretch(leftExt[leftExt.length - 1], row, at, w.length))
    end = at - 2
  }

  // An odd letter count puts the mirror ON a letter, an even one in the gap
  // before it, and a bare word gap in the middle of its space.
  const caretCol = display ? (s.pivotOdd ? c0 + 0.5 : c0) : c0 + 0.5
  return { left, right, center, caretCol, leftEnd, rightStart, leftExt, rightExt }
}

/* How wide to make the grid. Picked once per run from the size the search
   expects to reach, so that the finished poster is roughly the shape of the
   screen — a squarer block is bounded by height and leaves the width empty. */
function pickCols(expectLetters: number, availW: number, availH: number, advance: number) {
  const chars = expectLetters * CHARS_PER_LETTER
  const shape = (availW / availH) * (LINE / advance)
  return Math.max(24, Math.min(200, Math.round(Math.sqrt(chars * shape))))
}

/* A word, with any part of it that is currently paired lit up. The pairing runs
   by letter, not by word, so the lit run usually starts or ends mid-word — which
   is the whole point of showing it. */
function Word({
  text, rel, pick, mirror, onPick, tone,
}: {
  text: string
  rel: number
  pick: [number, number] | null
  mirror: [number, number] | null
  onPick: () => void
  tone?: string
}) {
  const runs = useMemo(() => {
    const letters = letterAt(text)
    const paint = (i: number) => {
      if (letters[i] < 0) return ""
      const mid = rel + 2 * letters[i] + 1   // odd, so it never sits on a boundary
      if (pick && mid > pick[0] && mid < pick[1]) return "bg-signal text-paper"
      if (mirror && mid > mirror[0] && mid < mirror[1]) return "bg-signal-soft text-signal"
      return ""
    }
    const out: { text: string; cls: string }[] = []
    for (let i = 0; i < text.length; i++) {
      const cls = paint(i)
      const last = out[out.length - 1]
      if (last && last.cls === cls) last.text += text[i]
      else out.push({ text: text[i], cls })
    }
    return out
  }, [text, rel, pick, mirror])

  return (
    <span
      onClick={(e) => { e.stopPropagation(); onPick() }}
      className={`cursor-pointer whitespace-pre ${tone ?? ""}`}
    >
      {runs.map((r, i) => (r.cls ? <span key={i} className={r.cls}>{r.text}</span> : r.text))}
    </span>
  )
}

/* Log-shaped: 90% by 3s, the last 10% across 3–20s, then the run times out. */
const TIMEOUT_MS = 20000
const REPLAY = "/replay.json"   // paragraphs written earlier, for when the generator is down
const VERSIONS = ["v1", "v2", "v3", "v4"]   // retired generators, each with saved outputs in /archive
function progress(t: number) {
  return t <= 3
    ? 0.9 * Math.log1p(3 * t) / Math.log1p(9)
    : Math.min(1, 0.9 + 0.1 * Math.log1p(t - 3) / Math.log1p(17))
}

const WAIT_WORDS = [
  "loading", "thinking", "pondering", "mirroring", "balancing", "reversing", "reflecting",
  "untangling", "weighing", "spelling", "rereading", "juggling", "folding", "wondering",
  "composing", "settling", "rhyming", "scheming", "musing", "flipping",
]

/* Under the pens while the search runs. Every word ends on one fixed line and
   the dots start there, so nothing moves when the word changes. The line sits
   right of centre by half the average word, so the phrase reads as centred. */
function Waiting() {
  const [n, setN] = useState(0)
  const [shift, setShift] = useState(0)
  const words = useRef<HTMLSpanElement>(null)
  const dots = useRef<HTMLSpanElement>(null)
  useLayoutEffect(() => {
    const measure = () => {
      if (!words.current || !dots.current) return
      const ws = [...words.current.children].map((c) => c.getBoundingClientRect().width)
      setShift((ws.reduce((a, b) => a + b, 0) / ws.length - dots.current.getBoundingClientRect().width) / 2)
    }
    measure()
    document.fonts?.ready.then(measure)
  }, [])
  useEffect(() => {
    const t = window.setInterval(() => setN((k) => k + 1), 450)
    return () => window.clearInterval(t)
  }, [])
  const cur = Math.floor(n / 3) % WAIT_WORDS.length
  const prev = (cur + WAIT_WORDS.length - 1) % WAIT_WORDS.length
  return (
    <div aria-hidden="true" className="label absolute inset-x-0 top-1/2 mt-8 grid"
         style={{ gridTemplateColumns: `calc(50% + ${shift}px) 1fr` }}>
      <span ref={words} className="grid justify-items-end">
        {WAIT_WORDS.map((w, i) => (
          <span key={w} className="wait-word col-start-1 row-start-1"
                data-s={i === cur ? "on" : i === prev && n >= 3 ? "out" : "off"}>{w}</span>
        ))}
      </span>
      <span ref={dots} className="flex justify-self-start">
        {[0, 1, 2].map((i) => <span key={i} className="wait-dot" data-on={i <= n % 3}>.</span>)}
      </span>
    </div>
  )
}

function Progress() {
  const [p, setP] = useState(0)
  useEffect(() => {
    const t0 = performance.now()
    let raf = 0
    const step = () => { setP(progress((performance.now() - t0) / 1000)); raf = requestAnimationFrame(step) }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [])
  return <div role="progressbar" className="absolute left-0 top-0 z-40 h-[2px] bg-signal" style={{ width: `${p * 100}%` }} />
}

export default function App() {
  const [prompt, setPrompt] = useState("")
  const [para, setPara] = useState<Paragraph | null>(null)
  const [phase, setPhase] = useState<Phase>("idle")
  const [result, setResult] = useState<Result | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState("")
  const [cardOpen, setCardOpen] = useState(false)
  const [vp, setVp] = useState(() => ({ w: window.innerWidth, h: window.innerHeight }))
  const [advance, setAdvance] = useState(0.6)   // character width, in font sizes
  const [cols, setCols] = useState(0)
  const [endRows, setEndRows] = useState(1)   // rows the finished poster expects to take
  const [pick, setPick] = useState<[number, number] | null>(null)
  const [runId, setRunId] = useState(0)
  const [converged, setConverged] = useState(false)
  const deadline = useRef(0)
  const scroller = useRef<HTMLDivElement>(null)
  /* Saved outputs of the retired versions, replayed in a box under the essay. */
  const [archives, setArchives] = useState<Record<string, Result[]>>({})

  /* v3's two extra controls. `cap` is the bank's own ceiling, asked for rather
     than assumed, so the dial cannot promise a length the material cannot
     reach. The fallback is what the deployed bank holds today. */
  const [target, setTarget] = useState(1200)
  const [cap, setCap] = useState(14500)
  const v3Timers = useRef<number[]>([])

  /* Words are written one at a time, fast enough that the whole paragraph takes DRAW_MS. */
  const [drawn, setDrawn] = useState({ l: 0, r: 0 })
  const drawnRef = useRef({ l: 0, r: 0 })
  const pace = useRef({ lt: 0, rt: 0, t0: 0, acc: 1 })
  const tick = useRef(0)
  const finished = useRef(false)

  const esRef = useRef<EventSource | null>(null)
  const expectRef = useRef(EXPECT_LETTERS)

  const done = phase === "done"
  const searching = phase === "searching"

  const write = useCallback((lt: number, rt: number) => {
    pace.current.lt = lt; pace.current.rt = rt
    /* A timer, not requestAnimationFrame: rAF stops in a background tab, and the
       pen is what decides the run is over. The pen keeps its own clock, so frames
       arriving often retarget it without restarting it. */
    const step = () => {
      const p = pace.current
      const now = performance.now()
      p.acc += ((now - p.t0) * (p.lt + p.rt)) / DRAW_MS
      p.t0 = now
      let { l, r } = drawnRef.current
      while (p.acc >= 1 && (l < p.lt || r < p.rt)) {
        // The side further behind its own target goes next.
        if (r >= p.rt || (l < p.lt && l / p.lt <= r / p.rt)) l++
        else r++
        p.acc -= 1
      }
      if (l !== drawnRef.current.l || r !== drawnRef.current.r) {
        drawnRef.current = { l, r }
        setDrawn({ l, r })
      }
      if (l < p.lt || r < p.rt) { tick.current = window.setTimeout(step, 16); return }
      tick.current = 0
      p.acc = 1   // the next frame's first word goes down at once
      // The credits wait for the pen, not for the search.
      if (finished.current) setPhase("done")
    }
    if (!tick.current) { pace.current.t0 = performance.now(); step() }
  }, [])

  useEffect(() => () => window.clearTimeout(tick.current), [])

  /* The block is pinned by the mirror to the centre of the SCREEN, so it grows
     the same distance up as down. The room it has is therefore twice the smaller
     gap, not the space between the two bars: subtracting the bars' combined
     height would let the taller side's overhang run under the credits. */
  const availW = vp.w - 32
  const availH = Math.max(160, vp.h - 2 * (cardOpen ? CARD_CHROME : TOP_CHROME))

  /* The advance of a monospace cell, read once. Every position on the poster is
     arithmetic on this number, so it is the only thing measured — and measuring
     it before the webfont lands would be measuring the fallback. */
  useEffect(() => {
    let alive = true
    const read = () => {
      const el = document.createElement("span")
      el.style.cssText =
        `position:absolute;visibility:hidden;white-space:pre;font-size:${FONT_PX}px;` +
        `font-family:var(--font-mono)`
      el.textContent = "M".repeat(100)
      document.body.appendChild(el)
      const w = el.offsetWidth / 100 / FONT_PX
      el.remove()
      if (alive && w > 0.1) setAdvance(w)
    }
    read()
    document.fonts?.ready.then(read)
    return () => { alive = false }
  }, [])

  useEffect(() => {
    const onResize = () => setVp({ w: window.innerWidth, h: window.innerHeight })
    window.addEventListener("resize", onResize)
    window.addEventListener("orientationchange", onResize)
    return () => {
      window.removeEventListener("resize", onResize)
      window.removeEventListener("orientationchange", onResize)
    }
  }, [])

  const grid = useMemo(
    () => (result && cols ? place(result, cols) : null),
    [result, cols],
  )

  const charW = advance * FONT_PX
  const lineH = LINE * FONT_PX

  /* The anchor never moves: the mirror is at a fixed cell of the grid, so the
     point the camera pins is a constant and only the scale is a function of how
     much text there is. */
  const anchorX = grid ? grid.caretCol * charW : 0
  const anchorY = lineH / 2

  /* Characters written, cumulatively, in drawing order on each side. The
     pull-back reads this; see why it is not counting rows. */
  const written = useMemo(() => {
    if (!grid) return null
    const run = (cells: Cell[]) => {
      const out = [0]
      for (const c of cells) out.push(out[out.length - 1] + c.text.length + 1)
      return out
    }
    return { l: run(grid.left), r: run(grid.right) }
  }, [grid])

  /* THE PULL-BACK.
   *
   * Framed on where the text is GOING, not on how far it has got. Fitting the
   * drawn extent — the obvious thing, and what this did first — spends the whole
   * zoom in the opening seconds and then sits still: the first row fills almost
   * at once, width becomes the binding constraint, and from then on only the
   * slow accumulation of rows moves the camera at all. Measured over a seven
   * second write, that ran 3.00 → 0.49 inside the first three seconds and
   * 0.49 → 0.45 across the remaining four.
   *
   * So the end is a PARAMETER. The search announces the size it is aiming at
   * before it sends a word, so the finished frame — the full grid width by
   * `endRows` rows — is known from the first tick, and the camera can be aimed
   * at it from the start. Deriving the target from the frames instead made it
   * move under the camera: a frame landing one row deeper reset where the camera
   * thought it was going, so the schedule bent every time one arrived.
   *
   * Between the two, OPEN_W and EASE. The camera starts close enough that the
   * text does not fit on the screen and stays that way for most of the write:
   * the visitor reads the middle of a palindrome that is running off both edges,
   * and the frame only closes on it at the end. Words leaving the screen is the
   * point, not a thing to tune out — a camera that never lets anything off
   * screen is pinned to the fit, and the fit is the fast zoom.
   *
   * FONT_PX and the grid are untouched. The type is the same size in the block's
   * own coordinates from first word to last; only the camera moves. */
  const scale = useMemo(() => {
    if (!grid || !cols) return MAX_SCALE
    const e = union(grid.leftExt[Math.min(drawn.l, grid.leftExt.length - 1)],
                    grid.rightExt[Math.min(drawn.r, grid.rightExt.length - 1)])
    const halfW = Math.max(anchorX - e.minCol * charW, e.maxCol * charW - anchorX, 1)
    const halfH = Math.max(anchorY - e.minRow * lineH, (e.maxRow + 1) * lineH - anchorY, 1)

    // The finished poster: the full width of the grid, and however many rows it
    // turns out to need.
    /* When the whole answer has arrived up front (v5 sends no drafts), the end
       is no longer an estimate: aim at the real extent, so the camera makes one
       move instead of zooming to the estimate and then again to the truth. */
    const known = finished.current && written
    const last = union(grid.leftExt[grid.leftExt.length - 1], grid.rightExt[grid.rightExt.length - 1])
    const endHalfW = known
      ? Math.max(anchorX - last.minCol * charW, last.maxCol * charW - anchorX, 1)
      : Math.max(anchorX, cols * charW - anchorX)
    const endHalfH = known
      ? Math.max(anchorY - last.minRow * lineH, (last.maxRow + 1) * lineH - anchorY, 1)
      : (endRows * lineH) / 2
    const fitEnd = Math.min(availW / (2 * endHalfW), availH / (2 * endHalfH))
    // Known end: one eased move to the true fit (see the transition below).
    if (known) return Math.min(MAX_SCALE, fitEnd)

    /* How far through the writing, as a fraction — and it has to be a FRACTION.
       Counting the rows the drawn text spans is the obvious measure and it is an
       integer: it holds still while a row fills and jumps when the text wraps,
       so the camera sat, lurched, and sat again, once per row. The lurch was the
       whole zoom for that row arriving in one 100ms transition.

       Characters written is the same schedule made continuous. endRows is
       ceil(expected chars / cols), so `endRows * cols` is the character count
       this is a fraction OF, and the ratio matches what rows/endRows meant —
       it just advances with every word the pen puts down instead of every
       thirteenth. */
    const chars = written
      ? written.l[Math.min(drawn.l, written.l.length - 1)]
        + written.r[Math.min(drawn.r, written.r.length - 1)]
      : 0
    const p = Math.min(1, chars / Math.max(1, endRows * cols))

    /* The open is a width, so it is the SCREEN that decides how close the camera
       starts, not the palindrome. `max` with the fit is the floor that keeps a
       poster too small to overflow from being zoomed backwards. */
    const open = Math.min(MAX_SCALE, Math.max(fitEnd, availW / OPEN_W))
    const s = fitEnd + (open - fitEnd) * (1 - p) ** EASE

    /* The catch-up. Every scale above is a function of the size the search SAID
       it was aiming at, and a search that stops short of it — or runs past it —
       leaves `p` somewhere other than 1 with the frame still off the text. So
       the last move is the only one the real extent decides: once the pen has
       stopped, the camera goes to the fit of what actually got written. It is a
       longer, eased transition rather than the writing tick, because arriving is
       a camera move and should read as one. */
    const fits = Math.min(availW / (2 * halfW), availH / (2 * halfH))
    return Math.min(MAX_SCALE, done ? fits : s)
  }, [grid, written, drawn, cols, endRows, done, anchorX, anchorY, charW, lineH,
      availW, availH])

  /* Click a word and its mirror image lights up. Measured from the mirror, the
     image of a span is just its negation — and because the pairing runs by
     letter while the text is cut into words, the answer almost never lines up
     with a word. That is the thing worth seeing. */
  const mirror = useMemo((): [number, number] | null =>
    (pick ? [-pick[1], -pick[0]] : null), [pick])

  /* Where the two pens are: in the gap just beyond the last word written on each
     side, which is exactly where the next one goes. */
  const heads = useMemo(() => {
    if (!grid) return []
    const l = grid.left[drawn.l - 1]
    const r = grid.right[drawn.r - 1]
    return [
      l ? { row: l.row, col: l.col - 1 } : { row: 0, col: grid.leftEnd + 1 },
      r ? { row: r.row, col: r.col + r.text.length } : { row: 0, col: grid.rightStart - 1 },
    ]
  }, [grid, drawn])

  /* Back to an empty poster, ready for a new run. */
  const reset = useCallback(() => {
    esRef.current?.close()
    window.clearTimeout(deadline.current)
    window.clearTimeout(tick.current); tick.current = 0
    setResult(null); setError(""); setElapsed(0); setPick(null)
    setCols(0); setEndRows(1); expectRef.current = EXPECT_LETTERS
    setDrawn({ l: 0, r: 0 }); drawnRef.current = { l: 0, r: 0 }; pace.current.acc = 1
    finished.current = false
    setPhase("searching"); setRunId((n) => n + 1); setConverged(false); setCardOpen(false)
    // A selection left on the page paints every word the pen writes into it.
    window.getSelection?.()?.removeAllRanges()
  }, [])

  /* Write out a paragraph that is already whole: a cached one when the
     generator is down, or a saved one from an earlier version. */
  const play = useCallback((pick: Result) => {
    finished.current = true
    window.clearTimeout(tick.current); tick.current = 0
    setDrawn({ l: 0, r: 0 }); drawnRef.current = { l: 0, r: 0 }; pace.current.acc = 1
    const n = pickCols(pick.letters, availW, availH, advance)
    setCols(n)
    setEndRows(Math.max(1, Math.ceil((pick.letters * CHARS_PER_LETTER) / n)))
    setResult(pick); setConverged(true)
    write(pick.left.length, pick.right.length)
  }, [availW, availH, advance, write])

  const generate = useCallback(() => {
    reset()

    /* When the generator is down or too busy to answer in time, play one of the
       paragraphs it wrote earlier instead of showing an error. */
    const fail = (message: string) => {
      es.close()
      if (finished.current) return
      finished.current = true
      window.clearTimeout(tick.current); tick.current = 0
      fetch(REPLAY)
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((list: Result[]) => {
          const pick = list[Math.floor(Math.random() * list.length)]
          if (!pick) throw new Error()
          play(pick)
        })
        .catch(() => { setError(message); setResult(null); setPhase("error") })
    }

    deadline.current = window.setTimeout(() => fail("timed out"), TIMEOUT_MS)
    const es = new EventSource(`${API}?prompt=${encodeURIComponent(prompt)}&budget=16`)
    esRef.current = es
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.type === "status") { setElapsed(msg.elapsed ?? 0); return }
      if (msg.type === "plan") { expectRef.current = msg.expectLetters ?? EXPECT_LETTERS; return }
      if (msg.type === "error") { fail(msg.message); return }

      /* Drafts and the final answer are the same object at different lengths:
         the search only ever sends text that extends what it has already sent,
         so taking the newest one wholesale still leaves every word already on
         screen exactly where it was. */
      if (msg.type === "partial" || msg.type === "result") {
        const shape = msg as Shape
        // Both are fixed for the whole run: a regrid would rewrap, and a
        // rewrap moves words that have already been written.
        const n = cols || pickCols(expectRef.current, availW, availH, advance)
        setCols(n)
        setEndRows(Math.max(1, Math.ceil((expectRef.current * CHARS_PER_LETTER) / n)))
        setResult(msg.type === "result" ? (msg as Result)
          : { ...shape, lm: null, coherence: 0, seconds: 0 })
        if (msg.type === "result") { finished.current = true; setConverged(true); es.close() }
        // The frame is now the target; the pen walks to it. `done` is set when
        // the pen arrives, not here.
        write(shape.left.length, shape.right.length)
      }
    }
    es.onerror = () => fail("could not reach the generator")
  }, [prompt, availW, availH, advance, write, cols, reset, play])

  /* v3, which arrives whole and is written out anyway.
   *
   * The composition comes back in one response, so there is nothing to stream
   * and the honest thing would be to print it. But `drawn` counts words out
   * from the MIRROR in both directions — that is how `place` lays the grid, and
   * it is why the poster holds still while it fills — so walking the pen out
   * from 0 to the full halves shows a reader the one thing about a palindrome
   * worth showing: it is built from the middle, in two directions, and the two
   * sides are the same letters. The frames are synthesised rather than
   * received; nothing else about the drawing differs. */
  const runV3 = useCallback(() => {
    v3Timers.current.forEach(window.clearTimeout)
    v3Timers.current = []
    window.clearTimeout(tick.current); tick.current = 0
    setResult(null); setError(""); setElapsed(0); setPick(null)
    setCols(0); setEndRows(1)
    setDrawn({ l: 0, r: 0 }); drawnRef.current = { l: 0, r: 0 }; pace.current.acc = 1
    finished.current = false
    setPhase("searching")

    const q = new URLSearchParams({
      letters: String(target),
      seed: String(Date.now() % 1e9),
    })
    if (prompt.trim()) q.set("centre", prompt.trim())

    fetch(`/api/v3/composition?${q}`)
      .then(async (r) => {
        const body = await r.json().catch(() => ({}))
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`)
        return body
      })
      .then((c) => {
        const shape = c.shape as Shape
        // Fixed for the whole run, as in the streaming path: a regrid rewraps,
        // and a rewrap moves words already on screen.
        const n = pickCols(c.letters, availW, availH, advance)
        setCols(n)
        setEndRows(Math.max(1, Math.ceil((c.letters * CHARS_PER_LETTER) / n)))
        setResult({
          ...shape, lm: null, coherence: 0, seconds: 0,
          written: c.text, yours: c.centre_is_yours,
        })
        const L = shape.left.length
        const R = shape.right.length
        for (let i = 1; i <= V3_FRAMES; i++) {
          const k = i / V3_FRAMES
          v3Timers.current.push(window.setTimeout(() => {
            // `finished` before the last frame, so `done` lands when the pen
            // arrives rather than when the fetch did — same as the stream.
            if (i === V3_FRAMES) finished.current = true
            write(Math.round(L * k), Math.round(R * k))
          }, i * V3_GAP))
        }
      })
      .catch((e) => { setError(String(e.message || e)); setPhase("error") })
  }, [prompt, target, availW, availH, advance, write])

  useEffect(() => () => {
    esRef.current?.close()
    v3Timers.current.forEach(window.clearTimeout)
  }, [])

  /* A version whose file is missing gets no button. */
  useEffect(() => {
    if (IS_V3) return
    let alive = true
    for (const v of VERSIONS) {
      fetch(`/archive/${v}.json`)
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((list: Result[]) => {
          const ok = Array.isArray(list) ? list.filter((x) => x?.left && x?.right) : []
          if (alive && ok.length) setArchives((a) => ({ ...a, [v]: ok }))
        })
        .catch(() => { /* no file, no button */ })
    }
    return () => { alive = false }
  }, [])

  /* The dial cannot offer a length the bank cannot reach, so it asks. */
  useEffect(() => {
    if (!IS_V3) return
    fetch("/api/v3/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((h) => {
        const max = h?.capacity?.novel?.max_letters
        if (max) { setCap(max); setTarget((t) => Math.min(t, max)) }
      })
      .catch(() => { /* the fallback is the deployed bank's size */ })
  }, [])

  const loadParagraph = useCallback(() => {
    // The prompt steers the theme rather than filtering it: "devil" matches
    // two centres, and a filter would return two sentences and call that a
    // paragraph. Seven whole sentences is the length here — the word mode
    // wanted 18 because its units were three words each.
    fetch(`/api/v2/paragraph?sentences=7&prompt=${encodeURIComponent(prompt)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setPara)
      .catch(() => setPara(null))
  }, [prompt])

  /* One button, whichever generator is behind it. */
  const run = IS_V3 ? runV3 : generate

  const status =
    // v3 is composed, not searched, so there is no elapsed time to report and
    // reporting one would be a fiction. What it has instead is the length that
    // was asked for, which is a thing v1 and v2 do not have at all.
    IS_V3 && searching ? `${target.toLocaleString()} letters`
    : IS_V3 && done && result
      ? `${result.letters.toLocaleString()} letters · ${result.words} words`
        + (result.yours ? " · yours at the centre" : "")
    : searching ? `${elapsed.toFixed(1)}s${result ? ` · ${result.letters} letters` : ""}`
    : phase === "error" ? error
    : done && result ? `${result.letters} letters · ${result.words} words · ${result.seconds}s`
      + (result.quotedCount ? ` · ${result.quotedCount} quoted` : "")
    : ""

  return (
    <div ref={scroller} className="no-scrollbar h-full overflow-y-auto overflow-x-hidden bg-paper">
    <div className="relative h-full w-full overflow-hidden bg-paper" onClick={() => setPick(null)}>
      {!IS_V3 && searching && !converged && <Progress key={runId} />}

      {/* v2 does not write its best sentences — it finds them already written and
          builds the mirror around them. Wikipedia is CC BY-SA, so the credit is
          part of the page rather than a line in a README nobody opens. */}
      {/* Anchored above the credit line with real clearance: at 375px the panel
          previously ran under the footer, clipping the last sentence and pushing
          the caption off-screen, and the control sat hard on the viewport edge
          where a tap could not land. */}
      {IS_V2 && (
        <div className="pointer-events-auto absolute inset-x-0 bottom-14 z-30 px-4 sm:bottom-8 sm:px-6">
          <div className="mx-auto flex max-w-3xl flex-col items-center gap-3">
            <button
              onClick={loadParagraph}
              className="slab slab-press rounded-[3px] border-0 bg-ink px-4 py-2 font-display text-[10px] font-bold uppercase tracking-[.14em] text-paper hover:bg-signal">
              {para ? "New paragraph" : "Palindromic paragraph"}
            </button>
            {para && (
              <div className="flex w-full flex-col gap-2">
                {/* The prose scrolls; the caption never does, so the reader can
                    always see which constraint was satisfied. */}
                <div className="max-h-[34vh] overflow-y-auto text-left sm:max-h-[38vh]">
                  <p className="font-body text-[13px] leading-relaxed text-ink">{para.text}</p>
                </div>
                {/* Which constraint was satisfied, always. A reader told
                    "palindrome" should never have to guess whether the letters
                    mirror or only the sentence order does. */}
                <p className="shrink-0 font-display text-[10px] uppercase tracking-[.12em] text-ink/40">
                  {para.words} words
                  {para.letterPalindrome
                    ? " · letters mirror"
                    : " · sentence order only · letters do NOT mirror"}
                  {para.units ? ` · ${para.units.length} sentences` : ""}
                  {para.theme ? ` · on “${para.theme}”` : ""}
                  {para.bank ? ` · ${para.bank}` : ""}
                </p>
                {/* Said out loud, because a prompt that matched nothing still
                    returns a paragraph and would otherwise read as a hit. */}
                {para.prompted && !para.theme && (
                  <p className="shrink-0 font-display text-[10px] uppercase tracking-[.12em] text-signal">
                    nothing in the inventory matched — showing its own theme
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {IS_V2 && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-30 px-4 pb-2 text-center">
          <p className="font-display text-[10px] uppercase tracking-[.14em] text-ink/40">
            v2
            {result?.attribution
              ? <> · quoted sentences from {result.attribution.source},{" "}
                  <a className="pointer-events-auto text-signal hover:text-ink"
                     href={result.attribution.url}
                     target="_blank" rel="noreferrer license noopener">
                    {result.attribution.license}
                  </a>
                </>
              : <> · places whole attested sentences</>}
          </p>
        </div>
      )}

      <div className="absolute inset-0">
        {/* Before the first frame there is no grid to hang the caret off, and
            the caret alone IS the idle state — the prompt with nothing typed
            into it yet. It sits where the mirror will be. */}
        {!grid && (
          <span
            aria-hidden="true"
            className="caret absolute bg-signal"
            style={{
              left: "50%", top: "50%",
              width: BAR_W, height: BAR_H,
              marginLeft: -BAR_W / 2, marginTop: -BAR_H / 2,
            }}
          />
        )}
        {/* Waiting: the two pens leave the mirror in opposite directions, over
            and over, the way the two halves will grow out of it. */}
        {!grid && searching && [-1, 1].map((d) => (
          <span
            key={d}
            aria-hidden="true"
            className="pen absolute bg-signal"
            style={{
              left: "50%", top: "50%",
              width: BAR_W, height: BAR_H,
              marginLeft: -BAR_W / 2, marginTop: -BAR_H / 2,
              ["--d" as string]: d,
            }}
          />
        ))}
        {!grid && searching && <Waiting />}
        {!IS_V3 && !grid && phase !== "searching" && (
          <div className="absolute inset-x-0 top-1/2 mt-8 flex flex-col items-center gap-2 px-4"
               onClick={(e) => e.stopPropagation()}>
            {phase === "error" && <p className="label text-center text-signal">{error}</p>}
            <button onClick={run} className="shimmer label" aria-label="generate">
              {[..."generate"].map((ch, i) => <span key={i} style={{ animationDelay: `${i * 90}ms` }}>{ch}</span>)}
            </button>
          </div>
        )}

        {grid && result && (
          <div
            style={{
              position: "absolute",
              left: "50%",
              top: "50%",
              fontSize: FONT_PX,
              lineHeight: `${lineH}px`,
              transformOrigin: "0 0",
              transform: `scale(${scale}) translate(${-anchorX}px, ${-anchorY}px)`,
              /* Short while writing, because the scale moves a word at a time
                 rather than a frame at a time and this only takes the edge off
                 the steps. The arrival is the exception: it is one move, from
                 wherever the schedule had got to onto the true fit, and at 140ms
                 it snaps. */
              transition: done
                ? "transform 900ms cubic-bezier(.22,1,.36,1)"
                : finished.current
                  ? "transform 1400ms cubic-bezier(.22,1,.36,1)"
                  : "transform 140ms linear",
            }}
            className={`font-mono text-ink ${done ? "" : "select-none"}`}
          >
            {[...grid.left.slice(0, drawn.l), ...grid.right.slice(0, drawn.r)].map((c) => (
              <span
                key={c.rel}
                style={{ position: "absolute", left: c.col * charW, top: c.row * lineH }}
              >
                <Word
                  text={c.text} rel={c.rel} pick={pick} mirror={mirror}
                  onPick={() => setPick((p) =>
                    p && p[0] === c.rel ? null : [c.rel, c.rel + 2 * letterCount(c.text)])}
                />
              </span>
            ))}

            {/* The centre is its own mirror image, so clicking it lights the one
                word twice over — which is the shortest way to say what a
                palindrome's middle is. */}
            {grid.center && (() => {
              const c = grid.center
              const tone = result.promptCenter ? "text-signal" : ""
              const held = pick !== null && pick[0] === c.rel
              // The whole centre carries the handler, the pivot letter included:
              // that letter is drawn by itself so it can blink, and a click
              // landing on it would otherwise fall through and clear.
              const toggle = () =>
                setPick((p) => (p && p[0] === c.rel ? null : [c.rel, c.rel + 2 * letterCount(c.text)]))
              const take = (e: MouseEvent) => { e.stopPropagation(); toggle() }
              return (
                <span
                  onClick={take}
                  className="cursor-pointer whitespace-pre"
                  style={{ position: "absolute", left: c.col * charW, top: c.row * lineH }}
                >
                  {result.pivotOdd ? (
                    <>
                      <Word text={c.text.slice(0, result.pivot)} rel={c.rel}
                            pick={pick} mirror={mirror} tone={tone} onPick={toggle} />
                      <span className={held ? "bg-signal text-paper"
                                            : done ? "text-signal" : "pivot"}>
                        {c.text[result.pivot]}
                      </span>
                      <Word text={c.text.slice(result.pivot + 1)} rel={1}
                            pick={pick} mirror={mirror} tone={tone} onPick={toggle} />
                    </>
                  ) : (
                    <Word text={c.text} rel={c.rel} pick={pick} mirror={mirror}
                          tone={tone} onPick={toggle} />
                  )}
                </span>
              )
            })()}

            {/* THREE cursors, because three places are alive.
                The mirror never moves — it is where the text turns around. The
                other two are the pens: the palindrome is written outward from
                the middle in both directions at once, so there is a writing head
                at each edge, and each steps one word further out every time a
                word lands. They are what makes the pace visible. */}
            {!done && !result.pivotOdd && <Bar x={anchorX} y={anchorY} />}
            {!done && heads.map((h, i) => (
              <Bar key={i} x={(h.col + 0.5) * charW} y={h.row * lineH + lineH / 2} />
            ))}
          </div>
        )}
      </div>

      <div className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-paper via-paper to-transparent sm:h-28" />
      {done && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-paper via-paper to-transparent" />
      )}

      {IS_V3 && (
      <div className="pointer-events-none absolute inset-x-0 top-0 grid place-items-center px-4 pt-6 sm:pt-10">
        <div className="pointer-events-auto flex w-full max-w-[34rem] flex-col gap-2"
             onClick={(e) => e.stopPropagation()}>
          <label htmlFor="p" className="label min-h-[0.9rem] pl-1">{status}</label>
          <div className="flex gap-2">
            <div className="slab min-w-0 flex-1 bg-paper">
              <Input
                id="p"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") run() }}
                placeholder={IS_V3 ? "add your own palindrome" : "never odd or even"}
                disabled={searching}
                autoComplete="off" autoCapitalize="none" autoCorrect="off"
                spellCheck={false} enterKeyHint="go"
                className="h-11 border-0 bg-transparent font-mono text-base shadow-none focus-visible:ring-0"
              />
            </div>
            <Button
              onClick={run}
              disabled={searching}
              className="slab slab-press h-11 shrink-0 rounded-[3px] border-0 bg-ink px-4 font-display text-[11px] font-bold uppercase tracking-[.14em] text-paper hover:bg-signal sm:px-6 sm:text-xs sm:tracking-[.16em]"
            >
              {searching ? "…" : "Generate"}
            </Button>
          </div>
          {/* Length, v3 only, and it reads out through the status line above
              rather than carrying a label of its own — one line of chrome
              instead of two. Releasing the handle runs it; dragging does not,
              because a request per pixel would be a request per pixel. */}
          {IS_V3 && (
            <div className="flex items-center gap-3 pl-1">
              <input
                type="range"
                aria-label="length in letters"
                min={0} max={LEN_STEPS} step={1}
                value={toSlider(Math.min(target, cap), cap)}
                onChange={(e) => setTarget(fromSlider(Number(e.target.value), cap))}
                onMouseUp={run}
                onTouchEnd={run}
                onKeyUp={(e) => { if (e.key.startsWith("Arrow")) run() }}
                disabled={searching}
                className="range h-9 flex-1"
              />
              <span className="label w-16 shrink-0 text-right tabular-nums">
                {target.toLocaleString()}
              </span>
            </div>
          )}
        </div>
      </div>
      )}

      {done && result && <Tutorial onOpenChange={setCardOpen} />}
    </div>
    {!IS_V3 && (
      <About archives={archives} />
    )}
    </div>
  )
}
