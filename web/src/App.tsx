import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { MouseEvent } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

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
}

type Phase = "idle" | "searching" | "done" | "error"
type View = "poster" | "read"

const FONT_PX = 42          // one unit of the block's own coordinate system
const LINE = 1.42           // line height, in those units
const TOP_CHROME = 108      // space kept clear for the prompt
const BOTTOM_CHROME = 132   // and for the hint, the buttons and the credits
const MAX_SCALE = 3         // scaling a large block UP is what strains the rasteriser
const EXPECT_LETTERS = 1200 // fallback if the search does not announce its size
const CHARS_PER_LETTER = 1.3  // what the spaces between words add
const DEFAULT_SPAN = 750    // the search's own publishing interval
const MIN_SPAN = 200        // floor and ceiling on how fast the pen is allowed
const MAX_SPAN = 1600       // to run when frames bunch up or straggle
const OPEN = 1.5            // how much closer the camera opens than it ends

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

  /* The centre word is the palindromic core, so it sits symmetrically across the
     mirror: its C letters run from −C to +C in half-letters, whatever C is. */
  const core = s.center.length
  const center: Cell | null = display
    ? { text: display, row: 0, col: cStart, rel: -core }
    : null

  let base: Ext = { minRow: 0, maxRow: 0, minCol: c0, maxCol: c0 }
  if (center) base = stretch(base, 0, cStart, display.length)

  // Right half: appended, so it fills rightward and wraps downward.
  const right: Cell[] = []
  const rightExt: Ext[] = [base]
  let row = 0
  let col = rightStart
  let rel = core
  for (const w of s.right) {
    if (col + w.length > cols) { row += 1; col = 0 }
    right.push({ text: w, row, col, rel })
    rightExt.push(stretch(rightExt[rightExt.length - 1], row, col, w.length))
    col += w.length + 1
    rel += 2 * w.length
  }

  // Left half: prepended, so it fills leftward and wraps UPWARD, each line
  // anchored at its right end. That anchor is what keeps a line still while
  // words are still being added to its left. Walking the array backwards puts
  // these in drawing order too — nearest the mirror first.
  const left: Cell[] = []
  const leftExt: Ext[] = [base]
  row = 0
  let end = leftEnd
  rel = -core
  for (let i = s.left.length - 1; i >= 0; i--) {
    const w = s.left[i]
    if (end - w.length + 1 < 0) { row -= 1; end = cols - 1 }
    const at = Math.max(0, end - w.length + 1)
    rel -= 2 * w.length
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

export default function App() {
  const [prompt, setPrompt] = useState("")
  const [phase, setPhase] = useState<Phase>("idle")
  const [result, setResult] = useState<Result | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState("")
  const [view, setView] = useState<View>("poster")
  const [copied, setCopied] = useState(false)
  const [vp, setVp] = useState(() => ({ w: window.innerWidth, h: window.innerHeight }))
  const [advance, setAdvance] = useState(0.6)   // character width, in font sizes
  const [cols, setCols] = useState(0)
  const [endRows, setEndRows] = useState(1)   // rows the finished poster expects to take
  const [pick, setPick] = useState<[number, number] | null>(null)

  /* Words are written one at a time, spread evenly across the gap the frames
     arrive on. The search publishes in chunks of twenty-odd words because that
     is how a beam search finds them, but nobody reads in chunks of twenty-odd
     words: paced out over the second it took to find them, the same text reads
     as writing rather than as a slideshow. The clock is the gap between the last
     two frames, so the pen keeps up with whatever the box is managing. */
  const [drawn, setDrawn] = useState({ l: 0, r: 0 })
  const drawnRef = useRef({ l: 0, r: 0 })
  const pace = useRef({ l0: 0, r0: 0, lt: 0, rt: 0, t0: 0, span: DEFAULT_SPAN })
  const tick = useRef(0)
  const lastFrame = useRef(0)
  const finished = useRef(false)

  const esRef = useRef<EventSource | null>(null)
  const expectRef = useRef(EXPECT_LETTERS)

  const done = phase === "done"
  const searching = phase === "searching"

  const write = useCallback((lt: number, rt: number) => {
    const now = performance.now()
    const gap = lastFrame.current ? now - lastFrame.current : DEFAULT_SPAN
    lastFrame.current = now
    pace.current = {
      l0: drawnRef.current.l, r0: drawnRef.current.r, lt, rt, t0: now,
      // A frame that arrives late must not make the pen crawl for a second and
      // a half, and one that arrives on top of the last must not make it bolt.
      span: Math.min(MAX_SPAN, Math.max(MIN_SPAN, gap)),
    }
    /* A timer, not requestAnimationFrame. rAF does not fire in a background tab,
       and the pen is what decides the run is over — so a visitor who looked away
       mid-search came back to a page still saying "searching", with the button
       disabled and nothing left that would ever finish it. A timer keeps firing
       (slowly) when hidden, and since every position is computed from the clock
       rather than accumulated per tick, a coarse tick rate costs smoothness and
       nothing else. Always clear before scheduling: exactly one pen. */
    const step = () => {
      const p = pace.current
      const k = Math.min(1, (performance.now() - p.t0) / p.span)
      const l = p.l0 + Math.round((p.lt - p.l0) * k)
      const r = p.r0 + Math.round((p.rt - p.r0) * k)
      if (l !== drawnRef.current.l || r !== drawnRef.current.r) {
        drawnRef.current = { l, r }
        setDrawn({ l, r })
      }
      if (k < 1) { tick.current = window.setTimeout(step, 16); return }
      tick.current = 0
      // The credits wait for the pen, not for the search.
      if (finished.current) setPhase("done")
    }
    window.clearTimeout(tick.current)
    step()
  }, [])

  useEffect(() => () => window.clearTimeout(tick.current), [])

  /* The block is pinned by the mirror to the centre of the SCREEN, so it grows
     the same distance up as down. The room it has is therefore twice the smaller
     gap, not the space between the two bars: subtracting the bars' combined
     height would let the taller side's overhang run under the credits. */
  const availW = vp.w - 32
  const availH = Math.max(160, vp.h - 2 * Math.max(TOP_CHROME, BOTTOM_CHROME))

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
   * So the schedule follows the one quantity that grows the whole way through:
   * rows. The camera opens OPEN times closer than the finished poster needs and
   * closes that gap in step with the writing head, which makes the apparent type
   * size drift gently and evenly instead of collapsing at the start.
   *
   * What this costs is that early rows run past the edges — the text is wider
   * than the frame until the frame catches up. That is not a bug to tune out; it
   * is what a slower zoom IS. A camera that never lets anything off screen is
   * pinned to the fit, and the fit is the fast zoom.
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
    // turns out to need. Taking the max with what is already drawn is what
    // guarantees the last frame fits even when the search overshoots its guess.
    /* The end is a PARAMETER, not something to be read off the stream. The
       search announces the size it is aiming at before it sends a word, so the
       finished frame — the full grid width by `endRows` rows — is known from the
       first tick, and the whole pull-back is a straight line from OPEN times
       that to exactly that. Deriving the target from the frames instead made it
       move under the camera: a frame landing one row deeper reset where the
       camera thought it was going, so the schedule bent every time one arrived. */
    const endHalfW = Math.max(anchorX, cols * charW - anchorX)
    const endHalfH = (endRows * lineH) / 2
    const fitEnd = Math.min(availW / (2 * endHalfW), availH / (2 * endHalfH))

    const rows = e.maxRow - e.minRow + 1
    const p = Math.min(1, rows / endRows)
    const s = fitEnd * (OPEN + (1 - OPEN) * p)

    // The one place the real text still gets a vote: a search that overshoots
    // its own estimate must not leave the finished poster hanging off the edges.
    const fits = Math.min(availW / (2 * halfW), availH / (2 * halfH))
    return Math.min(MAX_SCALE, done ? Math.min(s, fits) : s)
  }, [grid, drawn, cols, endRows, done, anchorX, anchorY, charW, lineH, availW, availH])

  const fullText = useMemo(() => {
    if (!result) return ""
    return [...result.left, result.centerDisplay || result.center, ...result.right]
      .filter(Boolean).join(" ")
  }, [result])

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

  const generate = useCallback(() => {
    esRef.current?.close()
    window.clearTimeout(tick.current); tick.current = 0
    setResult(null); setError(""); setElapsed(0); setCopied(false); setPick(null)
    setCols(0); setEndRows(1); expectRef.current = EXPECT_LETTERS
    setDrawn({ l: 0, r: 0 }); drawnRef.current = { l: 0, r: 0 }
    lastFrame.current = performance.now(); finished.current = false
    setView("poster"); setPhase("searching")

    const es = new EventSource(`/api/generate?prompt=${encodeURIComponent(prompt)}&budget=16`)
    esRef.current = es
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.type === "status") { setElapsed(msg.elapsed ?? 0); return }
      if (msg.type === "plan") { expectRef.current = msg.expectLetters ?? EXPECT_LETTERS; return }
      if (msg.type === "error") { setError(msg.message); setPhase("error"); es.close(); return }

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
        if (msg.type === "result") { finished.current = true; es.close() }
        // The frame is now the target; the pen walks to it. `done` is set when
        // the pen arrives, not here.
        write(shape.left.length, shape.right.length)
      }
    }
    es.onerror = () => {
      setPhase((cur) => {
        if (cur === "searching") { setError("could not reach the generator"); return "error" }
        return cur
      })
      es.close()
    }
  }, [prompt, availW, availH, advance, write, cols])

  useEffect(() => () => esRef.current?.close(), [])

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(fullText)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch { /* clipboard unavailable; read view is selectable */ }
  }, [fullText])

  const status =
    searching ? `${elapsed.toFixed(1)}s${result ? ` · ${result.letters} letters` : ""}`
    : phase === "error" ? error
    : done && result ? `${result.letters} letters · ${result.words} words · ${result.seconds}s`
    : ""

  return (
    <div className="relative h-full w-full overflow-hidden bg-paper" onClick={() => setPick(null)}>

      <div className={`absolute inset-0 ${view === "read" ? "hidden" : ""}`}>
        {/* display:none above, not visibility:hidden — the words used to set
            their own inline visibility, and an inline `visible` on a child
            overrides a hidden ancestor, so the poster kept drawing itself
            straight through the read view. */}
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
              // Short, because the scale now moves a word at a time rather than
              // a frame at a time; this only takes the edge off the steps.
              transition: "transform 100ms linear",
            }}
            className="font-mono text-ink"
          >
            {[...grid.left.slice(0, drawn.l), ...grid.right.slice(0, drawn.r)].map((c) => (
              <span
                key={c.rel}
                style={{ position: "absolute", left: c.col * charW, top: c.row * lineH }}
              >
                <Word
                  text={c.text} rel={c.rel} pick={pick} mirror={mirror}
                  onPick={() => setPick((p) =>
                    p && p[0] === c.rel ? null : [c.rel, c.rel + 2 * c.text.length])}
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
                setPick((p) => (p && p[0] === c.rel ? null : [c.rel, -c.rel]))
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

      {view === "read" && result && (
        <div className="absolute inset-0 overflow-y-auto overscroll-contain px-5 pb-32 pt-28">
          <p className="mx-auto max-w-[62ch] break-words text-left text-[15px] leading-[1.85] text-ink sm:text-base">
            {result.left.join(" ")}
            {/* A cursor belongs on the poster, not in the prose; here the mirror
                shows only when it is the visitor's own phrase. */}
            {result.centerDisplay
              ? <> <span className={result.promptCenter ? "text-signal" : ""}>{result.centerDisplay}</span> </>
              : " "}
            {result.right.join(" ")}
          </p>
        </div>
      )}

      <div className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-paper via-paper to-transparent sm:h-28" />
      {done && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-paper via-paper to-transparent" />
      )}

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
                onKeyDown={(e) => { if (e.key === "Enter") generate() }}
                placeholder="never odd or even"
                disabled={searching}
                autoComplete="off" autoCapitalize="none" autoCorrect="off"
                spellCheck={false} enterKeyHint="go"
                className="h-11 border-0 bg-transparent font-mono text-base shadow-none focus-visible:ring-0"
              />
            </div>
            <Button
              onClick={generate}
              disabled={searching}
              className="slab slab-press h-11 shrink-0 rounded-[3px] border-0 bg-ink px-4 font-display text-[11px] font-bold uppercase tracking-[.14em] text-paper hover:bg-signal sm:px-6 sm:text-xs sm:tracking-[.16em]"
            >
              {searching ? "…" : "Generate"}
            </Button>
          </div>
        </div>
      </div>

      {done && result && (
        <div className="absolute inset-x-0 bottom-0 grid place-items-center px-4 pb-5"
             onClick={(e) => e.stopPropagation()}>
          <div className="flex w-full max-w-[34rem] flex-col items-center gap-3">
            <p className="label text-center">
              {view === "read" ? " "
                : pick ? "the same letters, read the other way"
                : "click a word to find its mirror"}
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <button
                onClick={() => setView(view === "read" ? "poster" : "read")}
                className="slab slab-press h-9 rounded-[3px] bg-paper px-4 font-display text-[11px] font-bold uppercase tracking-[.14em] text-ink"
              >
                {view === "read" ? "Poster" : "Read it"}
              </button>
              <button
                onClick={copy}
                className="slab slab-press h-9 rounded-[3px] bg-paper px-4 font-display text-[11px] font-bold uppercase tracking-[.14em] text-ink"
              >
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <p className="label text-center leading-relaxed">
              reads the same backwards · gpt-2 on a mac mini ·{" "}
              <a href="https://ericspencer.us" target="_blank" rel="noopener noreferrer"
                 className="underline decoration-from-font underline-offset-2 hover:text-signal">
                ericspencer.us
              </a>{" "}
              · after{" "}
              <a href="https://norvig.com/pal-alg.html" target="_blank" rel="noopener noreferrer"
                 className="underline decoration-from-font underline-offset-2 hover:text-signal">
                norvig
              </a>{" "}
              &amp; hoey
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
