import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

/* THE CAMERA, in one paragraph.
 *
 * The block of text is pinned by its own centre to the middle of the screen
 * with translate(-50%,-50%), and scaled about that same centre. That is the
 * whole mechanism: no anchor to compute, no margin to offset, no coordinate
 * space to convert between. An earlier version pinned a chosen point by hand
 * with transform-origin plus negative margins and measured the revealed text
 * inside that transformed space on every tick; every arithmetic slip in it
 * showed up as the block drifting sideways.
 *
 * The scale is a pure function of how much text has been revealed, so it cannot
 * chase, lag or oscillate. The block's layout is fixed the moment the result
 * arrives, so its final scale is known immediately and the zoom just walks
 * towards it along a curve — fast at first, then easing off, which is what
 * standing too close to a painting and stepping back feels like.
 *
 * Nothing sets will-change. Promoting the block to its own layer made the
 * browser rasterise it whole; at the starting scale that is far past the
 * maximum texture size, and the failed raster was the black/white flicker.
 */

/* The mirror is a property of the LETTERS, so the server cuts the word list
 * where the letters actually turn around. That cut lands inside a word about as
 * often as it lands in a gap, which is why `center` and `pivot` exist: `center`
 * is the word the mirror runs through (empty when it falls in a gap) and `pivot`
 * indexes into `centerDisplay` — the middle letter itself when `pivotOdd`,
 * otherwise the gap in front of it. Splitting the list down the middle by word
 * count instead put the caret a word or two off whenever the two halves were
 * worded differently, which is most of the time. */
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

type Phase = "idle" | "searching" | "revealing" | "done" | "error"
type View = "poster" | "read"

const FIRST_STEP_MS = 150   // pace of the reveal, per word-pair
const LAST_STEP_MS = 26
const FONT_PX = 42          // one unit of the block's own coordinate system
const TOP_CHROME = 108      // space kept clear for the prompt
const BOTTOM_CHROME = 116   // and for the credits
const ZOOM_EASE = 2.2       // >1 front-loads the pull-back

/* The caret is a sized BAR, not a "|" glyph.
 *
 * A glyph's ink sits off-centre inside its character box — side bearings on one
 * axis, baseline and leading on the other — so rotating the box makes the ink
 * orbit rather than spin. Drawing the bar as an element makes the box and the
 * ink the same rectangle, and transform-origin: 50% 50% is then exactly the
 * visual middle by construction, with nothing to measure and nothing to drift.
 * It is sized in em so it still scales with the surrounding text.
 */
const BAR = "inline-block w-[0.085em] h-[0.82em] bg-signal align-[-0.09em]"

/* The spin rhythm itself lives in the caret-spin keyframes in index.css. */

function Caret({ spinning }: { spinning: boolean }) {
  return <span className={`${BAR} ${spinning ? "caret-spin" : "caret"}`} aria-hidden="true" />
}

/* The mirror mark, wherever the mirror happens to fall.
 *
 * A gap between words takes the caret. A letter — which is where an odd letter
 * count always puts it, there being no gap to occupy — takes the same cursor one
 * glyph wide: the letter inverts and blinks rather than being replaced, so the
 * word stays readable. Once the search is finished the mark stops moving but
 * still stands out, because a poster should show where it turns around without
 * anything on it flashing.
 */
function Mirror({ r, spinning, done }: { r: Shape | null; spinning: boolean; done: boolean }) {
  const caret = done ? null : <Caret spinning={spinning} />
  const text = r && (r.centerDisplay || r.center)
  if (!r || !text) return <span className="whitespace-nowrap">{caret}</span>

  const tone = r.promptCenter ? "text-signal" : ""
  if (r.pivotOdd) {
    return (
      <span className={`whitespace-nowrap ${tone}`}>
        {text.slice(0, r.pivot)}
        <span className={done ? "text-signal" : "pivot"}>{text[r.pivot]}</span>
        {text.slice(r.pivot + 1)}
      </span>
    )
  }
  return (
    <span className={`whitespace-nowrap ${tone}`}>
      {text.slice(0, r.pivot)}{caret}{text.slice(r.pivot)}
    </span>
  )
}

export default function App() {
  const [prompt, setPrompt] = useState("")
  const [phase, setPhase] = useState<Phase>("idle")
  const [result, setResult] = useState<Result | null>(null)
  const [shown, setShown] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState("")
  const [view, setView] = useState<View>("poster")
  const [copied, setCopied] = useState(false)
  const [vp, setVp] = useState(() => ({ w: window.innerWidth, h: window.innerHeight }))
  const [fit, setFit] = useState(1)

  const blockRef = useRef<HTMLDivElement>(null)
  const esRef = useRef<EventSource | null>(null)
  const drafted = useRef(0)

  const done = phase === "done"
  const searching = phase === "searching"
  const total = result ? Math.max(result.left.length, result.right.length) : 0
  /* A draft is shown whole: the search has already found all of it, and holding
     any of it back would be an animation pretending to be progress. Only the
     reveal itself withholds words — an error leaves the last draft standing
     rather than emptying the screen under the message. */
  const revealed = phase === "revealing" || done ? shown : total

  /* Block width is chosen so the finished text has roughly the screen's aspect
     ratio, otherwise a squarer block is bounded by height and leaves half the
     width empty. Monospace advance ~0.6em, line height 1.42em. */
  const worldWidth = useMemo(() => {
    if (!result) return 560
    const availW = vp.w - 32
    const availH = Math.max(160, vp.h - TOP_CHROME - BOTTOM_CHROME)
    const chars = result.letters + result.words
    const w = FONT_PX * Math.sqrt(0.6 * 1.42 * (availW / availH) * chars)
    return Math.min(6000, Math.max(560, w))
  }, [result, vp])

  /* Measured once per layout, never per revealed word. offsetWidth/Height are
     layout values, so the transform we set does not feed back into them. */
  /* Nothing to measure while the poster is display:none — offsetWidth would read
     0 and blow the fit up — so the read view re-measures on the way back. */
  useLayoutEffect(() => {
    const el = blockRef.current
    if (!el || view === "read") return
    const availW = vp.w - 32
    const availH = Math.max(160, vp.h - TOP_CHROME - BOTTOM_CHROME)
    setFit(Math.min(availW / Math.max(1, el.offsetWidth), availH / Math.max(1, el.offsetHeight)))
  }, [result, worldWidth, vp, view])

  useEffect(() => {
    const onResize = () => setVp({ w: window.innerWidth, h: window.innerHeight })
    window.addEventListener("resize", onResize)
    window.addEventListener("orientationchange", onResize)
    return () => {
      window.removeEventListener("resize", onResize)
      window.removeEventListener("orientationchange", onResize)
    }
  }, [])

  /* Start close enough that a couple of words fill the screen, and never above
     3 — scaling a large block up is what strains the rasteriser. */
  const startScale = useMemo(
    () => Math.min(3, Math.max(fit, (vp.w - 32) / (11 * FONT_PX))),
    [fit, vp.w],
  )

  const progress = total > 0 ? Math.min(1, shown / total) : 0
  /* Drafts do their own zooming: each one is longer than the last, so fitting
     every draft to the screen pulls the camera back exactly as fast as the
     search finds text. Capped at the same ceiling as the reveal, because it is
     scaling a block UP that strains the rasteriser. */
  const scale = !result ? startScale
    : searching ? Math.min(fit, startScale)
    : fit + (startScale - fit) * Math.pow(1 - progress, ZOOM_EASE)

  useEffect(() => {
    if (phase !== "revealing" || !result) return
    if (shown >= total) { setPhase("done"); return }
    const p = total > 1 ? shown / (total - 1) : 1
    const delay = FIRST_STEP_MS + (LAST_STEP_MS - FIRST_STEP_MS) * Math.sqrt(p)
    const t = window.setTimeout(() => setShown((n) => n + 1), delay)
    return () => clearTimeout(t)
  }, [phase, shown, total, result])

  const fullText = useMemo(() => {
    if (!result) return ""
    return [...result.left, result.centerDisplay || result.center, ...result.right]
      .filter(Boolean).join(" ")
  }, [result])

  const generate = useCallback(() => {
    esRef.current?.close()
    setResult(null); setShown(0); setError(""); setElapsed(0); setCopied(false)
    setView("poster"); setPhase("searching"); drafted.current = 0

    const es = new EventSource(`/api/generate?prompt=${encodeURIComponent(prompt)}&budget=16`)
    esRef.current = es
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.type === "status") { setElapsed(msg.elapsed ?? 0); return }
      if (msg.type === "error") { setError(msg.message); setPhase("error"); es.close(); return }
      /* Every closure the search improves on arrives here, so the page shows the
         palindrome growing for the ten seconds it used to spend showing a
         spinning caret and nothing else. */
      if (msg.type === "partial") {
        const draft = msg as Shape
        drafted.current = Math.max(draft.left.length, draft.right.length)
        setResult({ ...draft, lm: null, coherence: 0, seconds: 0 })
        return
      }
      if (msg.type === "result") {
        const r = msg as Result
        setResult(r)
        // Pick the reveal up where the drafts left off. Restarting from the
        // centre would replay ground the visitor has already watched, and the
        // final text is rarely longer than the last draft by much.
        setShown(Math.min(drafted.current, Math.max(r.left.length, r.right.length)))
        setPhase("revealing")
        es.close()
      }
    }
    es.onerror = () => {
      setPhase((cur) => {
        if (cur === "searching") { setError("could not reach the generator"); return "error" }
        return cur
      })
      es.close()
    }
  }, [prompt])

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
    : phase === "revealing" ? `${Math.min(shown * 2, result?.words ?? 0)} of ${result?.words ?? 0} words`
    : phase === "error" ? error
    : done && result ? `${result.letters} letters · ${result.words} words · ${result.seconds}s`
    : ""

  return (
    <div className="relative h-full w-full overflow-hidden bg-paper">

      {/* Camera: the block is pinned by its own centre to the middle of the
          screen and scaled about that same centre.

          translate(-50%,-50%) is doing the centring, NOT flex or grid. The
          block's LAYOUT width is the full unscaled width — wider than the
          viewport — and transforms do not change layout size, so grid alignment
          sees an oversized item, clamps it to the start edge, and the centre it
          then scales about sits off-screen. That was the drift to the right. */}
      {/* display:none, not visibility:hidden. The revealed words set their own
          inline visibility, and an inline `visible` on a child overrides a
          hidden ancestor — so the poster kept drawing itself straight through
          the read view, one state on top of the other. */}
      <div className={`absolute inset-0 ${view === "read" ? "hidden" : ""}`}>
        <div
          ref={blockRef}
          style={{
            position: "absolute",
            left: "50%",
            top: "50%",
            width: worldWidth,
            fontSize: FONT_PX,
            lineHeight: 1.42,
            transform: `translate(-50%, -50%) scale(${scale})`,
            // Smooths the step between revealed words. Linear, because the
            // shape of the zoom already lives in the curve above.
            transition: "transform 140ms linear",
          }}
          className="text-center text-ink"
        >
          {result?.left.map((w, i) => (
            <span key={`l${i}`} style={{ visibility: i >= result.left.length - revealed ? "visible" : "hidden" }}>
              {w}{" "}
            </span>
          ))}

          <Mirror r={result} spinning={searching} done={done} />

          {result?.right.map((w, i) => (
            <span key={`r${i}`} style={{ visibility: i < revealed ? "visible" : "hidden" }}>
              {" "}{w}
            </span>
          ))}
        </div>
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
        <div className="pointer-events-auto flex w-full max-w-[34rem] flex-col gap-2">
          <label htmlFor="p" className="label min-h-[0.9rem] pl-1">{status}</label>
          <div className="flex gap-2">
            <div className="slab min-w-0 flex-1 bg-paper">
              <Input
                id="p"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") generate() }}
                placeholder="never odd or even"
                disabled={phase === "searching"}
                autoComplete="off" autoCapitalize="none" autoCorrect="off"
                spellCheck={false} enterKeyHint="go"
                className="h-11 border-0 bg-transparent font-mono text-base shadow-none focus-visible:ring-0"
              />
            </div>
            <Button
              onClick={generate}
              disabled={phase === "searching"}
              className="slab slab-press h-11 shrink-0 rounded-[3px] border-0 bg-ink px-4 font-display text-[11px] font-bold uppercase tracking-[.14em] text-paper hover:bg-signal sm:px-6 sm:text-xs sm:tracking-[.16em]"
            >
              {phase === "searching" ? "…" : "Generate"}
            </Button>
          </div>
        </div>
      </div>

      {done && result && (
        <div className="absolute inset-x-0 bottom-0 grid place-items-center px-4 pb-5">
          <div className="flex w-full max-w-[34rem] flex-col items-center gap-3">
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
