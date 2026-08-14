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

type Result = {
  center: string
  centerDisplay: string
  left: string[]
  right: string[]
  letters: number
  words: number
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

  const done = phase === "done"
  const total = result ? Math.max(result.left.length, result.right.length) : 0

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
  useLayoutEffect(() => {
    const el = blockRef.current
    if (!el) return
    const availW = vp.w - 32
    const availH = Math.max(160, vp.h - TOP_CHROME - BOTTOM_CHROME)
    setFit(Math.min(availW / Math.max(1, el.offsetWidth), availH / Math.max(1, el.offsetHeight)))
  }, [result, worldWidth, vp])

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
  const scale = result ? fit + (startScale - fit) * Math.pow(1 - progress, ZOOM_EASE) : startScale

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
    setView("poster"); setPhase("searching")

    const es = new EventSource(`/api/generate?prompt=${encodeURIComponent(prompt)}&budget=16`)
    esRef.current = es
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.type === "status") { setElapsed(msg.elapsed ?? 0); return }
      if (msg.type === "error") { setError(msg.message); setPhase("error"); es.close(); return }
      if (msg.type === "result") { setResult(msg as Result); setShown(0); setPhase("revealing"); es.close() }
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
    phase === "searching" ? `${elapsed.toFixed(1)}s`
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
      <div
        className={`absolute inset-0 ${view === "read" ? "invisible" : ""}`}
        aria-hidden={view === "read"}
      >
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
            <span key={`l${i}`} style={{ visibility: i >= result.left.length - shown ? "visible" : "hidden" }}>
              {w}{" "}
            </span>
          ))}

          <span className="whitespace-nowrap">
            {result?.centerDisplay && <span className="text-signal">{result.centerDisplay} </span>}
            {!done && <Caret spinning={phase === "searching"} />}
          </span>

          {result?.right.map((w, i) => (
            <span key={`r${i}`} style={{ visibility: i < shown ? "visible" : "hidden" }}>
              {" "}{w}
            </span>
          ))}
        </div>
      </div>

      {view === "read" && result && (
        <div className="absolute inset-0 overflow-y-auto overscroll-contain px-5 pb-32 pt-28">
          <p className="mx-auto max-w-[62ch] break-words text-left text-[15px] leading-[1.85] text-ink sm:text-base">
            {result.left.join(" ")}{" "}
            <span className="text-signal">{result.centerDisplay || result.center}</span>{" "}
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
