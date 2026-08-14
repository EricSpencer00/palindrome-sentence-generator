import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { motion, useMotionValue, useSpring } from "motion/react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

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

const FIRST_STEP_MS = 150
const LAST_STEP_MS = 24
const MAX_SCALE = 5.6
const MIN_SCALE = 0.04
const FONT_PX = 42

/* Chrome heights reserved above and below the canvas, so the fit is computed
   against the space actually available rather than a guessed fraction of the
   viewport. Both are reserved for the whole run, including before the credits
   exist: varying the reserve on `done` meant the last measurement that mattered
   ran under the smaller one, and the finished text sat under the chrome. */
const TOP_CHROME = 108
const BOTTOM_CHROME = 116

export default function App() {
  const [prompt, setPrompt] = useState("")
  const [phase, setPhase] = useState<Phase>("idle")
  const [result, setResult] = useState<Result | null>(null)
  const [shown, setShown] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState("")
  const [view, setView] = useState<View>("poster")
  const [copied, setCopied] = useState(false)

  const innerRef = useRef<HTMLDivElement>(null)
  const centerRef = useRef<HTMLSpanElement>(null)
  const esRef = useRef<EventSource | null>(null)

  /* A spring has to TRACK a source value — setting the spring itself strands it
     partway to the target. */
  const scaleTarget = useMotionValue(MAX_SCALE)
  const scale = useSpring(scaleTarget, { stiffness: 90, damping: 24, mass: 0.9 })
  const [anchor, setAnchor] = useState({ x: 0, y: 0 })

  const done = phase === "done"
  const total = result ? Math.max(result.left.length, result.right.length) : 0

  /* The whole palindrome is laid out the moment it arrives — hidden words still
     occupy their slots — so nothing reflows during the reveal. */
  const worldWidth = useMemo(
    () => (result ? Math.min(2400, Math.max(560, Math.sqrt(result.letters) * 52)) : 560),
    [result],
  )

  const measure = useCallback(() => {
    const inner = innerRef.current
    const c = centerRef.current
    if (!inner || !c) return

    /* Extents of the revealed band, in the block's own coordinates.

       Measured per line-fragment via getClientRects(). offsetWidth on an inline
       span that wraps across a line counts its fragments rather than its visual
       box, which made the band measure nearly twice the block width and pinned
       the anchor to the block's right edge. Screen rects are converted back
       through the live scale. */
    const s = scale.get() || 1
    const ir = inner.getBoundingClientRect()
    const on = inner.querySelectorAll<HTMLElement>('[data-on="1"]')
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    on.forEach((el) => {
      for (const r of Array.from(el.getClientRects())) {
        x0 = Math.min(x0, (r.left - ir.left) / s)
        x1 = Math.max(x1, (r.right - ir.left) / s)
        y0 = Math.min(y0, (r.top - ir.top) / s)
        y1 = Math.max(y1, (r.bottom - ir.top) / s)
      }
    })
    if (!isFinite(x0) || !isFinite(y0)) {
      x0 = c.offsetLeft; x1 = c.offsetLeft + c.offsetWidth
      y0 = c.offsetTop;  y1 = c.offsetTop + c.offsetHeight
    }

    /* Anchor on the centre of the REVEALED band, not the caret and not the
       whole block. Early on the band is barely wider than the caret, so this is
       caret-centred; by the end it is the finished text, so it fills the screen.
       Anchoring on the caret instead forces the fit to cover twice its longer
       side and throws away half the width. */
    const ax = (x0 + x1) / 2
    const ay = (y0 + y1) / 2
    setAnchor((prev) =>
      Math.abs(prev.x - ax) < 0.5 && Math.abs(prev.y - ay) < 0.5 ? prev : { x: ax, y: ay })

    const availW = window.innerWidth - 24
    const availH = Math.max(160, window.innerHeight - TOP_CHROME - BOTTOM_CHROME)
    const fit = Math.min(availW / Math.max(1, x1 - x0), availH / Math.max(1, y1 - y0))
    scaleTarget.set(Math.max(MIN_SCALE, Math.min(MAX_SCALE, fit)))
  }, [scaleTarget, scale])

  useLayoutEffect(measure, [measure, shown, result, worldWidth, view])

  useEffect(() => {
    window.addEventListener("resize", measure)
    window.addEventListener("orientationchange", measure)
    return () => {
      window.removeEventListener("resize", measure)
      window.removeEventListener("orientationchange", measure)
    }
  }, [measure])

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
    setView("poster")
    setPhase("searching")
    scaleTarget.set(MAX_SCALE)

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
  }, [prompt, scaleTarget])

  useEffect(() => () => esRef.current?.close(), [])

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(fullText)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch { /* clipboard unavailable; the text is selectable in read view */ }
  }, [fullText])

  const status =
    phase === "searching" ? `searching · ${elapsed.toFixed(1)}s`
    : phase === "revealing" ? `${Math.min(shown * 2, result?.words ?? 0)} of ${result?.words ?? 0} words`
    : phase === "error" ? error
    : done && result
      ? `${result.letters} letters · ${result.words} words · ${result.seconds}s`
    // Idle says nothing: the placeholder already explains the field, and the
    // brief is that an untouched page is only the prompt and the caret.
    : ""

  return (
    <div className="relative h-full w-full overflow-hidden bg-paper">

      {/* ---------- canvas ---------- */}
      <div className={`absolute left-1/2 top-1/2 h-0 w-0 ${view === "read" ? "invisible" : ""}`}>
        <motion.div
          style={{
            scale,
            transformOrigin: `${anchor.x}px ${anchor.y}px`,
            marginLeft: -anchor.x,
            marginTop: -anchor.y,
          }}
          className="will-change-transform"
        >
          <div
            ref={innerRef}
            style={{ width: worldWidth, fontSize: FONT_PX, lineHeight: 1.42, position: "relative" }}
            className="text-center text-ink"
          >
            {result?.left.map((w, i) => {
              const on = i >= result.left.length - shown
              return (
                <span key={`l${i}`} data-on={on ? "1" : "0"} style={{ opacity: on ? 1 : 0 }}
                      className="transition-opacity duration-200">{w} </span>
              )
            })}

            <span ref={centerRef} data-on="1" className="whitespace-nowrap">
              {result?.centerDisplay && <span className="text-signal">{result.centerDisplay} </span>}
              {!done && <span className="caret text-signal" aria-hidden="true">|</span>}
            </span>

            {result?.right.map((w, i) => {
              const on = i < shown
              return (
                <span key={`r${i}`} data-on={on ? "1" : "0"} style={{ opacity: on ? 1 : 0 }}
                      className="transition-opacity duration-200"> {w}</span>
              )
            })}
          </div>
        </motion.div>
      </div>

      {/* ---------- read view: the whole thing at a size you can actually read ---------- */}
      {view === "read" && result && (
        <div className="absolute inset-0 overflow-y-auto overscroll-contain px-5 pb-32 pt-28">
          <p className="mx-auto max-w-[62ch] text-left text-[15px] leading-[1.85] break-words text-ink sm:text-base">
            {result.left.join(" ")}{" "}
            <span className="text-signal">{result.centerDisplay || result.center}</span>{" "}
            {result.right.join(" ")}
          </p>
        </div>
      )}

      {/* Bands so text passes under the chrome rather than colliding with it. */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-paper via-paper to-transparent sm:h-28" />
      {done && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-paper via-paper to-transparent" />
      )}

      {/* ---------- prompt ---------- */}
      <div className="pointer-events-none absolute inset-x-0 top-0 grid place-items-center px-4 pt-6 sm:pt-10">
        <div className="pointer-events-auto flex w-full max-w-[34rem] flex-col gap-2">
          {/* Height is reserved so the prompt does not jump when a status appears. */}
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
                autoComplete="off"
                autoCapitalize="none"
                autoCorrect="off"
                spellCheck={false}
                enterKeyHint="go"
                /* 16px stops iOS Safari zooming the page on focus. */
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

      {/* ---------- credits: only once there is something to credit ---------- */}
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
              <a href="https://tromp.github.io/pal/pal.html" target="_blank" rel="noopener noreferrer"
                 className="underline decoration-from-font underline-offset-2 hover:text-signal">
                john tromp
              </a>
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
