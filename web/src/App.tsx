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

const FIRST_STEP_MS = 150
const LAST_STEP_MS = 24
const MAX_SCALE = 5.6
const MIN_SCALE = 0.04
const FONT_PX = 42

export default function App() {
  const [prompt, setPrompt] = useState("")
  const [phase, setPhase] = useState<Phase>("idle")
  const [result, setResult] = useState<Result | null>(null)
  const [shown, setShown] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState("")

  const innerRef = useRef<HTMLDivElement>(null)
  const centerRef = useRef<HTMLSpanElement>(null)
  const esRef = useRef<EventSource | null>(null)

  /* A spring has to TRACK a source value. Calling .set() on the spring itself
     strands it partway to the target, which is why the camera used to stop
     mid-pull with the text still overflowing. */
  const scaleTarget = useMotionValue(MAX_SCALE)
  const scale = useSpring(scaleTarget, { stiffness: 90, damping: 24, mass: 0.9 })

  /* The anchor is the point held at the viewport centre while everything scales
     about it. Horizontally it starts on the caret — early on there is little
     else to look at — and migrates to the middle of the block once the text
     stacks into lines, because the two halves are not the same width and
     caret-anchoring alone leaves the block hanging off one edge. Vertically it
     stays on the mirror line throughout. Both are sprung so the handover is a
     drift rather than a jump. */
  // Set directly, not sprung: this is a layout pin, and the mirror point must
  // sit exactly at the viewport centre on every frame. Smoothness comes from the
  // scale spring; the caret-to-midline handover moves gradually anyway because
  // it is driven by the line count.
  const [anchor, setAnchor] = useState({ x: 0, y: 0 })

  const total = result ? Math.max(result.left.length, result.right.length) : 0

  /* The whole palindrome is laid out the moment it arrives — hidden words still
     occupy their slots. Nothing reflows during the reveal, so once a word has a
     position it keeps it. */
  const worldWidth = useMemo(
    () => (result ? Math.min(2400, Math.max(620, Math.sqrt(result.letters) * 52)) : 620),
    [result],
  )

  const measure = useCallback(() => {
    const inner = innerRef.current
    const c = centerRef.current
    if (!inner || !c) return

    const caretX = c.offsetLeft + c.offsetWidth / 2
    const caretY = c.offsetTop + c.offsetHeight / 2

    // Extents of the revealed band, in the block's own coordinates.
    const on = inner.querySelectorAll<HTMLElement>('[data-on="1"]')
    let x0 = c.offsetLeft, x1 = c.offsetLeft + c.offsetWidth
    let y0 = c.offsetTop, y1 = c.offsetTop + c.offsetHeight
    on.forEach((el) => {
      x0 = Math.min(x0, el.offsetLeft); x1 = Math.max(x1, el.offsetLeft + el.offsetWidth)
      y0 = Math.min(y0, el.offsetTop);  y1 = Math.max(y1, el.offsetTop + el.offsetHeight)
    })

    /* Anchor halfway between the mirror point and the middle of the revealed
       band. Pure caret-anchoring is exactly centred but forces the fit to cover
       twice the longer side, throwing away half the screen; pure band-centring
       fills the screen but lets the mirror point wander. Halfway keeps the
       caret near the middle and the text large — equal-ish, which is all the
       asymmetric word boundaries allow anyway. */
    const ax = (caretX + (x0 + x1) / 2) / 2
    const ay = (caretY + (y0 + y1) / 2) / 2
    setAnchor((prev) =>
      Math.abs(prev.x - ax) < 0.5 && Math.abs(prev.y - ay) < 0.5 ? prev : { x: ax, y: ay })

    // Fit the band, allowing for the anchor sitting off its centre.
    const w = 2 * Math.max(ax - x0, x1 - ax)
    const h = 2 * Math.max(ay - y0, y1 - ay)
    const fit = Math.min(
      (window.innerWidth * 0.88) / Math.max(1, w),
      (window.innerHeight * 0.62) / Math.max(1, h),
    )
    scaleTarget.set(Math.max(MIN_SCALE, Math.min(MAX_SCALE, fit)))
  }, [scaleTarget])

  useLayoutEffect(measure, [measure, shown, result, worldWidth])

  useEffect(() => {
    window.addEventListener("resize", measure)
    return () => window.removeEventListener("resize", measure)
  }, [measure])

  useEffect(() => {
    if (phase !== "revealing" || !result) return
    if (shown >= total) { setPhase("done"); return }
    const p = total > 1 ? shown / (total - 1) : 1
    const delay = FIRST_STEP_MS + (LAST_STEP_MS - FIRST_STEP_MS) * Math.sqrt(p)
    const t = window.setTimeout(() => setShown((n) => n + 1), delay)
    return () => clearTimeout(t)
  }, [phase, shown, total, result])

  const generate = useCallback(() => {
    esRef.current?.close()
    setResult(null); setShown(0); setError(""); setElapsed(0)
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

  const status =
    phase === "searching" ? `searching · ${elapsed.toFixed(1)}s`
    : phase === "revealing" ? `${Math.min(shown * 2, result?.words ?? 0)} of ${result?.words ?? 0} words`
    : phase === "done" && result
      ? `${result.letters} letters · ${result.words} words · ${Math.round((result.coherence ?? 0) * 100)}% real bigrams · ${result.seconds}s`
    : phase === "error" ? error
    : "a phrase to mirror — optional"

  return (
    <div className="relative h-full w-full overflow-hidden bg-paper">

      <div className="absolute left-1/2 top-1/2 h-0 w-0">
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
            // position:relative makes this the offsetParent, so the centre
            // span's offsetLeft/Top are in the same coordinate space as the
            // transformOrigin below. Without it they resolve against an outer
            // element and the whole block sits off-centre.
            style={{ width: worldWidth, fontSize: FONT_PX, lineHeight: 1.42, position: "relative" }}
            className="text-center text-ink"
          >
            {result?.left.map((w, i) => {
              const on = i >= result.left.length - shown
              return (
                <span key={`l${i}`} data-on={on ? "1" : "0"}
                      style={{ opacity: on ? 1 : 0 }}
                      className="transition-opacity duration-200">{w} </span>
              )
            })}

            <span ref={centerRef} data-on="1" className="whitespace-nowrap">
              {result?.centerDisplay && <span className="text-signal">{result.centerDisplay} </span>}
              {phase !== "done" && <span className="caret text-signal" aria-hidden="true">|</span>}
            </span>

            {result?.right.map((w, i) => {
              const on = i < shown
              return (
                <span key={`r${i}`} data-on={on ? "1" : "0"}
                      style={{ opacity: on ? 1 : 0 }}
                      className="transition-opacity duration-200"> {w}</span>
              )
            })}
          </div>
        </motion.div>
      </div>

      {/* Bands so the text passes under the chrome instead of colliding with it. */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-paper via-paper to-transparent" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-20 bg-gradient-to-t from-paper via-paper to-transparent" />

      <div className="pointer-events-none absolute inset-x-0 top-0 grid place-items-center pt-10">
        <div className="pointer-events-auto flex w-[min(92vw,34rem)] flex-col gap-2">
          <label htmlFor="p" className="label pl-1">{status}</label>
          <div className="flex gap-2">
            <div className="slab flex-1 bg-paper">
              <Input
                id="p"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") generate() }}
                placeholder="never odd or even"
                disabled={phase === "searching"}
                className="h-11 border-0 bg-transparent font-mono text-base shadow-none focus-visible:ring-0"
              />
            </div>
            <Button
              onClick={generate}
              disabled={phase === "searching"}
              className="slab slab-press h-11 rounded-[3px] border-0 bg-ink px-6 font-display text-xs font-bold uppercase tracking-[.16em] text-paper hover:bg-signal"
            >
              {phase === "searching" ? "…" : "Generate"}
            </Button>
          </div>
        </div>
      </div>

      <div className="absolute inset-x-0 bottom-0 grid place-items-center pb-6">
        <p className="label">reads the same backwards · gpt-2 on a mac mini · after john tromp</p>
      </div>
    </div>
  )
}
