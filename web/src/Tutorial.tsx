import { useEffect, useRef, useState, type ReactNode } from "react"

const STEPS: ReactNode[] = [
  <>
    A palindrome is a phrase that mirrors itself halfway through. ABC at the start becomes CBA
    at the end. It can be as short as “racecar” or as long as Peter Norvig’s 21,012-word{" "}
    <a href="https://norvig.com/palindrome.html" target="_blank"
       rel="noopener noreferrer" className="text-signal hover:text-ink">
      “A man, a plan, a caretaker, …, a canal, Panama”
    </a>
    .
  </>,
  <>Click on any word above and see where its letters lie in the other half of the phrase.</>,
]

const FADE_MS = 180

/* A pill that opens into a card. Steps cross-fade: the old one fades out, then
   the next fades in, so two never overlap. */
export default function Tutorial({ onOpenChange }: { onOpenChange: (open: boolean) => void }) {
  const [open, setOpenState] = useState(false)
  const setOpen = (o: boolean) => { setOpenState(o); onOpenChange(o) }
  const [step, setStep] = useState(0)
  const [shown, setShown] = useState(true)
  const timer = useRef(0)
  const pill = useRef<HTMLButtonElement>(null)
  const wasOpen = useRef(false)
  useEffect(() => { if (wasOpen.current && !open) pill.current?.focus(); wasOpen.current = open }, [open])
  useEffect(() => () => window.clearTimeout(timer.current), [])

  const go = (to: number) => {
    if (to === step || !shown) return
    setShown(false)
    timer.current = window.setTimeout(() => { setStep(to); setShown(true) }, FADE_MS)
  }
  const close = () => { window.clearTimeout(timer.current); setOpen(false); setStep(0); setShown(true) }
  const last = step === STEPS.length - 1

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-30 flex justify-center px-4 pb-5">
      {!open ? (
        <div className="tut-in pointer-events-auto flex items-center gap-2">
          <button
            ref={pill}
            onClick={(e) => { e.stopPropagation(); setOpen(true) }}
            className="rounded-full border border-border bg-paper px-5 py-2.5 font-display text-[15px] text-ink transition-colors hover:border-signal hover:text-signal"
          >
            What is this?
          </button>
          <button
            aria-label="Read more"
            onClick={(e) => {
              e.stopPropagation()
              document.getElementById("about")?.scrollIntoView({ behavior: "smooth" })
            }}
            className="nudge grid h-11 w-11 place-items-center rounded-full bg-ink text-paper transition-colors hover:bg-signal"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor"
                 strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M8 3v10M3.5 8.5 8 13l4.5-4.5" />
            </svg>
          </button>
        </div>
      ) : (
        <div
          role="dialog" aria-label="What is this?"
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => { if (e.key === "Escape") close() }}
          className="tut-in pointer-events-auto w-full max-w-[24rem] overflow-hidden rounded-2xl border border-border bg-paper shadow-[0_8px_32px_rgba(13,13,16,.12)]"
        >
          {/* Every step sits in one grid cell, so the card is always as tall as
              the longest step and never resizes between them. */}
          <div className="grid px-5 pb-3 pt-5 font-display text-[14px] leading-relaxed text-ink">
            {STEPS.map((body, i) => (
              <p
                key={i}
                aria-hidden={i !== step}
                aria-live={i === step ? "polite" : undefined}
                className="col-start-1 row-start-1 transition-[opacity,transform,filter] ease-out"
                style={{
                  transitionDuration: `${FADE_MS}ms`,
                  visibility: i === step ? "visible" : "hidden",
                  opacity: i === step && shown ? 1 : 0,
                  transform: i === step && shown ? "none" : "translateY(4px)",
                  filter: i === step && shown ? "none" : "blur(2px)",
                }}
              >
                {body}
              </p>
            ))}
          </div>
          <nav className="flex items-center justify-between border-t border-border px-3 py-2">
            <button onClick={() => (step ? go(step - 1) : close())}
                    className="rounded-full px-3 py-1 font-display text-[13px] text-muted-foreground hover:text-ink">
              {step ? "Back" : "Close"}
            </button>
            <div className="flex gap-1.5">
              {STEPS.map((_, i) => (
                <span key={i} className={`h-1.5 rounded-full transition-[width] duration-300 ${
                  i === step ? "w-4 bg-ink" : "w-1.5 bg-border"}`} />
              ))}
            </div>
            <button autoFocus onClick={() => (last ? close() : go(step + 1))}
                    className="rounded-full bg-ink px-3 py-1 font-display text-[13px] text-paper hover:bg-signal">
              {last ? "Done" : "Next"}
            </button>
          </nav>
        </div>
      )}
    </div>
  )
}
