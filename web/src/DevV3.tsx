import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

/* /dev — v3, in the poster's clothes.
 *
 * Same page as `/`: white, one control cluster at the top, the text, a credit
 * line at the bottom. Two differences, and both are because v3 is a different
 * kind of generator, not because the page wanted more furniture.
 *
 * The prompt is not a theme. v1 takes a word and steers toward it; v3 composes
 * from a fixed bank and has nothing to steer. What it does have is one slot the
 * algebra leaves free — every position but the centre is half of a mirror-pair
 * and is fixed by its opposite number — so the box takes the visitor's own
 * palindrome and puts it there. It is checked and refused, never repaired.
 *
 * The slider is length. v1's length is whatever the search closes at; v3's is a
 * parameter, so it is a control rather than a statistic.
 *
 * Everything else that was on this page — the stat grid, the criteria table,
 * the toggles, the explanation of what a mirror-pair is — is in README.md and
 * docs/NORTH-STAR.md, which is where a reader can go looking for it. On the
 * page it was spending attention the text needs.
 */

type Chunk = { slot: number; role: "left" | "centre" | "right"; text: string; source: string }

type Composition = {
  text: string
  letters: number
  words: number
  capacity_letters: number
  chunks: Chunk[]
  centre_is_yours: boolean
}

const MIN_LETTERS = 40
const FALLBACK_CAP = 14500

export default function DevV3() {
  const [own, setOwn] = useState("")
  const [target, setTarget] = useState(1200)
  const [cap, setCap] = useState(FALLBACK_CAP)
  const [comp, setComp] = useState<Composition | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    fetch("/api/v3/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((h) => { if (h?.capacity?.novel?.max_letters) setCap(h.capacity.novel.max_letters) })
      .catch(() => { /* the fallback is the deployed bank's size */ })
  }, [])

  const generate = useCallback(() => {
    abort.current?.abort()
    const ac = new AbortController()
    abort.current = ac
    setBusy(true)
    setError(null)
    const q = new URLSearchParams({ letters: String(target), seed: String(Date.now() % 1e9) })
    if (own.trim()) q.set("centre", own.trim())
    fetch(`/api/v3/composition?${q}`, { signal: ac.signal })
      .then(async (r) => {
        const body = await r.json().catch(() => ({}))
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`)
        return body
      })
      .then((c: Composition) => { setComp(c); setBusy(false) })
      .catch((e) => {
        if (e.name === "AbortError") return
        setError(String(e.message || e)); setComp(null); setBusy(false)
      })
  }, [target, own])

  useEffect(() => { generate() }, [])          // one on arrival, so the page is never empty
  useEffect(() => () => abort.current?.abort(), [])

  const copy = useCallback(async () => {
    if (!comp) return
    try {
      await navigator.clipboard.writeText(comp.text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch { /* clipboard unavailable; the text is selectable */ }
  }, [comp])

  /* Size the type off the length so a short one is a poster and a long one is
     still a page. Monospace, like `/`, because the mirror is a property of the
     characters and proportional type hides that. */
  const size = !comp ? 28
    : comp.letters < 150 ? 40
    : comp.letters < 500 ? 28
    : comp.letters < 2000 ? 20
    : comp.letters < 6000 ? 15
    : 12

  /* The centre marked in signal, which is the one thing on this page worth a
     second colour: it is where the visitor's own text goes, and it is the only
     position that is not determined by another. */
  const parts = useMemo(() => {
    if (!comp) return null
    const centre = comp.chunks.find((c) => c.role === "centre")
    if (!centre) return null
    const lets = (s: string) => s.toLowerCase().replace(/[^a-z]/g, "")
    const before = comp.chunks.slice(0, comp.chunks.indexOf(centre))
      .reduce((n, c) => n + lets(c.text).length, 0)
    const want = lets(centre.text).length
    let seen = 0, start = -1, end = -1
    for (let p = 0; p < comp.text.length && end < 0; p++) {
      const isLetter = /[a-z]/i.test(comp.text[p])
      if (start < 0 && seen === before && isLetter) start = p
      if (isLetter && ++seen === before + want && start >= 0) end = p + 1
    }
    if (start < 0 || end < 0) return null
    return [comp.text.slice(0, start), comp.text.slice(start, end), comp.text.slice(end)]
  }, [comp])

  const status = busy ? "…"
    : error ? error
    : comp ? `${comp.letters.toLocaleString()} letters · ${comp.words} words`
      + (comp.centre_is_yours ? " · yours at the centre" : "")
    : ""

  return (
    <div className="relative h-full w-full overflow-hidden bg-paper">

      <div className="absolute inset-0 overflow-y-auto px-4 pb-32 pt-40 sm:pt-48">
        <p className="mx-auto max-w-6xl text-center font-mono text-ink"
           style={{ fontSize: size, lineHeight: 1.5 }}>
          {parts
            ? <>{parts[0]}<span className="text-signal">{parts[1]}</span>{parts[2]}</>
            : comp?.text}
        </p>
      </div>

      {comp && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-paper via-paper to-transparent" />
      )}

      {/* Top: what it is, then the two controls. The status line doubles as the
          slider's readout while dragging, so the slider needs no label of its
          own — one line of chrome instead of two. */}
      <div className="pointer-events-none absolute inset-x-0 top-0 grid place-items-center px-4 pt-6 sm:pt-10">
        <div className="pointer-events-auto flex w-full max-w-[34rem] flex-col gap-2">
          <label htmlFor="own" className={`label min-h-[0.9rem] pl-1 ${error ? "text-signal" : ""}`}>
            {status}
          </label>
          <div className="flex gap-2">
            <div className="slab min-w-0 flex-1 bg-paper">
              <Input
                id="own"
                value={own}
                onChange={(e) => setOwn(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") generate() }}
                placeholder="add your own palindrome"
                disabled={busy}
                autoComplete="off" autoCapitalize="none" autoCorrect="off"
                spellCheck={false} enterKeyHint="go"
                className="h-11 border-0 bg-transparent font-mono text-base shadow-none focus-visible:ring-0"
              />
            </div>
            <Button
              onClick={generate}
              disabled={busy}
              className="slab slab-press h-11 shrink-0 rounded-[3px] border-0 bg-ink px-4 font-display text-[11px] font-bold uppercase tracking-[.14em] text-paper hover:bg-signal sm:px-6 sm:text-xs sm:tracking-[.16em]"
            >
              {busy ? "…" : "Generate"}
            </Button>
          </div>
          <div className="flex items-center gap-3 pl-1">
            <input
              type="range"
              aria-label="length in letters"
              min={MIN_LETTERS} max={cap} step={20}
              value={Math.min(target, cap)}
              onChange={(e) => setTarget(Number(e.target.value))}
              onMouseUp={generate}
              onTouchEnd={generate}
              onKeyUp={(e) => { if (e.key.startsWith("Arrow")) generate() }}
              className="range h-11 flex-1"
            />
            <span className="label w-16 shrink-0 tabular-nums text-right">
              {target.toLocaleString()}
            </span>
          </div>
        </div>
      </div>

      {comp && (
        <div className="absolute inset-x-0 bottom-0 grid place-items-center px-4 pb-5">
          <div className="flex w-full max-w-[34rem] flex-col items-center gap-3">
            <button
              onClick={copy}
              className="slab slab-press h-9 rounded-[3px] bg-paper px-4 font-display text-[11px] font-bold uppercase tracking-[.14em] text-ink"
            >
              {copied ? "Copied" : "Copy"}
            </button>
            <p className="label text-center leading-relaxed">
              reads the same backwards · composed, not searched ·{" "}
              <a href="/" className="underline decoration-from-font underline-offset-2 hover:text-signal">
                the poster
              </a>{" "}
              ·{" "}
              <a href="https://ericspencer.us" target="_blank" rel="noopener noreferrer"
                 className="underline decoration-from-font underline-offset-2 hover:text-signal">
                ericspencer.us
              </a>
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
