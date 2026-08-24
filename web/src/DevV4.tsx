import { useCallback, useEffect, useMemo, useRef, useState } from "react"

/* /dev — the v4 reader.
 *
 * `/` is the poster: it watches a search write a palindrome and the drama is
 * the writing. This page is the opposite and deliberately so. v4 does not
 * search on request; it composes from a bank that was walked out offline and
 * is re-verified on load, so there is nothing to watch and the only thing
 * worth looking at is the text. It is therefore laid out as something to
 * read — punctuation, capitals, sentence breaks — with the structure
 * underneath available on demand rather than always on screen.
 *
 * Every mark here is free. `validator.normalize` strips case, spaces and
 * punctuation before the mirror is checked, which is the same licence the
 * catalogue takes when it writes "A man, a plan, a canal: Panama". The server
 * asserts the letters are unchanged before it answers; this page checks the
 * same thing again in the browser, because a claim a reader can verify on the
 * page beats one they have to take from a README.
 */

type Chunk = { slot: number; role: "left" | "centre" | "right"; text: string; source: string }

type Composition = {
  version: number
  text: string
  plain: string
  letters: number
  words: number
  pairs: number
  requested_letters: number
  capacity_letters: number
  chunks: Chunk[]
  distinct_chunks: number
  repeats: number
}

type Health = {
  ok: boolean
  bank: number
  generated: number
  catalogue: number
  capacity: { novel?: { pairs: number; max_letters: number }; all?: { pairs: number; max_letters: number } }
  error: string | null
}

/* The mirror check, run on what is actually on the screen.
 *
 * Deliberately over the RENDERED string rather than the server's `plain`
 * field: the point is that the punctuation this page adds is invisible to the
 * constraint, and checking the plain text would test the wrong thing. */
const letters = (s: string) => s.toLowerCase().replace(/[^a-z]/g, "")
const mirrors = (s: string) => {
  const n = letters(s)
  return n.length > 0 && n === [...n].reverse().join("")
}

/* Length presets. A slider over 40..14,500 spends most of its travel in a
   range nobody wants, and the interesting comparisons are between orders of
   magnitude, not between 812 and 844 letters. */
const LENGTHS = [
  { n: 120, label: "120" },
  { n: 400, label: "400" },
  { n: 1200, label: "1,200" },
  { n: 4000, label: "4,000" },
  { n: 14500, label: "14,500" },
]

function Toggle({ on, onClick, children, title }: {
  on: boolean; onClick: () => void; children: React.ReactNode; title: string
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-pressed={on}
      className={`slab slab-press rounded-[3px] border-0 px-3 py-1.5 font-display text-[10px]
        font-bold uppercase tracking-[.14em] transition-colors
        ${on ? "bg-ink text-paper" : "bg-haze text-ash hover:text-ink"}`}>
      {children}
    </button>
  )
}

function Stat({ k, v, tone }: { k: string; v: string; tone?: "signal" | "bad" }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="label">{k}</span>
      <span className={`font-mono text-[13px] tabular-nums
        ${tone === "signal" ? "text-signal" : tone === "bad" ? "text-destructive" : "text-ink"}`}>
        {v}
      </span>
    </div>
  )
}

export default function DevV4() {
  const [health, setHealth] = useState<Health | null>(null)
  const [comp, setComp] = useState<Composition | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [target, setTarget] = useState(400)
  const [novel, setNovel] = useState(true)
  const [longestFirst, setLongestFirst] = useState(false)
  const [showSeams, setShowSeams] = useState(false)
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1e9))
  const [copied, setCopied] = useState(false)

  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    fetch("/api/v4/health")
      .then((r) => (r.ok ? r.json() : null))
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  const load = useCallback(() => {
    abort.current?.abort()
    const ac = new AbortController()
    abort.current = ac
    setLoading(true)
    setError(null)
    const q = new URLSearchParams({
      letters: String(target),
      seed: String(seed),
      novel: String(novel),
      longest_first: String(longestFirst),
    })
    fetch(`/api/v4/composition?${q}`, { signal: ac.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`)
        return r.json()
      })
      .then((c: Composition) => { setComp(c); setLoading(false) })
      .catch((e) => {
        if (e.name === "AbortError") return
        setError(String(e.message || e)); setLoading(false)
      })
  }, [target, seed, novel, longestFirst])

  useEffect(() => { load() }, [load])
  useEffect(() => () => abort.current?.abort(), [])

  const verified = useMemo(() => (comp ? mirrors(comp.text) : false), [comp])

  const copy = useCallback(async () => {
    if (!comp) return
    try {
      await navigator.clipboard.writeText(comp.text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch { /* clipboard unavailable; the text is selectable */ }
  }, [comp])

  /* Reading size falls with length. 120 letters wants to be a headline; 14,500
     wants to be a page, and set at headline size it is a scroll nobody
     finishes. */
  const size = !comp ? 20
    : comp.letters < 200 ? 30
    : comp.letters < 700 ? 24
    : comp.letters < 2500 ? 19
    : 16

  const cap = health?.capacity?.[novel ? "novel" : "all"]?.max_letters

  return (
    <div className="h-full w-full overflow-y-auto bg-paper">
      <div className="mx-auto flex min-h-full max-w-3xl flex-col gap-8 px-5 py-10 sm:px-8 sm:py-14">

        <header className="flex flex-col gap-3">
          <div className="flex items-baseline gap-3">
            <span className="slab rounded-[3px] bg-signal px-2 py-1 font-display text-[10px] font-bold uppercase tracking-[.16em] text-paper">
              v4 · dev
            </span>
            <a href="/" className="label hover:text-ink">← the poster</a>
          </div>
          <h1 className="font-display text-[26px] font-bold leading-[1.1] tracking-tight text-ink sm:text-[32px]">
            A palindrome, written out.
          </h1>
          {/* What is actually different, in the two sentences it takes. The
              poster's own credit line says v1 composes from single words, so
              the contrast a returning visitor needs is the unit and the
              presentation, not the fact that it mirrors. */}
          <p className="max-w-xl font-mono text-[13px] leading-relaxed text-ash">
            Composed from verified mirror-pairs rather than searched for on request, then
            punctuated: capitals, sentence breaks, one colon where a run is sentence-shaped
            without a verb. The marks are free — case, spaces and punctuation are invisible
            to the mirror.
          </p>
        </header>

        {/* ------------------------------------------------------- controls */}
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <span className="label">Length, in letters</span>
            <div className="flex flex-wrap gap-2">
              {LENGTHS.map(({ n, label }) => (
                <Toggle
                  key={n}
                  on={target === n}
                  onClick={() => setTarget(n)}
                  title={cap && n > cap ? `the bank holds ${cap} letters; this asks for more` : `${label} letters`}>
                  {label}
                </Toggle>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Toggle on={novel} onClick={() => setNovel(!novel)}
              title="Restrict to palindromes this project's own enumeration found. Off admits the catalogue — text somebody else wrote.">
              {novel ? "Ours only" : "Catalogue too"}
            </Toggle>
            <Toggle on={longestFirst} onClick={() => setLongestFirst(!longestFirst)}
              title="Spend the bank on fewer, longer chunks. Every pair boundary is a seam where two unrelated fragments meet — but this has not been judged, so it is not claimed to read better.">
              Fewer seams
            </Toggle>
            <Toggle on={showSeams} onClick={() => setShowSeams(!showSeams)}
              title="Tint alternating chunks so the assembly is visible.">
              Show seams
            </Toggle>
            <button
              onClick={() => setSeed(Math.floor(Math.random() * 1e9))}
              className="slab slab-press rounded-[3px] border-0 bg-ink px-3 py-1.5 font-display text-[10px] font-bold uppercase tracking-[.14em] text-paper hover:bg-signal">
              Again
            </button>
            <button
              onClick={copy}
              disabled={!comp}
              className="slab slab-press rounded-[3px] border-0 bg-haze px-3 py-1.5 font-display text-[10px] font-bold uppercase tracking-[.14em] text-ash hover:text-ink disabled:opacity-40">
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </div>

        {/* ----------------------------------------------------------- text */}
        <div className="slab min-h-[40vh] rounded-[3px] bg-paper p-5 sm:p-7">
          {error ? (
            <p className="font-mono text-[13px] leading-relaxed text-destructive">{error}</p>
          ) : loading && !comp ? (
            <p className="font-mono text-[13px] text-ash">composing<span className="caret">_</span></p>
          ) : comp ? (
            <p
              className={`font-display text-ink transition-opacity ${loading ? "opacity-40" : "opacity-100"}`}
              style={{ fontSize: size, lineHeight: 1.55, hyphens: "none" }}>
              {showSeams
                ? comp.chunks.map((c, i) => (
                    /* Tinted by ROLE and by parity, so a left chunk and the
                       right chunk carrying its reversed letters do not read as
                       the same object. The centre is the one slot that has no
                       partner. */
                    <span
                      key={c.slot}
                      title={`${c.role} · ${c.source}`}
                      className={
                        c.role === "centre" ? "bg-signal text-paper"
                        : i % 2 ? "bg-signal-soft"
                        : ""
                      }>
                      {renderSlice(comp.text, comp.chunks, i)}
                    </span>
                  ))
                : comp.text}
            </p>
          ) : null}
        </div>

        {/* ---------------------------------------------------------- stats */}
        {comp && (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Stat k="Letters" v={comp.letters.toLocaleString()} />
              <Stat k="Words" v={comp.words.toLocaleString()} />
              <Stat k="Mirror-pairs" v={String(comp.pairs)} />
              <Stat
                k="Repeated chunks"
                v={String(comp.repeats)}
                tone={comp.repeats ? "bad" : undefined} />
            </div>

            {/* Checked here, in the browser, on the punctuated string that is
                actually on the screen. */}
            <p className={`font-mono text-[12px] ${verified ? "text-ink" : "text-destructive"}`}>
              {verified
                ? "✓ Re-checked in your browser: strip the case, spaces and punctuation above and the letters read the same both ways."
                : "✗ The text on this page does not mirror. That is a bug — please report it."}
            </p>

            {comp.requested_letters > comp.capacity_letters && (
              <p className="font-mono text-[12px] text-ash">
                Asked for {comp.requested_letters.toLocaleString()} letters; the bank holds{" "}
                {comp.capacity_letters.toLocaleString()} of usable material, so this is as long as it goes.
              </p>
            )}
          </div>
        )}

        {/* --------------------------------------------------------- footer */}
        <footer className="mt-auto flex flex-col gap-2 pt-6">
          {health && (
            <p className="font-mono text-[11px] text-ash">
              Bank: {health.bank} verified palindromes — {health.generated} walked out by this
              project, {health.catalogue} from the record. Every one is re-verified when the
              server loads it and again before it is served.
            </p>
          )}
          <p className="font-mono text-[11px] text-ash">
            Structure: L1 L2 … C … R2 R1, where each Ri is Li's letters reversed and is
            therefore different text. A run of self-palindromic units would have to repeat
            itself; this does not.
          </p>
        </footer>
      </div>
    </div>
  )
}

/* Cut the presented text at chunk boundaries.
 *
 * The server returns the chunks as PLAIN word runs and the text as a
 * punctuated string, so the two cannot be matched by string equality — the
 * marks and capitals only exist in one of them. They do agree letter for
 * letter, which is the whole invariant, so the boundaries are found by
 * counting letters and the slice is taken in the presented string. */
function renderSlice(text: string, chunks: Chunk[], i: number): string {
  let before = 0
  for (let k = 0; k < i; k++) before += letters(chunks[k].text).length
  const want = letters(chunks[i].text).length

  let seen = 0
  let start = -1
  for (let p = 0; p < text.length; p++) {
    const isLetter = /[a-z]/i.test(text[p])
    if (start < 0 && seen === before && isLetter) start = p
    if (isLetter) {
      seen++
      if (start >= 0 && seen === before + want) {
        // Trailing punctuation and the space belong to the chunk that ends
        // here, or the tint stops one glyph early and the marks all sit in
        // the next chunk's colour.
        let end = p + 1
        while (end < text.length && !/[a-z]/i.test(text[end])) end++
        return text.slice(start, end)
      }
    }
  }
  return start >= 0 ? text.slice(start) : ""
}
