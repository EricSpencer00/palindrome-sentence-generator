import { useEffect, useMemo, useState } from "react"

const A = ({ href, children }: { href: string; children: React.ReactNode }) => (
  <a href={href} target="_blank" rel="noopener noreferrer"
     className="text-signal hover:text-ink">{children}</a>
)

const H = ({ children }: { children: React.ReactNode }) => (
  <h2 className="label mb-3 mt-14 first:mt-0">{children}</h2>
)

/* Quoted letters and phrases: underlined, never italic. */
const U = ({ children }: { children: React.ReactNode }) => (
  <span className="underline decoration-from-font underline-offset-4">{children}</span>
)

const P = ({ children }: { children: React.ReactNode }) => <p className="mt-4 first:mt-0">{children}</p>

type Saved = { left: string[]; center?: string; centerDisplay?: string; pivot?: number; pivotOdd?: boolean; right: string[]; letters: number }

/* A saved paragraph from a retired version, written out from its mirror in a
   small box of its own, so the poster above stays where it is. */
function Replay({ version, item }: { version: string; item: Saved }) {
  const mid = item.centerDisplay || item.center || ""
  /* Every letter gets its place on one tape; letter j mirrors letter N-1-j. */
  const words = useMemo(() => {
    let n = 0
    return [...item.left, ...(mid ? [mid] : []), ...item.right].map((w) => {
      const at = [...w].map((c) => (/[a-z]/i.test(c) ? n++ : -1))
      return { w, at, lo: Math.min(...at.filter((x) => x >= 0)), hi: Math.max(...at) }
    })
  }, [item, mid])
  const N = words.reduce((m, x) => Math.max(m, x.hi + 1), 0)
  const half = N / 2
  // How far each word's outer edge sits from the mirror: both ends finish together.
  const dist = (x: { lo: number; hi: number }) => Math.max(half - x.lo, x.hi + 1 - half, 0)
  const total = Math.ceil(half)
  const [p, setP] = useState(0)
  const [pick, setPick] = useState<Set<number> | null>(null)
  useEffect(() => {
    setP(0); setPick(null)
    const t0 = performance.now()
    const t = window.setInterval(() => {
      const k = Math.min(1, (performance.now() - t0) / 3000)
      setP(k * total)
      if (k >= 1) window.clearInterval(t)
    }, 30)
    return () => window.clearInterval(t)
  }, [item, total])
  const mirror = pick ? new Set([...pick].map((j) => N - 1 - j)) : null
  const bar = <span className="caret mx-px inline-block h-[1.1em] w-[2px] translate-y-[.2em] bg-signal" />
  return (
    <div className="tut-in mt-4 rounded-2xl border border-haze p-4" onClick={() => setPick(null)}>
      <div className="label mb-2">{version} · {item.letters} letters</div>
      <p className="break-words text-center font-mono text-[13px] leading-relaxed">
        {words.map((x, i) => (
          <span key={i}
                onClick={(e) => { e.stopPropagation(); setPick(new Set(x.at.filter((j) => j >= 0))) }}
                className={`cursor-pointer transition-opacity duration-300 ${dist(x) <= p ? "opacity-100" : "opacity-0"}`}>
            {[...x.w].map((c, k) => {
              const j = x.at[k]
              const cls = j < 0 ? "" : pick?.has(j) ? "bg-signal text-paper"
                : mirror?.has(j) ? "bg-signal-soft text-signal"
                : N % 2 === 1 && j === (N - 1) / 2 ? "pivot" : ""
              return <span key={k}>{N % 2 === 0 && j === half ? bar : null}<span className={cls}>{c}</span></span>
            })}{" "}
          </span>
        ))}
      </p>
    </div>
  )
}

/* Below the poster: why it exists, how it works, and what failed first. */
export default function About({ archives = {} }: { archives?: Record<string, Saved[]> }) {
  const versions = Object.keys(archives).sort()
  const [shown, setShown] = useState<{ v: string; item: Saved } | null>(null)
  const replay = (v: string) => {
    const list = archives[v]
    const item = list[Math.floor(Math.random() * list.length)]
    if (item) setShown({ v, item })
  }
  return (
    <article id="about" className="mx-auto max-w-[38rem] px-5 pb-24 pt-16 font-display text-[15px] leading-relaxed text-ink">
      <h1 className="sr-only">Palindrome Paragraph Generator</h1>
      <H>Why a paragraph</H>
      <P>
        Palindromes are words or phrases that mirror themselves halfway. One side’s letters
        reverse into the other side. However, I always thought the existing palindromes out there
        were too short or had exploited a trick that made them unimpressive to me.
      </P>

      <H>How it works</H>
      <P>
        Imagine writing a balance sheet for a financial firm. As you add up one side of the
        ledger, you must reconcile the other side. My intuition on what a palindromic paragraph
        generator would look like was similar to this idea. As you grow from the middle of the
        palindrome, you have no idea what either side is going to end with. So, you can play
        around in a depth-first search kind of way.
      </P>
      <P>
        Click on a longer word above. Longer words usually mean a few small words on the other
        side. But what happens when the word does not flip into another word easily? You can see
        how it must pay the debt that it has created for itself by using other letters from the
        starts or ends of other words. <U>Hate me here</U> flips into <U>ereh em etah</U>, but
        this is uninteresting and gibberish. Instead, split the second part into <U>ere he met
        ah</U> via Brown-corpus sentence construction. Then, further add letters onto the second
        part to create more interesting phrases such as <U>ere he met a hill</U>. But wait, that
        means we need to add more letters to the first phrase. <U>Illi hate me here</U> doesn’t
        work, but <U>Will I hate me here?</U> does. That also means adding a <U>w</U> to the
        second phrase. You can easily see how small additions pile onto each other, creating
        coherent phrases one on top of another.
      </P>

      <H>Choosing among millions</H>
      <P>
        Each step offers dozens of legal moves, so the number of possible paragraphs explodes
        within a few words. A depth-first search commits to one path and dies in its first dead
        end. This generator runs a beam search instead: it keeps the few hundred most promising
        half-built paragraphs, extends every one of them, and throws the rest away.
      </P>
      <P>
        Promise is measured twice. A trigram language model, trained on 3.8 million words of the
        Brown corpus and Project Gutenberg, scores how naturally each step reads. When a sentence
        closes, a counting verifier grades it, with no neural network in the loop: it reads the
        sentence’s grammar off part-of-speech patterns and punishes filler, repetition and
        non-words, so garbage dies in the beam. It runs in a fraction of a
        millisecond, so every candidate is judged, not just the survivor. Commas and full stops
        cost no letters, so a sentence ends wherever the grammar wants it to.
      </P>

      <H>Where it comes from</H>
      <P>
        In 1984, Dan Hoey wrote a program that stretched <U>A man, a plan, a canal: Panama</U> to
        hundreds of words by tracking exactly this kind of letter debt. In 2002, Peter Norvig
        rebuilt it as a search and <A href="https://norvig.com/pal-alg.html">wrote it up</A>. Their
        palindromes are records, not reading: exact, enormous lists of names and nouns. This
        generator keeps their bookkeeping and demands sentences.
      </P>

      <H>What didn’t work</H>
      <ol className="list-none space-y-4">
        <li>
          <b>v1</b> searched word by word, scored by a bigram model and reranked by GPT-2. It made
          exact palindromes at any length and word salad at every length, one unpunctuated run per
          paragraph. GPT-2 rates it six times more perplexing than this version. A blinded judge
          rejected all 25 of its sentences.
        </li>
        <li>
          <b>v2</b> stopped generating and started quoting, placing about 1,300 whole sentences
          mined from Wikipedia into the mirror. It read better because it was borrowing, not
          writing.
        </li>
        <li>
          <b>v3</b> built paragraphs from a bank of 540 verified palindromes, nesting short ones
          inside longer ones. The bank ran dry near a thousand letters, and readers preferred the
          shorter original over the grown paragraph in 20 of 20 pairs. Longer was worse.
        </li>
        <li>
          <b>v4</b> searched for grammatical paths while matching letters at the seams. Its best
          reached 650 letters and still repeated names and broke clauses mid-thought.
        </li>
      </ol>
      <P>
        Each failure narrowed the problem. Exactness is easy. Readability has to be rewarded at
        every step, not checked at the end.
      </P>

      <H>Research</H>
      <P>
        Long searches fall into ruts: 46 of 52 earlier outputs contained <U>no it a</U>. In a
        controlled study of 360 matched runs, penalizing the six most reused phrases removed them
        and raised lexical diversity, and the search found new ruts to replace them. Banning bad
        phrases doesn’t produce good ones. That is why this generator rewards good sentences
        instead of punishing bad ones.
      </P>

      <p className="mt-10">
        <a href="/how/" className="text-signal hover:text-ink">
          How the algorithm works, with walkthroughs and measurements →
        </a>
      </p>

      {versions.length > 0 && (
        <div className="mt-10 flex flex-wrap items-center gap-2">
          <span className="label mr-1">Earlier versions</span>
          {versions.map((v) => (
            <button key={v} onClick={() => replay(v)}
                    className="rounded-[3px] bg-haze px-2 py-0.5 font-mono text-xs text-ink hover:bg-signal hover:text-paper">
              {v}
            </button>
          ))}
        </div>
      )}
      {shown && <Replay version={shown.v} item={shown.item} />}

      <p className="label mt-16">
        <A href="https://github.com/EricSpencer00/palindrome-sentence-generator">source</A> ·{" "}
        <A href="https://ericspencer.us">ericspencer.us</A>
      </p>
    </article>
  )
}
