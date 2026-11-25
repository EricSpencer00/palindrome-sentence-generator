const A = ({ href, children }: { href: string; children: React.ReactNode }) => (
  <a href={href} target="_blank" rel="noopener noreferrer"
     className="underline decoration-from-font underline-offset-2 hover:text-signal">{children}</a>
)

const H = ({ children }: { children: React.ReactNode }) => (
  <h2 className="label mb-3 mt-14 first:mt-0">{children}</h2>
)

/* Below the poster: how it works, where it comes from, and what failed first. */
export default function About() {
  return (
    <article className="mx-auto max-w-[38rem] px-5 pb-24 pt-16 font-display text-[15px] leading-relaxed text-ink">
      <H>How it works</H>
      <p>
        The paragraph grows from the middle out. Each time a word goes on one side, the other side
        owes its letters back, in reverse. Put down <i>rats live on</i> and the right half owes{" "}
        <i>no evil star</i>. That debt is the whole state of the search: whichever side owes adds
        words that pay it off, and a word that overpays hands the leftover to the other side. When
        nothing is owed, a palindrome like <i>racecar</i> or <i>noon</i> slips in at the seam, on
        both sides at once.
      </p>
      <p className="mt-4">
        A beam search keeps a few hundred of these half-built paragraphs at a time. Commas and full
        stops cost no letters, so sentences end wherever the grammar wants. A trigram model trained
        on 3.8 million words of Brown and Gutenberg prose scores every step, and a counting verifier
        judges each sentence as it closes. That verifier places text on a line from repetitive
        garbage to stock phrases and rewards the middle: readable, but not a cliché. It takes a
        fraction of a millisecond, so it runs on every candidate instead of once at the end.
      </p>

      <H>Where it comes from</H>
      <p>
        In 1984 Dan Hoey wrote a program that stretched <i>A man, a plan, a canal: Panama</i> into
        a palindrome hundreds of words long, by matching the letters one side owed the other. In
        2002 Peter Norvig rebuilt the idea as a search over those leftovers and{" "}
        <A href="https://norvig.com/pal-alg.html">wrote it up</A>. Their palindromes are exact and
        unreadable by design: long lists of names and nouns. This one keeps their debt and asks
        for sentences.
      </p>

      <H>What didn't work</H>
      <ol className="list-none space-y-4">
        <li>
          <b>v1</b> searched word by word with a bigram model and a GPT-2 rerank. Exact at any
          length, word salad at every length: 52.6% of its words had one or two letters, against
          18.5% in English, and a blinded judge rejected all 25 of its sentences.
        </li>
        <li>
          <b>v2</b> placed whole sentences mined from Wikipedia. It read better because it was
          quoting, not writing.
        </li>
        <li>
          <b>v3</b> nested 540 verified palindromes from an offline bank. It stalled near a
          thousand letters as the bank ran dry, and blind readers preferred the shorter, ungrown
          version in 20 of 20 pairs.
        </li>
        <li>
          <b>v4</b> searched grammatical paths while matching letters at the seams. Its best
          650-letter result still repeated names and broke clauses.
        </li>
      </ol>

      <H>Research</H>
      <p>
        Long searches fall into ruts. 46 of 52 earlier outputs contained <i>no it a</i>. Across 360
        matched runs, penalising the six most reused phrases removed them completely and raised
        lexical diversity, and the search found new ruts to replace them. Forbidding
        bad phrases does not produce good ones, which is why this version rewards sentences
        instead.
      </p>

      <p className="label mt-16">
        <A href="https://github.com/EricSpencer00/palindrome-sentence-generator">source</A> ·{" "}
        <A href="https://ericspencer.us">ericspencer.us</A>
      </p>
    </article>
  )
}
