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
    <article id="about" className="mx-auto max-w-[38rem] px-5 pb-24 pt-16 font-display text-[15px] leading-relaxed text-ink">
      <H>How it works</H>
      <p>
        A palindrome is a ledger that has to balance. Start in the middle and write outward, and
        every word placed on one side runs up a debt: the other side must eventually spell the same
        letters backward. Lay down <i>rats live on</i> on the left and the right is already in
        arrears, owing <i>no evil star</i>. Whichever side owes goes shopping for words that pay it
        off. A word that overpays is not wasted; the surplus becomes the other side's debt. And when
        the books come out even, the generator can slip a small palindrome such as <i>racecar</i> or{" "}
        <i>noon</i> into the seam, on both sides at once.
      </p>
      <p className="mt-4">
        One paragraph is hopeless, so a beam search keeps a few hundred half-built ones alive and
        lets them compete. Commas and full stops cost no letters, which means a sentence can end
        almost anywhere it pleases. A trigram language model, schooled on 3.8 million words of the
        Brown corpus and Project Gutenberg prose, scores every step. Then comes the censor: a
        counting verifier, no neural network, that judges each sentence the moment it closes. It
        sorts text along a spectrum from repetitive garbage to stock phrases, and it rewards the
        middle, where prose is readable without being cliché. It takes a fraction of a millisecond
        per sentence, so it sits in judgment over every candidate, not just the survivor.
      </p>

      <H>Where it comes from</H>
      <p>
        In 1984, Dan Hoey wrote a program that took <i>A man, a plan, a canal: Panama</i> and kept
        going, matching the letters one side owed the other until the palindrome ran hundreds of
        words. In 2002, Peter Norvig rebuilt the idea as a search over those leftovers and{" "}
        <A href="https://norvig.com/pal-alg.html">wrote it up</A>. Both were astonishing feats. Both
        produced text nobody could read aloud without embarrassment: long, exact, breathless lists
        of names and nouns. This generator keeps their debt and asks for something harder, which
        is sentences.
      </p>

      <H>What didn't work</H>
      <ol className="list-none space-y-4">
        <li>
          <b>v1</b> searched word by word with a bigram model and GPT-2 reranking. It was a
          failure, and an instructive one. The palindromes were exact at any length and word salad
          at every length. Of its words, 52.6% had one or two letters, against 18.5% in ordinary
          English; the search had discovered that tiny words are cheap. A blinded judge rejected
          all 25 of its sentences.
        </li>
        <li>
          <b>v2</b> placed whole sentences mined from Wikipedia, about 1,300 of them. It read
          better for a dishonest reason: it quoted instead of writing.
        </li>
        <li>
          <b>v3</b> nested 540 verified palindromes from an offline bank. It stalled near a
          thousand letters, when the bank ran dry, and blind readers preferred the shorter,
          ungrown version in 20 of 20 pairs. Growth made it worse.
        </li>
        <li>
          <b>v4</b> searched grammatical paths while matching letters at the seams. Its best, a
          650-letter result, still repeated names and broke clauses mid-thought.
        </li>
      </ol>

      <H>Research</H>
      <p>
        Long searches fall into ruts, and the ruts are deep: 46 of 52 earlier outputs contained
        the phrase <i>no it a</i>. In a controlled study across 360 matched runs,
        penalising the six most reused phrases wiped them out entirely and raised lexical
        diversity. The search shrugged and found new ruts. Banning bad phrases does not produce
        good ones, which is why the current generator rewards good sentences instead.
      </p>

      <p className="label mt-16">
        <A href="https://github.com/EricSpencer00/palindrome-sentence-generator">source</A> ·{" "}
        <A href="https://ericspencer.us">ericspencer.us</A>
      </p>
    </article>
  )
}
