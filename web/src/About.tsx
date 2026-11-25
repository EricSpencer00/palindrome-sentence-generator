const A = ({ href, children }: { href: string; children: React.ReactNode }) => (
  <a href={href} target="_blank" rel="noopener noreferrer"
     className="underline decoration-from-font underline-offset-2 hover:text-signal">{children}</a>
)

const H = ({ children }: { children: React.ReactNode }) => (
  <h2 className="label mb-3 mt-14 first:mt-0">{children}</h2>
)

const P = ({ children }: { children: React.ReactNode }) => <p className="mt-4 first:mt-0">{children}</p>

/* Below the poster: why it exists, how it works, and what failed first. */
export default function About() {
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
        starts or ends of other words. <i>Hate me here</i> flips into <i>ereh em etah</i>, but
        this is uninteresting and gibberish. Instead, split the second part into <i>ere he met
        ah</i> via Brown-corpus sentence construction. Then, further add letters onto the second
        part to create more interesting phrases such as <i>ere he met a hill</i>. But wait, that
        means we need to add more letters to the first phrase. <i>Illi hate me here</i> doesn’t
        work, but <i>Will I hate me here?</i> does. That also means adding a <i>w</i> to the
        second phrase, and you can easily see how small additions can pile onto each other,
        creating coherent phrases one on top of another.
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
        In 1984, Dan Hoey wrote a program that stretched <i>A man, a plan, a canal: Panama</i> to
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
        Long searches fall into ruts: 46 of 52 earlier outputs contained <i>no it a</i>. In a
        controlled study of 360 matched runs, penalizing the six most reused phrases removed them
        and raised lexical diversity, and the search found new ruts to replace them. Banning bad
        phrases doesn’t produce good ones. That is why this generator rewards good sentences
        instead of punishing bad ones.
      </P>

      <p className="mt-10">
        <a href="/how/" className="underline decoration-from-font underline-offset-2 hover:text-signal">
          How the algorithm works, with walkthroughs and measurements →
        </a>
      </p>

      <p className="label mt-16">
        <A href="https://github.com/EricSpencer00/palindrome-sentence-generator">source</A> ·{" "}
        <A href="https://ericspencer.us">ericspencer.us</A>
      </p>
    </article>
  )
}
