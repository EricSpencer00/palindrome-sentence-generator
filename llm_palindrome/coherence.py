"""Global coherence: does a text's own opening inform its own ending?

Every other score in this project has a horizon of two words. `_coverage` in
the service counts adjacent attested bigrams, `BigramModel` scores word pairs,
and GPT-2's `lm_score` is a mean over tokens that rewards local fluency without
ever asking whether the text is about anything. A palindrome can max all three
and still change subject every three words, which is exactly what the current
output does — so none of them can serve as the target of a search for
coherence, and pointing an optimizer at them would just buy more of what we
already have.

What "holds a topic" means operationally is that the beginning is EVIDENCE
about the end. So: score the second half twice, once conditioned on the text's
own first half and once conditioned on somebody else's, and take the
difference.

    gain = logp(tail | own head) - logp(tail | foreign head)      [nats/token]

Zero says the text's own opening was worth no more than a stranger's — locally
fluent, globally about nothing. Positive says the opening predicted the ending.
The difference is what makes this immune to the failure that has bitten every
other metric here: absolute fluency cancels, so a text cannot raise its score
by using longer words, commoner words, or repeating itself, which is precisely
how `lm_score` was gamed for +0.30 in `docs/training.md`.

The controls are foreign prefixes rather than a shuffle of the text's own,
because a shuffled prefix keeps the vocabulary and therefore keeps the topic —
it would measure syntax, not subject.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


class ConditionalScorer(Protocol):
    """Per-unit logprobs of `tail`, conditioned on `prefix`.

    The unit is the scorer's own (words for the test doubles, BPE tokens for
    GPT-2). The one requirement is that the returned length depends only on
    `tail`, never on `prefix` — the metric subtracts these elementwise across
    different prefixes, so a tokenization that shifted with the context would
    be comparing different things.
    """

    def conditional_logprobs(self, prefix: str, tail: str) -> list[float]: ...


@dataclass
class CoherenceResult:
    gain: Optional[float]       # nats/token; None when the text is too short
    own: Optional[float]        # mean logprob given the text's own opening
    control: Optional[float]    # mean logprob given foreign openings
    head: str
    tail: str
    scored_tokens: int


def split_at_word(text: str) -> tuple[str, str]:
    """Halve the text on a word boundary, head first."""
    words = text.split()
    half = len(words) // 2
    return " ".join(words[:half]), " ".join(words[half:])


class SelfShuffledControls:
    """Controls made by shuffling the text's own head.

    The first design drew controls from other texts, and calibration killed it:
    a foreign prefix differs from the real one in VOCABULARY as well as in
    order, so the gain measured how much the text reuses its own words. Fully
    word-shuffled prose scored higher than the prose it was made from, and the
    palindrome output scored 0.58 for repeating itself.

    Shuffling the text's own head fixes the vocabulary exactly and varies only
    the order, so what is left is the information carried by ARRANGEMENT. It
    also supplies a free zero point: a text that is already word salad reads
    the same shuffled again, so it must score ~0 by construction.
    """

    def __init__(self, n: int = 4, seed: int = 0):
        self.n = n
        self.seed = seed

    def __call__(self, head: str) -> list[str]:
        import random
        rng = random.Random(self.seed)
        words = head.split()
        out = []
        for _ in range(self.n):
            shuffled = words[:]
            rng.shuffle(shuffled)
            out.append(" ".join(shuffled))
        return out


class CoherenceMetric:
    """Long-range conditional gain, in nats per token.

    `skip_tokens` drops the first units of the tail. A foreign prefix makes a
    jarring seam and the model pays for it in the first few tokens; that cost
    is about the junction rather than about the topic, and counting it would
    hand every text a positive score for merely continuing itself.
    """

    def __init__(self, scorer: ConditionalScorer, controls: Sequence[str],
                 skip_tokens: int = 5):
        if not controls:
            raise ValueError("need at least one control prefix")
        self.scorer = scorer
        self.controls = list(controls)
        self.skip_tokens = skip_tokens

    def score(self, text: str,
              controls: Optional[Sequence[str]] = None) -> CoherenceResult:
        """`controls` overrides the constructor's, for controls derived from
        `text` itself — see `SelfShuffledControls`."""
        head, tail = split_at_word(text)

        own_all = self.scorer.conditional_logprobs(head, tail) if head and tail else []
        own = own_all[self.skip_tokens:]
        if not head or not tail or not own:
            return CoherenceResult(None, None, None, head, tail, 0)

        control_means = []
        for prefix in (controls if controls is not None else self.controls):
            vals = self.scorer.conditional_logprobs(prefix, tail)[self.skip_tokens:]
            if vals:
                control_means.append(sum(vals) / len(vals))
        if not control_means:
            return CoherenceResult(None, None, None, head, tail, 0)

        own_mean = sum(own) / len(own)
        control_mean = sum(control_means) / len(control_means)
        return CoherenceResult(gain=own_mean - control_mean, own=own_mean,
                               control=control_mean, head=head, tail=tail,
                               scored_tokens=len(own))
