"""Score each half in the direction the search actually builds it.

The measured problem is that the half grown by prepending reads worse than the
half grown by appending, in both search directions, so the cost belongs to
backward construction rather than to position in the text. A forward language
model cannot fix that incrementally: choosing a word to put *before* a fixed
suffix means asking p(word | what follows), and a forward model can only answer
by rescoring the whole suffix once per candidate.

A backward model — the same GPT-2, fine-tuned on reversed token order — answers
it in one pass. That pass produces a distribution over the neighbouring token,
which scores every candidate word at once. So the cost is one forward pass per
beam state per step, not one per candidate: roughly beam_width passes where
in-loop reranking needs beam_width x candidate_limit.

Two approximations, both deliberate:

- A word is scored by the logprob of its **first** token. Word-aligned
  tokenization makes that the first token of a whole-word block, and most of
  the frequency-ranked vocabulary is a single token with its leading space;
  `single_token_fraction` reports how much of the vocabulary that covers.
- Contexts are truncated to the nearest `max_context` words. The neighbouring
  word carries most of the signal and truncation keeps the passes short.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .scoring import adjacent


class _Directional:
    """One model, plus the token bookkeeping for reading text its way."""

    def __init__(self, path: str, reversed_order: bool, device: str):
        self.reversed_order = reversed_order
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path).to(device).eval()
        self._word_ids: dict[str, list[int]] = {}

    def word_tokens(self, word: str) -> list[int]:
        ids = self._word_ids.get(word)
        if ids is None:
            ids = self.tok(" " + word, add_special_tokens=False)["input_ids"]
            self._word_ids[word] = ids
        return ids

    def context_ids(self, words: Sequence[str], max_context: int) -> list[int]:
        """Token ids for the context, in the order this model reads them.

        The forward model reads the words that precede its target, so the
        nearest context is the tail. The backward model reads the words that
        follow its target, nearest first, so the sequence is reversed.
        """
        if self.reversed_order:
            chosen = list(words[:max_context])
            ids = [i for w in chosen for i in self.word_tokens(w)]
            return ids[::-1]
        chosen = list(words[-max_context:])
        return [i for w in chosen for i in self.word_tokens(w)]

    @torch.no_grad()
    def next_token_logprobs(self, contexts: Sequence[list[int]],
                            batch_size: int = 32) -> list[torch.Tensor]:
        """For each context, the logprob of every possible next token."""
        out: list[torch.Tensor] = []
        eos = self.tok.eos_token_id
        for i in range(0, len(contexts), batch_size):
            chunk = [c or [eos] for c in contexts[i:i + batch_size]]
            width = max(len(c) for c in chunk)
            # Left-pad so the final position is the real last token for all rows.
            ids = torch.tensor([[eos] * (width - len(c)) + c for c in chunk],
                               device=self.device)
            mask = torch.tensor([[0] * (width - len(c)) + [1] * len(c)
                                 for c in chunk], device=self.device)
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, -1]
            out.extend(torch.log_softmax(logits.float(), dim=-1).cpu())
        return out


class DirectionalScorer:
    """Adds a directional LM term to a base scorer's word_delta.

    `appends` names the half that the search grows by appending; the other half
    is the prepended one. Outside-in appends on the left, center-out on the
    right. Getting this backwards silently scores every word against the far end
    of its half, so the search that owns the convention passes it in.
    """

    def __init__(self, base, forward_path: str = "gpt2",
                 backward_path: Optional[str] = None,
                 appends: str = "left", weight: float = 1.0,
                 max_context: int = 8, device: Optional[str] = None):
        self.base = base
        self.weight = weight
        self.max_context = max_context
        self.appends = appends
        dev = device or ("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available() else "cpu")
        self.fwd = _Directional(forward_path, reversed_order=False, device=dev)
        self.bwd = (_Directional(backward_path, reversed_order=True, device=dev)
                    if backward_path else None)
        self._cache: dict[tuple, torch.Tensor] = {}
        self.passes = 0
        self.misses = 0

    def single_token_fraction(self, vocab: Sequence[str]) -> float:
        """Share of the vocabulary the first-token approximation scores exactly."""
        return sum(len(self.fwd.word_tokens(w)) == 1 for w in vocab) / max(1, len(vocab))

    def _model_for(self, growth: str) -> Optional[_Directional]:
        if growth == "append":
            return self.fwd
        return self.bwd

    def prepare(self, states) -> None:
        """Precompute the neighbour distribution for every state in the beam.

        Called once per search step. Every candidate expansion of a state reads
        the same distribution, so this is where the work is amortized.
        """
        wanted: dict[tuple, list[int]] = {}
        for st in states:
            for growth, words in (("append", st.left if self.appends == "left" else st.right),
                                  ("prepend", st.right if self.appends == "left" else st.left)):
                model = self._model_for(growth)
                if model is None:
                    continue
                key = (growth, tuple(words[-self.max_context:] if growth == "append"
                                     else words[:self.max_context]))
                if key not in self._cache and key not in wanted:
                    wanted[key] = model.context_ids(words, self.max_context)
        if not wanted:
            return
        keys = list(wanted)
        for growth in ("append", "prepend"):
            group = [k for k in keys if k[0] == growth]
            model = self._model_for(growth)
            if not group or model is None:
                continue
            dists = model.next_token_logprobs([wanted[k] for k in group])
            self.passes += len(group)
            for k, d in zip(group, dists):
                self._cache[k] = d

    def _lm_term(self, left: tuple, right: tuple, placement: str, word: str,
                 growth: str) -> float:
        model = self._model_for(growth)
        if model is None:
            return 0.0
        seq = left if placement == "L" else right
        context = seq[:-1] if growth == "append" else seq[1:]
        key = (growth, tuple(context[-self.max_context:] if growth == "append"
                             else context[:self.max_context]))
        dist = self._cache.get(key)
        if dist is None:
            self.misses += 1
            dist = model.next_token_logprobs([model.context_ids(context, self.max_context)])[0]
            self._cache[key] = dist
        first = model.word_tokens(word)[0]
        return float(dist[first])

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str,
                   growth: str) -> float:
        base = self.base.word_delta(left, right, placement, word, growth)
        return base + self.weight * self._lm_term(left, right, placement, word, growth)


class ForwardOnlyScorer(DirectionalScorer):
    """Control: the appended half gets its LM term, the prepended half gets
    none.

    This is what the search does today, and it is the honest comparison. A
    forward model has no incremental answer for "which word belongs before this
    suffix" — feeding it the following words in reading order asks a different
    question — so the alternative to a backward model is not a worse LM term on
    that half, it is no LM term at all.
    """

    def __init__(self, base, forward_path: str = "gpt2", **kw):
        kw.pop("backward_path", None)
        super().__init__(base, forward_path=forward_path, backward_path=None, **kw)
