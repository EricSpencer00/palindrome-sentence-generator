"""Score each half in the direction the search actually builds it.

The forward half is scored left-to-right. The half inserted before an existing
suffix is scored with a model trained on reversed token streams, so it answers
the matching conditional distribution. Both directions score every token in a
proposed word or phrase; a one-token shortcut is not safe for a 30k-word search.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def context_key(words: Sequence[str], growth: str, max_context: int) -> tuple:
    """The nearest words visible to the directional model."""
    tail = (tuple(words[-max_context:]) if growth == "append"
            else tuple(words[:max_context]))
    return growth, tail


def leading_token(ids: Sequence[int], reversed_order: bool) -> int:
    """Compatibility helper for old diagnostics, not the production score."""
    return ids[-1] if reversed_order else ids[0]


def ordered_tokens(ids: Sequence[int], reversed_order: bool) -> list[int]:
    """Return a word block in the token order the selected model reads."""
    return list(reversed(ids)) if reversed_order else list(ids)


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Position real tokens from zero even when a batch is left-padded."""
    return (mask.long().cumsum(dim=-1) - 1).clamp_min(0)


class _Directional:
    """One model and exact whole-word conditional likelihoods."""

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

    def target_tokens(self, word: str) -> list[int]:
        return ordered_tokens(self.word_tokens(word), self.reversed_order)

    def context_ids(self, words: Sequence[str], max_context: int) -> list[int]:
        """Token ids for the context, in the order this model reads them."""
        if self.reversed_order:
            chosen = list(words[:max_context])
            ids = [i for word in chosen for i in self.word_tokens(word)]
            return ids[::-1]
        chosen = list(words[-max_context:])
        return [i for word in chosen for i in self.word_tokens(word)]

    @torch.no_grad()
    def next_token_logprobs(self, contexts: Sequence[list[int]],
                            batch_size: int = 32) -> list[torch.Tensor]:
        """Distribution diagnostics, with left-padding positions corrected."""
        out: list[torch.Tensor] = []
        eos = self.tok.eos_token_id
        for start in range(0, len(contexts), batch_size):
            chunk = [context or [eos] for context in contexts[start:start + batch_size]]
            width = max(len(context) for context in chunk)
            ids = torch.tensor([[eos] * (width - len(context)) + context
                                for context in chunk], device=self.device)
            mask = torch.tensor([[0] * (width - len(context)) + [1] * len(context)
                                 for context in chunk], device=self.device)
            logits = self.model(input_ids=ids, attention_mask=mask,
                                position_ids=position_ids_from_mask(mask)).logits[:, -1]
            out.extend(torch.log_softmax(logits.float(), dim=-1).cpu())
        return out

    @torch.no_grad()
    def word_logprobs(self, context: list[int], words: Sequence[str],
                      batch_size: int = 32) -> list[float]:
        """Exact log p(word | context), including every token in the word.

        Rows are right-padded so batch companions cannot alter absolute token
        positions. The return is a total log probability; callers normalize it
        only over the legal candidates from this structural state.
        """
        if not words:
            return []
        eos = self.tok.eos_token_id
        prefix = context or [eos]
        targets = [self.target_tokens(word) for word in words]
        out: list[float] = []
        for start in range(0, len(words), batch_size):
            chunk_targets = targets[start:start + batch_size]
            rows = [prefix + target for target in chunk_targets]
            width = max(len(row) for row in rows)
            ids = torch.tensor([row + [eos] * (width - len(row)) for row in rows],
                               device=self.device)
            mask = torch.tensor([[1] * len(row) + [0] * (width - len(row))
                                 for row in rows], device=self.device)
            logits = self.model(input_ids=ids, attention_mask=mask,
                                position_ids=position_ids_from_mask(mask)).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            first = len(prefix) - 1
            for row, target in enumerate(chunk_targets):
                positions = torch.arange(first, first + len(target), device=self.device)
                target_ids = torch.tensor(target, device=self.device)
                picked = logprobs[row, positions].gather(
                    -1, target_ids.unsqueeze(-1)).squeeze(-1)
                out.append(float(picked.sum()))
        return out


class DirectionalScorer:
    """Base score plus exact directional word likelihoods.

    ``word_deltas`` is a batch API because a language-model z-score is meaningful
    only relative to candidates legal at the current overhang, not a global
    vocabulary. ``beam_search`` and ``centerout_search`` call it per parent.
    """

    def __init__(self, base, forward_path: str = "gpt2",
                 backward_path: Optional[str] = None,
                 appends: str = "left", weight: float = 1.0,
                 max_context: int = 8, device: Optional[str] = None,
                 vocab: Optional[Sequence[str]] = None, batch_size: int = 32):
        self.base = base
        self.weight = weight
        self.max_context = max_context
        self.appends = appends
        self.batch_size = batch_size
        dev = device or ("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available() else "cpu")
        self.fwd = _Directional(forward_path, reversed_order=False, device=dev)
        self.bwd = (_Directional(backward_path, reversed_order=True, device=dev)
                    if backward_path else None)
        self._word_cache: dict[tuple, float] = {}
        # Retained solely for the old one-token coverage diagnostic.
        self.vocab = tuple(vocab or ())
        self.passes = 0
        self.misses = 0

    def single_token_fraction(self, vocab: Sequence[str]) -> float:
        """Share that the former one-token approximation scored exactly."""
        return sum(len(self.fwd.word_tokens(word)) == 1 for word in vocab) / max(1, len(vocab))

    def _model_for(self, growth: str) -> Optional[_Directional]:
        return self.fwd if growth == "append" else self.bwd

    def prepare(self, states) -> None:
        """Compatibility hook; exact scores are batched over legal children."""

    def _context(self, left: tuple[str, ...], right: tuple[str, ...],
                 placement: str, growth: str) -> tuple[str, ...]:
        sequence = left if placement == "L" else right
        return sequence[:-1] if growth == "append" else sequence[1:]

    def word_deltas(self, choices: Sequence[tuple[tuple[str, ...], tuple[str, ...],
                                                  str, str, str]]) -> list[float]:
        """Return base-plus-LM deltas for one parent's legal children."""
        base = [self.base.word_delta(left, right, placement, word, growth)
                for left, right, placement, word, growth in choices]
        groups: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
        for index, (left, right, placement, word, growth) in enumerate(choices):
            if self._model_for(growth) is None:
                continue
            context = self._context(left, right, placement, growth)
            groups[(growth, context_key(context, growth, self.max_context))].append((index, word))

        out = list(base)
        for (growth, key), entries in groups.items():
            model = self._model_for(growth)
            assert model is not None
            context_words = key[1]
            missing: list[str] = []
            for _, word in entries:
                cache_key = (growth, context_words, word)
                if cache_key not in self._word_cache and word not in missing:
                    missing.append(word)
            if missing:
                scores = model.word_logprobs(model.context_ids(context_words, self.max_context),
                                              missing, batch_size=self.batch_size)
                self.passes += math.ceil(len(missing) / self.batch_size)
                self.misses += len(missing)
                for word, score in zip(missing, scores):
                    self._word_cache[(growth, context_words, word)] = score
            values = [self._word_cache[(growth, context_words, word)]
                      for _, word in entries]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            scale = max(1e-3, variance ** 0.5)
            for (index, _), value in zip(entries, values):
                out[index] += self.weight * ((value - mean) / scale)
        return out

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str,
                   growth: str) -> float:
        """Single-choice fallback; search callers should use ``word_deltas``."""
        return self.base.word_delta(left, right, placement, word, growth)


class ForwardOnlyScorer(DirectionalScorer):
    """Control: score only the half that grows in normal reading direction."""

    def __init__(self, base, forward_path: str = "gpt2", **kw):
        kw.pop("backward_path", None)
        super().__init__(base, forward_path=forward_path, backward_path=None, **kw)
