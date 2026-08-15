"""Language-model scoring for palindrome candidates.

GPT2Scorer scores whole texts (mean token logprob per letter) in batches.
Used two ways:
  1. rerank: pick the most fluent of many letter-valid closed palindromes
  2. prune: periodically rescore the beam during search so fluent branches
     survive (passed to beam_search as lm_prune)

A forward LM reads the right half in its natural order too, since the right
half is stored in final reading order — only the *search* grows it backwards.
"""
from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPT2ConditionalScorer:
    """Per-token logprobs of a tail, conditioned on a prefix.

    Backs `coherence.CoherenceMetric`, which scores the same tail against two
    different prefixes and subtracts. That only means anything if the tail
    tokenizes identically both times, so the tail is tokenized ON ITS OWN and
    the ids concatenated — never by tokenizing the joined string, where BPE
    would merge across the seam and the two runs would be over different units.
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def conditional_logprobs(self, prefix: str, tail: str) -> list[float]:
        pre_ids = self.tok(prefix.strip(), add_special_tokens=False).input_ids
        # The leading space is part of the first tail token: GPT-2 spells a
        # mid-sentence word as " word", and dropping it would score a
        # start-of-line variant that never occurs in the text being measured.
        tail_ids = self.tok(" " + tail.strip(), add_special_tokens=False).input_ids
        if not pre_ids or not tail_ids:
            return []

        ids = torch.tensor([pre_ids + tail_ids], device=self.device)
        logits = self.model(ids).logits
        logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
        targets = ids[0, 1:]
        picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # `picked[i]` scores ids[i+1], so the first tail token sits at
        # len(pre_ids) - 1.
        return picked[len(pre_ids) - 1:].tolist()


class GPT2Scorer:
    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_texts(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        """Mean token logprob per alphabetic character, per text."""
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(self.device)
            logits = self.model(**enc).logits
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:]
            tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask
            for j, text in enumerate(batch):
                letters = max(1, sum(c.isalpha() for c in text))
                out.append(tok_lp[j].sum().item() / letters)
        return out
