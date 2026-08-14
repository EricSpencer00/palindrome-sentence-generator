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
