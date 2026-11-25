"""Whole-paragraph log-likelihood under a causal LM, batched on a GPU.

The centre-out search scores tokens with a trigram as it goes; every few steps
it re-anchors each surviving state on this, which reads the whole paragraph
left to right, the way a reader does.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class NeuralScorer:
    def __init__(self, model: str, device: str = "cuda", dtype=torch.bfloat16, batch: int = 32):
        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(device).eval()
        self.device, self.batch = device, batch
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    @torch.no_grad()
    def totals(self, texts: list[str]) -> list[float]:
        """Sum of token log-probs of each text, conditioned on a BOS token."""
        out: list[float] = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            ids = torch.cat([torch.full((len(chunk), 1), self.bos), enc.input_ids], dim=1).to(self.device)
            mask = torch.cat([torch.ones(len(chunk), 1, dtype=enc.attention_mask.dtype),
                              enc.attention_mask], dim=1).to(self.device)
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            out += (lp * mask[:, 1:]).sum(dim=1).tolist()
        return out
