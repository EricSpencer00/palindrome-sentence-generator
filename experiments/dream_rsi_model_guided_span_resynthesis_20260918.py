"""Dream-RSI: model-guided complete-span resynthesis.

The earlier denoising lanes logged a mismatch mask but never let a model
propose a replacement.  This lane closes that implementation gap.  A local
GPT-2 policy proposes a *complete* phrase for the live seam of a short,
authored scaffold.  The proposal is inserted on the left and its character
residual is placed on the reflected interval, with a fresh lexical
segmentation on the right.  Thus the tape is exact after every transition;
the model is still responsible for proposing and ranking the span, not for
certifying readability.

The scaffold is the user's 38-letter seed and is explicitly marked as a
bootstrap, not as a novel output.  Every longer row is retained, independently
audited, and withheld from the reader package until blinded human raters accept
the intact sentence over a shuffled control.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "dream-rsi-model-guided-span-resynthesis-20260918"
MODEL_NAME = "gpt2"

# This is a user-supplied short seed used only as a construction scaffold.
# The run never presents it alone as a new discovery.
SEED_LEFT = "An aide rips nine memos;"
SEED_RIGHT = "some men inspire Diana."

WORD_RE = re.compile(r"[A-Za-z]+")
STOPWORDS = {
    "a", "an", "the", "and", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "after", "before", "near", "over", "under", "some",
}
COMMON_ONE_LETTER = {"a", "i"}


def letters(text: str) -> str:
    return "".join(WORD_RE.findall(text.casefold()))


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [
        {"left_index": i, "right_index": len(tape) - 1 - i,
         "left": tape[i], "right": tape[-1 - i]}
        for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]
    ]
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:8],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
        "sha_equal_under_reversal": tape == tape[::-1],
    }


def _word_ok(word: str) -> bool:
    if len(word) == 1:
        return word in COMMON_ONE_LETTER
    # A model can emit a rare name, but not arbitrary subword fragments.
    return zipf_frequency(word, "en") >= 2.8


def segment_reverse(tape: str, max_words: int = 7) -> tuple[str, dict]:
    """Segment the reflected residual into ordinary words.

    This is a bounded lexical chart, not a reversal of the finished tape.  A
    no-solution chart is retained as a failed proposal rather than silently
    emitting character chunks.
    """
    n = len(tape)
    best: list[tuple[float, list[str]] | None] = [None] * (n + 1)
    best[n] = (0.0, [])
    for i in range(n - 1, -1, -1):
        options: list[tuple[float, list[str]]] = []
        for j in range(i + 1, min(n, i + 13) + 1):
            word = tape[i:j]
            if not _word_ok(word) or best[j] is None:
                continue
            freq = zipf_frequency(word, "en")
            # Mildly prefer longer lexical words; penalise word salad.
            score = freq + 0.12 * len(word) - 0.42
            tail_score, tail_words = best[j]
            options.append((score + tail_score, [word, *tail_words]))
        if options:
            best[i] = max(options, key=lambda row: (row[0], -len(row[1])))
    if best[0] is None:
        return "", {"solved": False, "words": [], "score": None}
    score, words = best[0]
    if len(words) > max_words:
        return "", {"solved": False, "words": words, "score": score,
                     "reason": "too_many_words"}
    return " ".join(words), {"solved": True, "words": words, "score": score}


def shortcut_flags(text: str, generated_span: str) -> dict:
    words = [w.casefold() for w in WORD_RE.findall(text)]
    content = [w for w in words if w not in STOPWORDS]
    return {
        "word_order_only": False,
        "self_palindromic_content_words": [w for w in content if len(w) > 1 and w == w[::-1]],
        "repeated_content_words": sorted({w for w in content if content.count(w) > 1}),
        "catalogue_text": False,
        "finished_tape_reversed": False,
        "whole_seed_reused_without_resynthesis": not generated_span.strip(),
    }


class GPT2SpanPolicy:
    """Small local policy used for actual proposal and whole-row reranking."""

    def __init__(self, model_name: str = MODEL_NAME, seed: int = 91018):
        self.model_name = model_name
        self.seed = seed
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def propose(self, prompt: str, count: int = 12) -> list[dict]:
        torch.manual_seed(self.seed)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **encoded, max_new_tokens=14, do_sample=True, top_p=0.94,
            temperature=0.82, num_return_sequences=count,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prefix_len = encoded["input_ids"].shape[1]
        rows = []
        for row in out:
            raw = self.tokenizer.decode(row[prefix_len:], skip_special_tokens=True)
            # Keep the first complete lexical span; punctuation is supplied by
            # the scaffold and cannot change the letter tape.
            words = WORD_RE.findall(raw.casefold())[:8]
            if len(words) < 2:
                continue
            span = " ".join(words)
            rows.append({"raw_model_text": raw, "span": span,
                         "prompt": prompt, "model": self.model_name})
        return rows

    @torch.no_grad()
    def score(self, texts: Iterable[str]) -> list[float]:
        batch = list(texts)
        if not batch:
            return []
        enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(self.device)
        logits = self.model(**enc).logits
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        target = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:]
        picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * mask
        return [float(picked[i].sum().item() / max(1, int(mask[i].sum().item())))
                for i in range(len(batch))]


def render(span: str, reflected: str) -> str:
    return f"{SEED_LEFT} {span}; {reflected}, {SEED_RIGHT}"


def build_candidate(proposal: dict) -> dict:
    span = proposal["span"]
    tape = letters(span)
    reflected_tape = tape[::-1]
    reflected_surface, segmentation = segment_reverse(reflected_tape)
    if not reflected_surface:
        reflected_surface = reflected_tape
        segmentation = {"solved": False, "words": [], "score": None,
                         "fallback": "unsegmented_residual"}
    text = render(span, reflected_surface)
    a = audit(text)
    return {
        "rendered": text,
        "generated_span": span,
        "reflected_residual_tape": reflected_tape,
        "reflected_surface": reflected_surface,
        "reverse_segmentation": segmentation,
        "audit": a,
        "shortcut_flags": shortcut_flags(text, span),
        "provenance": {
            "fresh_model_proposal": True,
            "model": proposal["model"],
            "prompt": proposal["prompt"],
            "raw_model_text": proposal["raw_model_text"],
            "seed_scaffold": f"{SEED_LEFT} {SEED_RIGHT}",
            "seed_is_user_supplied_bootstrap": True,
            "catalogue_imported": False,
            "finished_tape_reversed": False,
            "whole_tape_reversed": False,
            "mirrored_interval_only": True,
            "reader_certified": False,
        },
    }


def run(count: int = 12, model_name: str = MODEL_NAME) -> dict:
    prompt = (
        "Complete the missing middle of a short English scene with one concise "
        "natural phrase (two to eight words). Before the gap: An aide rips "
        "nine memos. After the gap: some men inspire Diana. Middle phrase:"
    )
    policy = GPT2SpanPolicy(model_name=model_name)
    proposals = policy.propose(prompt, count=count)
    rows = [build_candidate(p) for p in proposals]
    scores = policy.score([r["rendered"] for r in rows])
    for row, score in zip(rows, scores):
        row["model_score_per_token"] = score
        row["exact_admission"] = bool(
            row["audit"]["two_pointer_exact"]
            and not row["shortcut_flags"]["self_palindromic_content_words"]
            and not row["shortcut_flags"]["repeated_content_words"]
            and row["reverse_segmentation"].get("solved", False)
        )
    rows.sort(key=lambda row: (-row["model_score_per_token"], row["audit"]["letters"]))
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in rows if row["exact_admission"]]
    return {
        "experiment": EXPERIMENT,
        "method": "Dream-RSI actual local-model complete-span proposal plus reflected-residual lexical chart",
        "model_policy": {"model": model_name, "local_files_only": True,
                          "proposal_count": count, "prompt": prompt,
                          "generation_is_model_guided": True,
                          "reranking_is_diagnostic_only": True},
        "bootstrap": {"left": SEED_LEFT, "right": SEED_RIGHT,
                      "letters": len(letters(SEED_LEFT + SEED_RIGHT)),
                      "exact": audit(SEED_LEFT + " " + SEED_RIGHT),
                      "used_as_long_output": False},
        "rendered_candidates": rows,
        "fresh_exact_closures": exact,
        "admitted_without_reader": admitted,
        "stats": {"proposals": len(proposals), "rendered": len(rows),
                  "exact": len(exact), "lexically_segmented_reflections": sum(
                      r["reverse_segmentation"].get("solved", False) for r in rows),
                  "shortcut_free_exact": len(admitted),
                  "longest_letters": max((r["audit"]["letters"] for r in rows), default=0)},
        "reader_gate": {"status": "closed",
                        "reason": "No blinded intact-vs-shuffled human ratings; model score cannot certify readability.",
                        "programmatic_metrics_are_diagnostic": True},
        "next_repair": {
            "operator": "replace the bootstrap with two independently model-authored boundary clauses and carry a live reflected-residual chart across both seams",
            "reason": "the seed scaffold makes every row exact but does not establish a novel readable long sentence",
            "required_reader_test": "randomized blinded intact versus word-shuffled controls for every exact survivor",
        },
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "human_readability_certified": False},
    }


if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        directory.mkdir(exist_ok=True)
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))
    for row in payload["rendered_candidates"][:5]:
        print(f"{row['audit']['letters']} letters | exact={row['audit']['two_pointer_exact']} | {row['rendered']}")
