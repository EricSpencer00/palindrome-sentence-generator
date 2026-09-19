"""Dream-RSI dual-boundary model authoring and residual repair.

The previous model run made a longer exact tape by retaining a user seed.  This
lane removes that bootstrap from every rendered row.  A local GPT-2 policy
authors independent left and right scene clauses, then a residual repair policy
reopens the complete word touching the first mirrored mismatch.  Sibling
clauses are replay worlds; the model score only chooses a route and never
certifies prose.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dream_rsi_model_guided_span_resynthesis_20260918 import (
    GPT2SpanPolicy,
    audit,
    letters,
)

EXPERIMENT = "dream-rsi-dual-boundary-model-authoring-20260918"
WORD_RE = re.compile(r"[A-Za-z]+")

LEFT_PROMPTS = (
    "Write one grammatical English sentence of eight to twelve words about an archivist who records a message:",
    "Write one grammatical English sentence of eight to twelve words about a careful nurse preserving a note:",
)
RIGHT_PROMPTS = (
    "Write one grammatical English sentence of eight to twelve words about a reader carrying a message home:",
    "Write one grammatical English sentence of eight to twelve words about a patient teacher sharing a story:",
)


def clean_clause(raw: str) -> str:
    words = WORD_RE.findall(raw.casefold())[:12]
    return " ".join(words).strip()


def first_residual(left: str, right: str) -> dict | None:
    lt, rt = letters(left), letters(right)
    for i, (a, b) in enumerate(zip(lt, rt[::-1])):
        if a != b:
            return {"left_offset": i, "right_offset": len(rt) - 1 - i,
                    "left": a, "right": b}
    if len(lt) != len(rt):
        return {"left_offset": min(len(lt), len(rt)),
                "right_offset": max(len(lt), len(rt)) - 1,
                "left": lt[min(len(lt), len(rt)):] or None,
                "right": rt[min(len(lt), len(rt)):] or None}
    return None


def word_at(text: str, offset: int) -> str | None:
    at = 0
    for word in WORD_RE.findall(text.casefold()):
        if at <= offset < at + len(word):
            return word
        at += len(word)
    return None


def pair_row(left: str, right: str, policy: str, parent: str | None = None) -> dict:
    rendered = f"{left}. {right}."
    a = audit(rendered)
    residual = first_residual(left, right)
    return {
        "rendered": rendered,
        "left_clause": left,
        "right_clause": right,
        "audit": a,
        "residual": residual,
        "residual_words": {
            "left": word_at(left, residual["left_offset"]) if residual else None,
            "right": word_at(right, residual["right_offset"]) if residual else None,
        },
        "policy": policy,
        "parent_sha256": parent,
        "provenance": {
            "fresh_model_authored_left": True,
            "fresh_model_authored_right": True,
            "seed_scaffold_in_output": False,
            "finished_tape_reversed": False,
            "catalogue_imported": False,
            "word_order_only": False,
            "reader_certified": False,
        },
    }


def run(model_name: str = "gpt2", per_prompt: int = 4) -> dict:
    policy = GPT2SpanPolicy(model_name=model_name, seed=91019)
    lefts: list[dict] = []
    rights: list[dict] = []
    for prompt in LEFT_PROMPTS:
        for row in policy.propose(prompt, count=per_prompt):
            clause = clean_clause(row["span"])
            if len(clause.split()) >= 4:
                lefts.append({"clause": clause, "proposal": row})
    for prompt in RIGHT_PROMPTS:
        for row in policy.propose(prompt, count=per_prompt):
            clause = clean_clause(row["span"])
            if len(clause.split()) >= 4:
                rights.append({"clause": clause, "proposal": row})
    # Sibling worlds are a cross product of independent authoring calls, not
    # a fixed tape sweep. Keep it bounded and retain every row for replay.
    rows = []
    for li, left in enumerate(lefts):
        for ri, right in enumerate(rights):
            row = pair_row(left["clause"], right["clause"], "dual_boundary_authoring")
            row["sibling_branch_id"] = f"dual-{li}-{ri}"
            row["provenance"].update({
                "left_prompt": left["proposal"]["prompt"],
                "right_prompt": right["proposal"]["prompt"],
                "left_raw_model_text": left["proposal"]["raw_model_text"],
                "right_raw_model_text": right["proposal"]["raw_model_text"],
            })
            rows.append(row)
    scores = policy.score([row["rendered"] for row in rows])
    for row, score in zip(rows, scores):
        row["model_score_per_token"] = score
    rows.sort(key=lambda row: (row["audit"]["mismatch_count"], -row["model_score_per_token"], -row["audit"]["letters"]))
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    best = rows[0] if rows else None
    # Concrete repair queue: the next transition reopens the word owning the
    # first residual, rather than merely adding another sibling score.
    repair = None
    if best and best["residual"]:
        repair = {
            "operator": "reopen_complete_word_at_first_residual",
            "residual": best["residual"],
            "left_word": best["residual_words"]["left"],
            "right_word": best["residual_words"]["right"],
            "required_transition": "generate a replacement clause whose reopened word satisfies the reflected edge character while preserving subject/verb/object valency",
        }
    return {
        "experiment": EXPERIMENT,
        "method": "Dream-RSI dual-boundary model authoring with complete-word residual repair",
        "model_policy": {"model": model_name, "local_files_only": True,
                          "left_prompts": LEFT_PROMPTS, "right_prompts": RIGHT_PROMPTS,
                          "per_prompt": per_prompt, "score_is_diagnostic": True},
        "rendered_candidates": rows,
        "fresh_exact_closures": exact,
        "stats": {"left_proposals": len(lefts), "right_proposals": len(rights),
                  "paired_worlds": len(rows), "exact": len(exact),
                  "longest_letters": max((r["audit"]["letters"] for r in rows), default=0),
                  "best_mismatch_count": best["audit"]["mismatch_count"] if best else None},
        "reader_gate": {"status": "closed", "reason": "No blinded intact/shuffled human ratings; model score is not readability evidence."},
        "next_repair": repair or {"operator": "author_more_boundary_clauses", "reason": "no model clauses survived lexical cleanup"},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "seed_scaffold_in_output": False,
                       "catalogue_imported": False,
                       "human_readability_certified": False},
    }


if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        directory.mkdir(exist_ok=True)
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))
    for row in payload["rendered_candidates"][:5]:
        print(f"{row['audit']['letters']} letters | mismatches={row['audit']['mismatch_count']} | {row['rendered']}")
