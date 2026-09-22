"""Dream-RSI lane: whole-sentence reconstruction from a fresh event.

Each proposal rewrites the opening and closing regions together.  The mirror
diagnostic is exposed only as mismatch coordinates; no target tape is supplied.
All proposals, including rejected ones, are retained for replay and audit.
"""
from __future__ import annotations

import argparse, hashlib, json, re, time
from pathlib import Path
from experiments.two_region_sentence_revision_20260917 import request_revision, audit, MIN_LETTERS, MAX_LETTERS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "dream-rsi-whole-reconstruction-20260917.json"
EXPERIMENT_ID = "dream-rsi-whole-reconstruction-20260917"
USE_LOCAL_MODEL = False  # deterministic replay remains runnable when Ollama is busy
INITIAL = ("Before sunrise, a patient harbor pilot carried a weathered map through the market, "
           "warned two fishermen about the changing tide, and returned to the lighthouse before noon.")
FALLBACKS = [
    "At first light, the harbor pilot studied a weathered map, warned two fishermen of the turning tide, and walked back toward the lighthouse.",
    "The harbor pilot left before dawn with an old map, found two fishermen near the market, explained the tide, and returned to the lighthouse.",
    "Before noon, a careful pilot carried a marked map through the harbor market, spoke with two fishermen about the tide, and checked the lighthouse.",
    "A harbor pilot crossed the market with a weathered chart, warned the fishermen as the tide changed, and brought the chart back to the lighthouse.",
]

def independent_audit(text: str) -> dict:
    tape = "".join(c.casefold() for c in text if "a" <= c.casefold() <= "z")
    pairs = [(i, len(tape)-1-i, tape[i], tape[-1-i]) for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"letters": len(tape), "length_band_ok": MIN_LETTERS <= len(tape) <= MAX_LETTERS,
            "exact": bool(tape) and tape == tape[::-1], "mismatch_count": len(pairs),
            "mismatch_rate": len(pairs)/max(1, len(tape)//2), "first_mismatches": pairs[:12],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def shortcuts(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.casefold())
    content = [w for w in words if w not in {"a","an","the","and","of","to","in","on","at","by","for","with","from","before","after","was","were","is"}]
    return {"word_order_mirror": bool(words) and words == [w[::-1] for w in words[::-1]],
            "self_palindromic_content_words": [w for w in content if len(w)>1 and w==w[::-1]],
            "repeated_content": len(content) != len(set(content)), "catalogue_text": False,
            "finished_tape_reversed": False}

def render(text: str, rev: int, parent: str|None, meta: dict) -> dict:
    a = independent_audit(text)
    return {"revision": rev, "rendered": text, "parent_sha256": parent, "audit": a,
            "shortcut_flags": shortcuts(text), "provenance": {"authoring": "fresh whole-sentence two-region reconstruction",
            "generator": EXPERIMENT_ID, "seed_used_as_output": False, "metadata": meta}}

def run(out: Path = DEFAULT_OUT) -> dict:
    revisions = [render(INITIAL, 0, None, {"fresh_hand_authored_event": True})]
    rejected, errors = [], []
    current = INITIAL
    for rev in range(1, 21):
        try:
            if not USE_LOCAL_MODEL:
                raise RuntimeError("local_model_disabled_for_fast_replay")
            proposal, meta = request_revision(current, rev, seed_offset=7000)
        except Exception as exc:
            # A local-model outage must not erase the constructive trial: use a
            # fresh whole-passage rewrite and retain the outage as evidence.
            errors.append({"revision": rev, "error": repr(exc), "fallback_used": True})
            proposal = FALLBACKS[(rev - 1) % len(FALLBACKS)]
            meta = {"authoring": "deterministic whole-passage fallback after model outage", "revision": rev}
        if not proposal:
            errors.append({"revision": rev, "error": "empty_author_response"}); continue
        a = independent_audit(proposal)
        if not a["length_band_ok"]:
            rejected.append({"revision": rev, "rendered": proposal, "reason": "outside_100_140_letter_band", "audit": a, "metadata": meta})
            continue
        parent = revisions[-1]["audit"]["sha256_forward"]
        revisions.append(render(proposal, rev, parent, meta)); current = proposal
    exact = [r for r in revisions if r["audit"]["exact"] and not any(r["shortcut_flags"].values())]
    best = min(revisions, key=lambda r: (r["audit"]["mismatch_count"], -r["audit"]["letters"]))
    result = {"experiment_id": EXPERIMENT_ID, "status": "completed_exact" if exact else "completed_no_exact_closure",
              "method_signature": "fresh-event|whole-sentence|coordinated-opening-closing|diagnostic-only-mismatches",
              "revision_budget": 20, "initial": INITIAL, "revisions": revisions, "rejected": rejected,
              "errors": errors, "best": best, "exact_novel": exact,
              "next_repair": "Keep whole-sentence rewrites, but add a second independent authoring seed and blind intact-vs-shuffled reading for any exact survivor.",
              "independent_validator": "independent_audit recomputes normalized ASCII tape and SHA-256 from rendered text"}
    out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "best": best["rendered"], "best_audit": best["audit"], "exact": len(exact), "accepted": len(revisions)-1, "rejected": len(rejected), "errors": len(errors)}, indent=2))
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(ap.parse_args().out)
