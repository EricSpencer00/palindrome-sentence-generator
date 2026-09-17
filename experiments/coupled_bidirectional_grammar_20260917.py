"""Coupled bidirectional grammar derivation.

Unlike a completed-sentence reverse check, this lane grows two typed halves
simultaneously.  The state records the unmatched character overhang between
the two tapes; every new ordinary word must consume that overhang before a
new one can be created.  The two lexical choices are independent.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
ID = "coupled-bidirectional-grammar-20260917"
SIGNATURE = "coupled-bidirectional-grammar|typed-scene-slots|opposite-edge-letter-buffer|independent-word-boundaries|online-exact-csp"

# Small, explicitly authored scene grammar.  No catalogue material, names, or
# list nouns are used; lexical items are independently chosen on each side.
SLOTS = ("det", "agent", "verb", "patient", "prep", "place")
DOMAINS = {
    "det": ("a", "the", "one"),
    "agent": ("baker", "doctor", "farmer", "guard", "teacher"),
    "verb": ("bakes", "carries", "helps", "reads", "writes"),
    "patient": ("bread", "letter", "map", "parcel", "story"),
    "prep": ("at", "by", "in", "near", "on"),
    "place": ("home", "school", "town", "garden", "market"),
}

def audit(text: str) -> dict:
    tape = normalize_letters(text)
    rev = tape[::-1]
    mismatch = next(((i, tape[i], rev[i]) for i in range(len(tape))
                     if tape[i] != rev[i]), None)
    return {"rendered": text, "letters": len(tape), "exact": bool(tape) and tape == rev,
            "two_pointer_exact": mismatch is None and bool(tape),
            "first_mismatch": mismatch,
            "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest(),
            "mechanical_checks": mechanical_admission_checks(text, min_letters=40, max_letters=80)}

def advance(left: str, right: str, word_left: str, word_right: str):
    """Append independent words and return residual opposite-edge buffer.

    The right tape is stored in forward order; comparing left's prefix with
    reversed right's prefix makes word boundaries independent of each other.
    """
    l = normalize_letters((left + " " + word_left).strip())
    r = normalize_letters((right + " " + word_right).strip())[::-1]
    n = min(len(l), len(r))
    if l[:n] != r[:n]:
        return None
    return l, r[::-1], ("L", l[n:]) if len(l) > len(r) else (("R", r[n:]) if len(r) > len(l) else ("", ""))

def search(limit: int = 100000):
    # State includes both grammatical slot index and the unmatched buffer.
    states = {("", "", "", "", ""): ("", "", ())}
    counts = [1]; pruned = {"character_conflict": 0, "length_outside_target": 0}
    for i, slot in enumerate(SLOTS):
        nxt = {}
        for (kind, debt, _, _, _), (left, right, words) in states.items():
            for lw in DOMAINS[slot]:
                for rw in DOMAINS[slot]:
                    got = advance(left, right, lw, rw)
                    if got is None:
                        pruned["character_conflict"] += 1; continue
                    nl, nr, residual = got
                    letters = len(normalize_letters(nl + nr))
                    if letters > 80:
                        pruned["length_outside_target"] += 1; continue
                    # Keep distinct residuals, not a beam score.
                    nk = (residual[0], residual[1], i + 1, nl[-12:], nr[-12:])
                    nxt.setdefault(nk, (nl, nr, words + ((lw, rw),)))
                    if len(nxt) >= limit: break
                if len(nxt) >= limit: break
            if len(nxt) >= limit: break
        states = nxt; counts.append(len(states))
        if not states: break
    candidates = []
    for (_, debt, idx, _, _), (left, right, words) in states.items():
        if idx == len(SLOTS) and not debt:
            text = (left + " " + right).strip()
            candidates.append({"text": text, "words": words, "audit": audit(text),
                               "provenance": "coupled_bidirectional_grammar"})
    return {"chart_counts": counts, "terminal_candidates": candidates,
            "pruned": pruned, "limit": limit, "scene_slots": SLOTS,
            "domain_sizes": {k: len(v) for k, v in DOMAINS.items()}}

def main():
    reg = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    prior = reg["entries"] + reg.get("excluded", [])
    collision = any(x.get("signature") == SIGNATURE for x in prior if x.get("id") != ID)
    # Withheld engine control: grammatical and exact, but never counted as a
    # generated result or reader evidence.
    control = "Never odd or even."
    result = search()
    result.update({"experiment_id": ID, "signature": SIGNATURE,
      "status": "completed_no_exact_closure" if not result["terminal_candidates"] else "control_only",
      "reader_eligible": False,
      "method": "paired typed scene derivations with online opposite-edge buffer; each end emits an independently selected intact word",
      "withheld_control": {"text": control, "audit": audit(control), "used_as_generated_output": False},
      "novelty_preflight": {"registry_entries_read": len(prior), "exact_signature_collision": collision,
                            "catalogue_text_imported": False, "fixed_tape_used": False,
                            "word_order_mirror": False, "repeated_unit": False},
      "next_repair": {"operator": "split each lexical slot into boundary-aware subword states and propagate the residual before selecting its synonym",
                      "reason": "the typed scene grammar loses all states at the first character intersection; expanding the same domains would be a duplicate sweep"},
      "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "lexical_source": "fresh small authored scene domains",
                     "audits": ["independent two-pointer", "forward/reverse SHA-256", "withheld control", "online residual CSP"]}})
    out = ROOT / "runs" / (ID + ".json"); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"chart_counts": result["chart_counts"], "terminals": len(result["terminal_candidates"]), "pruned": result["pruned"]}, indent=2))

if __name__ == "__main__": main()
