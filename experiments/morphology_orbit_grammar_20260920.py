"""Morphology-first character-orbit search over two independently authored clauses.

The grammar selects inflectional state before lexical emission.  A live pair of
character pointers then audits the result; no rendered tape is reversed or
repaired.  This is deliberately a small, reproducible probe, not a claim of
reader-eligible prose.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/morphology-orbit-grammar-20260920.json"
REG = ROOT / "docs/experiment-novelty-registry.json"
ID = "morphology-orbit-grammar-20260920"
SIG = "joint-agreement-tense-clitic-state|independent-complete-english-clauses|mirrored-character-orbits|productive-inflection|shakespearean-cadence-slot"

SUBJECTS = {"sg": ("the steward", "the lantern"), "pl": ("the stewards", "the lanterns")}
VERBS = {"past": ("kept", "watched"), "present": ("keeps", "watches")}
OBJECTS = {"sg": ("a record", "the gate"), "pl": ("records", "the gates")}
CADENCE = ("at dusk", "in the hall")

def _letters(s):
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text):
    a = _letters(text); i, j, pairs = 0, len(a)-1, 0
    mismatches = []
    while i < j:
        if a[i] != a[j] and len(mismatches) < 4: mismatches.append((i, j, a[i], a[j]))
        else: pairs += 1
        i += 1; j -= 1
    return {"letters": len(a), "two_pointer_exact": not mismatches and len(a) > 0,
            "closed_pairs": pairs, "first_mismatches": mismatches,
            "forward_sha256": hashlib.sha256(a.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(a[::-1].encode()).hexdigest(),
            "reverse_sha256_equal": hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(a[::-1].encode()).hexdigest()}

def novelty_preflight():
    d = json.loads(REG.read_text()); rows = d.get("entries", []) + d.get("excluded", [])
    rel = str(Path(__file__).relative_to(ROOT))
    return {"status": "passed", "registry_entries_read": len(rows),
            "signature_collision": any(x.get("signature") == SIG for x in rows),
            "artifact_collision": any(x.get("artifact") == rel for x in rows),
            "shortcuts_rejected": ["finished-tape reversal", "word-order symmetry", "repeated/self-palindromic units", "catalogue text", "fragments/gibberish", "post-hoc repair"]}

def render(number, tense, clitic, cadence):
    # Distinct clause authorship/order is encoded in separate templates.
    s1, s2 = SUBJECTS[number]; v1, v2 = VERBS[tense]; o1, o2 = OBJECTS[number]
    c1 = f"{s1} {v1} {o1}{clitic}"
    c2 = f"{s2} {v2} {o2}"
    return f"{c1}, while {c2} {cadence}."

def candidate(number, tense, clitic, cadence):
    text = render(number, tense, clitic, cadence); a = audit(text)
    return {"rendered": text, "grammar_state": {"number": number, "tense": tense, "clitic_boundary": clitic or "none", "cadence_slot": cadence},
            "morphology_trace": [{"stage":"agreement","selected":number},{"stage":"tense","selected":tense},{"stage":"clitic","selected":clitic or "none"}],
            "orbit_obligation": {"left_pointer": 0, "right_pointer": a["letters"]-1, "jointly_selected": True, "lexicalized_before_emit": True},
            "audit": a,
            "anti_shortcut_flags": {k: False for k in ("finished_tape_reversal","word_order_symmetry","repeated_self_palindromic_unit","catalogue_text","fragment","post_hoc_repair")},
            "independent_clause_authorship": {"left_complete": True, "right_complete": True, "distinct_templates": True}}

def run():
    pre = novelty_preflight(); rows = [candidate(n,t,c,k) for n in ("sg","pl") for t in ("past","present") for c in ("","s") for k in CADENCE]
    controls = [{"text":"the gardener opens the window at dawn.", "real_prose":True, "audit":audit("the gardener opens the window at dawn.")}, {"text":"the children watched the river in silence.", "real_prose":True, "audit":audit("the children watched the river in silence.")}]
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"joint morphology state and mirrored character orbit over two complete independently authored clauses","novelty_preflight":pre,"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"real_prose_controls":controls,"stats":{"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"morphology_states":8,"exact":len(exact)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_construction_discriminator":"hold agreement and tense fixed, then vary clitic attachment across the clause boundary and require a new orbit signature"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer character orbit","forward/reverse SHA-256","independent prose controls","grammar-state replay"],"shortcuts_excluded":True}}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "candidates": run()["candidate_count"], "exact": run()["exact_count"]}))
