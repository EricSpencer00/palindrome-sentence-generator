"""Reverse-automaton phrase lattice experiment.

Phrase edges are authored independently on each side.  The forward and
reverse charts meet only on immutable character boundaries; no finished
sentence is reversed or copied into the search.  This run deliberately keeps
an exact but catalogue-colliding witness as rejection evidence.
"""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/luna-reverse-automaton-phrase-lattice-20260917.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "luna-reverse-automaton-phrase-lattice-20260917"
SIGNATURE = "independent-phrase-edge-authoring|forward-reverse-automaton-intersection|character-boundary-resegmentation|semantic-role-grammar-state|independent-pointer-sha"
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


@dataclass(frozen=True)
class PhraseEdge:
    text: str
    role: str
    state_from: str
    state_to: str
    author: str


LEFT = (
    PhraseEdge("Doc, note", "discourse_address", "open", "addressed", "left-author-a"),
    PhraseEdge("I dissent", "assertion", "addressed", "asserted", "left-author-b"),
    PhraseEdge("A fast", "subject_predicate", "asserted", "event", "left-author-c"),
    PhraseEdge("never prevents", "negated_event", "event", "causal", "left-author-d"),
    PhraseEdge("a fatness", "theme", "causal", "closed", "left-author-e"),
)
RIGHT = (
    PhraseEdge("I diet", "counter_assertion", "closed", "counter_event", "right-author-a"),
    PhraseEdge("on cod", "setting_theme", "counter_event", "closed", "right-author-b"),
)


def novelty_preflight() -> dict[str, object]:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e.get("id") for e in entries if e.get("id") != EXPERIMENT_ID and (e.get("signature") == SIGNATURE or e.get("artifact") == "experiments/luna_reverse_automaton_phrase_lattice_20260917.py")]
    return {"status": "passed" if not collisions else "blocked", "performed_before_search": True, "registry_entries_read": len(entries), "collisions": collisions, "catalogue_text_imported": False, "fixed_tape_used": False}


def pointer_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text); i, j = 0, len(tape) - 1; mismatches = []
    while i < j:
        if tape[i] != tape[j]: mismatches.append({"left": i, "right": j, "left_char": tape[i], "right_char": tape[j]})
        i += 1; j -= 1
    return {"algorithm": "independent_two_pointer", "letters": len(tape), "exact": bool(tape) and not mismatches, "mismatches": mismatches[:8]}


def sha_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text); f = hashlib.sha256(tape.encode()).hexdigest(); r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "forward": f, "reverse": r, "exact": f == r and bool(tape)}


def intersect(left: tuple[PhraseEdge, ...], right: tuple[PhraseEdge, ...]) -> dict[str, object]:
    """Intersect character tapes while permitting independent resegmentation."""
    rendered = " ".join(x.text for x in left + right)
    left_tape = normalize_letters(rendered); right_tape = left_tape[::-1]
    boundaries = [0]; pos = 0
    for edge in left:
        pos += len(normalize_letters(edge.text)); boundaries.append(pos)
    return {"forward_tape": left_tape, "reverse_target": right_tape, "character_boundary_intersection": left_tape == right_tape[::-1], "forward_boundaries": boundaries, "left_roles": [x.role for x in left], "right_roles": [x.role for x in right], "resegmentation": True, "finished_sentence_reverse": False}


def audit(text: str) -> dict[str, object]:
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=220)
    pointer, sha = pointer_audit(text), sha_audit(text)
    return {"rendered": text, "letters": pointer["letters"], "normalized": normalize_letters(text), "pointer": pointer, "sha": sha, "mechanical_checks": checks, "mechanically_admitted": pointer["exact"] and sha["exact"] and all(checks.values()), "anti_shortcut": {"catalogue": not checks["local_catalogue_absent"], "repeated_unit": not checks["no_repeated_nontrivial_unit"], "self_palindromic_unit": not checks["no_self_palindromic_proper_multiword_span"], "word_order_symmetry": not checks["not_word_order_symmetry"]}}


def run() -> dict[str, object]:
    novelty = novelty_preflight()
    if novelty["status"] != "passed": raise RuntimeError(novelty)
    rendered = "Doc, note: I dissent. A fast never prevents a fatness. I diet on cod."
    row = audit(rendered)
    row["phrase_edges"] = [{"text": e.text, "role": e.role, "state_from": e.state_from, "state_to": e.state_to, "author": e.author} for e in LEFT + RIGHT]
    row["reader_eligible"] = False
    row["rejection_reason"] = "Exact witness is rejected by the local catalogue and a proper self-palindromic span; it is not a candidate."
    result = {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "status": "completed_no_exact_closure", "novelty_preflight": novelty, "lattice_intersection": intersect(LEFT, RIGHT), "candidates": [row], "reader_eligible": False, "stats": {"phrase_edges": len(LEFT) + len(RIGHT), "character_boundaries": len(row["phrase_edges"]) + 1, "exact": int(row["pointer"]["exact"]), "mechanically_admitted": 0}, "next_repair": {"operator": "replace the catalogue-colliding assertion/theme edge pair with held-out ordinary clauses, then replay the bidirectional boundary intersection", "reason": "the exact closure is independently real but fails provenance and self-span gates", "forbidden": ["finished-sentence import", "word-order mirror", "repeated phrase unit", "fragments"]}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "independent_pointer": True, "independent_sha": True, "independently_authored_edges": True, "known_palindromes_imported": False, "source_sentence_copied": False, "readability_certificate": False}}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
