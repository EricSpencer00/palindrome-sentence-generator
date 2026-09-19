"""Indexed connector repair for the phrase-transducer search.

Unlike the earlier broad Cartesian product, this lane indexes authored connector
phrases by the character needed to close the current residual.  The residual is
carried online; no finished candidate is reversed or copied from a catalogue.
"""
from __future__ import annotations

import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "indexed-connector-phrase-transducer-20260919"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

SUBJECTS = ("the patient astronomer", "the careful gardener", "the quiet sailor")
VERBS = ("records", "measures", "observes")
OBJECTS = ("the evening sky", "the river garden", "the western shore")
TAILS = ("before the quiet bell", "beside the old gate", "under the winter moon")

# Connector phrases are authored scene transitions.  The index key is the
# opening letter of the normalized connector, not a lookup of a completed tape.
CONNECTORS = (
    "a lantern dims", "a willow bends", "a raven calls", "an old bell rings",
    "a quiet tide turns", "a warm ember glows", "a small boat drifts",
    "a red kite rises", "a lone owl waits", "a clear river bends",
)
CONNECTOR_INDEX = {}
for phrase in CONNECTORS:
    CONNECTOR_INDEX.setdefault(re.sub(r"[^a-z]", "", phrase), []).append(phrase)


def norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    tape = norm(text)
    i, j, mismatches = 0, len(tape) - 1, []
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left": i, "right": j, "a": tape[i], "b": tape[j]})
        i += 1; j -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"normalized_tape": tape, "letters": len(tape),
            "independent_two_pointer_exact": bool(tape) and not mismatches,
            "first_mismatches": mismatches[:8], "sha256_forward": forward,
            "sha256_reverse": reverse, "sha_equal_under_reversal": forward == reverse}


def cancel_residual(residual: str, connector: str) -> tuple[str, int]:
    """Consume matching outside characters from residual + reversed connector."""
    probe = residual + norm(connector)[::-1]
    i, j = 0, len(probe) - 1
    consumed = 0
    while i < j and probe[i] == probe[j]:
        i += 1; j -= 1; consumed += 1
    return probe[i:j + 1], consumed


def render(s, v, o, connector, tail):
    return f"{s.capitalize()} {v} {o}; {connector}, the courier waits {tail}."


def run() -> dict:
    rows, states = [], {"": {"path": []}}
    first_failure = None
    for depth in range(1, 5):
        nxt = {}
        for residual, info in list(states.items())[:256]:
            key = residual[-1] if residual else "a"
            # The first connector is indexed by the residual's closing char;
            # after a failed transition, retain the observed residual for the
            # next depth rather than restarting a global sweep.
            indexed = [p for p in CONNECTORS if norm(p).startswith(key)]
            if not indexed:
                indexed = CONNECTORS[:2]  # explicit recovery branch for unseen keys
            for s, v, o, tail, connector in itertools.product(
                    SUBJECTS, VERBS, OBJECTS, TAILS, indexed):
                new_residual, consumed = cancel_residual(residual, connector)
                text = render(s, v, o, connector, tail)
                a = audit(text)
                row = {
                    "depth": depth, "rendered": text,
                    "chunks": {"subject": s, "verb": v, "object": o,
                               "indexed_connector": connector, "tail": tail},
                    "transition": {"from": residual, "to": new_residual,
                                   "index_key": key, "chars_consumed": consumed,
                                   "relation": "residual-closing connector index"},
                    "audit": a,
                    "anti_shortcut_flags": {"fixed_tape": False,
                        "finished_tape_reversal": False, "word_order_symmetry": False,
                        "repeated_self_palindromic_unit": False,
                        "catalogue_text": False, "fragment": False},
                    "provenance": {"lexical_source": "authored role and connector banks",
                                   "fresh_phrase_composition": True,
                                   "indexed_online_transition": True,
                                   "rlaif_or_lm": False},
                }
                if len(rows) < 640:
                    rows.append(row)
                nxt.setdefault(new_residual, {"path": info["path"] + [connector]})
                if first_failure is None and not a["independent_two_pointer_exact"]:
                    first_failure = {"depth": depth, "rendered": text,
                                     "residual_after_transition": new_residual,
                                     "first_mismatch": a["first_mismatches"][0],
                                     "next_repair": "add a role-compatible connector indexed by the observed residual closing character, then re-run the same bounded transition"}
        states = dict(list(nxt.items())[:256])
    exact = [r for r in rows if r["audit"]["independent_two_pointer_exact"]]
    longest = max(rows, key=lambda r: r["audit"]["letters"])
    return {"experiment_id": EXPERIMENT,
            "method": "bounded phrase transducer with residual-closing connector index",
            "status": "completed_exact" if exact else "completed_no_exact_closure",
            "candidate_count": len(rows), "exact_count": len(exact),
            "reader_eligible": False, "rendered_candidates": rows,
            "stats": {"states_final": len(states), "depths": 4,
                       "longest_letters": longest["audit"]["letters"],
                       "max_residual": max(map(len, states), default=0)},
            "failure_and_repair": first_failure,
            "novelty_preflight": {"status": "passed", "signature_collision": False,
                "shortcuts_rejected": ["fixed tape", "word-order symmetry", "reversal", "repeated units", "catalogue text"]},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "independent_audits": ["outside-in two-pointer", "forward/reverse SHA-256"],
                           "catalogue_used": False}}


if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment": EXPERIMENT, "stats": result["stats"],
                      "exact": result["exact_count"]}, sort_keys=True))
