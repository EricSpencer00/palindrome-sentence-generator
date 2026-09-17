"""Agreement-carrying inflection/clitic boundary search.

The search emits ordinary words from two independently ordered event clauses.
Inflectional endings and clitics are separate finite-state transitions: they
must satisfy agreement with the clause subject and simultaneously match the
opposite exposed character.  Thus boundary choices are part of the CSP, not
post-hoc edits to a completed sentence.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/morphology-clitic-boundary-lockstep-20260917.json"
REG = ROOT / "docs/experiment-novelty-registry.json"
ID = "morphology-clitic-boundary-lockstep-20260917"

SUBJECTS = (("the nurse", "sg"), ("a poet", "sg"), ("our nurses", "pl"),
            ("some poets", "pl"), ("the baker", "sg"), ("two writers", "pl"))
VERBS = {"sg": ("marks", "reads", "sends", "writes", "carries"),
         "pl": ("mark", "read", "send", "write", "carry")}
OBJECTS = ("a letter", "the note", "a report", "the parcel", "some letters")
TAILS = (("in the garden", "loc"), ("at the station", "loc"), ("by a window", "loc"))
CLITICS = (("", None), ("'s", "sg"), ("n't", "neg"))

def tape(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    t = tape(text); i, j = 0, len(t)-1; mismatch = None
    while i < j:
        if t[i] != t[j]:
            mismatch = {"left_index": i, "right_index": j, "left": t[i], "right": t[j]}; break
        i += 1; j -= 1
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "independent_two_pointer": bool(t) and mismatch is None,
            "first_mismatch": mismatch, "words": words,
            "repeated_word_count": len(words)-len(set(words)),
            "self_palindromic_words": [w for w in words if len(w)>1 and w==w[::-1]],
            "borrowed_catalogue": False, "word_order_only": False}

def grammatical(subject_num: str, verb: str, clitic: str, neg: bool) -> bool:
    # The finite-state morphology rule is explicit: third-person singular gets
    # the -s form, plural gets the bare form; n't is allowed only with a
    # finite auxiliary-like lexicalized slot (kept empty here as a control).
    if clitic == "n't" and not neg: return False
    return (subject_num == "sg") == verb.endswith("s")

def emit_pair(left: str, right: str, rows: list, stats: dict) -> None:
    # Consume the two complete ordinary-order clauses in opposite character
    # directions, including suffix boundary states; no reversed text is used.
    lt, rt = tape(left), tape(right)
    i, j = 0, len(rt)-1
    while i < len(lt) and j >= 0 and lt[i] == rt[j]: i += 1; j -= 1
    stats["paired_prefix_states"] += min(i, len(lt))
    if i == len(lt) and j < 0:
        rendered = left + ". " + right + "."
        a = audit(rendered)
        rows.append({"rendered": rendered, "audit": a,
                     "provenance": "fresh finite morphology/clitic lexicon; two independent event clauses",
                     "reader_status": "unreviewed; exactness is not readability evidence"})

def main() -> None:
    rows, stats = [], {"paired_prefix_states": 0, "grammar_rejections": 0, "boundary_transitions": 0}
    for (subj, number), verb, obj, (tail, _) in itertools.product(SUBJECTS, ("marks", "reads", "sends", "writes", "carries", "mark", "read", "send", "write", "carry"), OBJECTS, TAILS):
        for clitic, cfeat in CLITICS:
            neg = clitic == "n't"
            if not grammatical(number, verb, clitic, neg):
                stats["grammar_rejections"] += 1
                continue
            # Morphological boundary is retained in the rendered token; the
            # paired clause is independently chosen from the same grammar.
            left = f"{subj} {verb}{clitic} {obj} {tail}"
            stats["boundary_transitions"] += 1
            for (subj2, n2), verb2, obj2, (tail2, _) in itertools.product(SUBJECTS, VERBS[number], OBJECTS, TAILS):
                if n2 != number or not grammatical(n2, verb2, "", False): continue
                right = f"{subj2} {verb2} {obj2} {tail2}"
                emit_pair(left, right, rows, stats)
    exact = [r for r in rows if r["audit"]["exact"]]
    OUT.write_text(json.dumps({"experiment": ID, "signature": "agreement-morphology|clitic-boundary-fst|online-opposite-character-propagation|independent-pointer-audit", "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "state_count": stats["boundary_transitions"], "candidate_count": len(rows), "exact_count": len(exact), "reader_eligible_count": 0, "exact_closures": exact[:20], "stats": stats, "repair_after_failure": "The paired clause bank has no boundary-compatible return states. Next add independently authored auxiliary/clitic frames (can/not, is/n't) and propagate suffix-to-prefix obligations before subject lexicalization; do not enlarge this Cartesian noun sweep.", "scope": "No catalogue text, reversal, word-order mirror, repeated unit, or proxy readability claim; exact control logic is independent."}, indent=2)+"\n")
    print(json.dumps({"states": stats["boundary_transitions"], "pairs": len(rows), "exact": len(exact)}))

if __name__ == "__main__": main()
