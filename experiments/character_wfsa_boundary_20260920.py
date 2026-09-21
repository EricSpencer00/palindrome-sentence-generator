"""Online character-WFSA search with grammatical word-boundary states.

Each side is an independently selected clause.  The transducer emits complete
words from POS arcs, immediately pairing exposed characters from the two
opposing cursors.  A boundary state records which side still has unmatched
characters, so no finished tape is reversed or scored afterward.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/character-wfsa-boundary-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "character-wfsa-boundary-20260920"
SIGNATURE = "online-character-wfsa|pos-arcs|unmatched-boundary-buffer|independent-clause-choice"

# Small checked-in lexicon: word/POS arcs, not catalogue or API text.
ARCS = {
    "det": ("the", "a", "our"),
    "subj": ("sailor", "teacher", "writer", "gardener"),
    "verb": ("guides", "marks", "records", "carries"),
    "objdet": ("the", "a", "our"),
    "obj": ("harbor", "letter", "parcel", "garden"),
    "prep": ("near", "beside", "beyond"),
    "place": ("shore", "station", "garden", "harbor"),
}
PHASES = ("det", "subj", "verb", "objdet", "obj", "prep", "det", "place")

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    tape = letters(text); i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]: i += 1; j -= 1
    return {"letters": len(tape), "pointer_exact": i >= j,
            "first_mismatch": None if i >= j else [i, j, tape[i], tape[j]],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def novelty_preflight() -> dict:
    d = json.loads(REGISTRY.read_text())
    all_entries = [x for x in d.get("entries", []) + d.get("excluded", []) if x.get("id") != EXPERIMENT_ID]
    overlap = [x["id"] for x in all_entries if x.get("signature") == SIGNATURE]
    artifact = str(Path(__file__).relative_to(ROOT))
    collision = [x["id"] for x in all_entries if x.get("artifact") == artifact]
    return {"status": "passed" if not overlap and not collision else "blocked",
            "signature_overlaps": overlap, "artifact_collisions": collision,
            "excluded_routes": ["finished-tape reversal", "post-search repair", "word-order mirror"]}

def render(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    # Independent clause choices are joined as a complete, readable scene.
    return " ".join(left) + "; while " + " ".join(right) + "."

def pair_obligation(left: str, right: str, pending: str) -> tuple[bool, str, int]:
    """Consume opposing characters and return (valid, pending, checks)."""
    a, b = pending + letters(left), letters(right)[::-1]
    n = min(len(a), len(b)); checks = n
    if a[:n] != b[:n]: return False, pending, checks
    return True, a[n:], checks

def run() -> dict:
    pre = novelty_preflight()
    if pre["status"] != "passed": raise RuntimeError(pre)
    rows, controls, rejects = [], [], 0
    # Select POS arcs online. The state is (phase, opposing phase, pending,
    # left words, right words); only a bounded Cartesian product is explored.
    states = [(0, 0, "", (), ())]; transitions = 0
    while states:
        li, ri, pending, left, right = states.pop()
        if li == len(PHASES) and ri == len(PHASES):
            if pending: rejects += 1; continue
            text = render(left, right); a = audit(text)
            rows.append({"rendered": text, "audit": a,
                         "wfsa_state": {"left_phase": li, "right_phase": ri, "pending": ""},
                         "provenance": {"lexicon": "checked-in POS arcs", "online": True,
                                        "independent_left_right_choices": True}})
            continue
        # Advance one arc on each side at a time; each word is consumed before
        # the next POS transition, making boundary ownership explicit.
        if li < len(PHASES) and ri < len(PHASES):
            for lw in ARCS[PHASES[li]]:
                for rw in ARCS[PHASES[ri]]:
                    transitions += 1
                    ok, nxt, checks = pair_obligation(lw, rw, pending)
                    if ok:
                        states.append((li + 1, ri + 1, nxt, left + (lw,), right + (rw,)))
                    else: rejects += 1
    # Intact controls are grammatical surfaces from the same WFSA, not exact
    # candidates and not post-hoc repaired strings.
    for ix in range(4):
        lw = tuple(ARCS[p][ix % len(ARCS[p])] for p in PHASES)
        rw = tuple(ARCS[p][(ix + 1) % len(ARCS[p])] for p in PHASES)
        text = render(lw, rw)
        controls.append({"rendered": text, "audit": audit(text), "reader_eligible": False,
                         "provenance": {"complete_grammar_surface": True, "shuffled_control_source": "WFSA arcs"}})
    exact = [x for x in rows if x["audit"]["letters"] > 38 and x["audit"]["pointer_exact"] and x["audit"]["sha256_forward"] == x["audit"]["sha256_reverse"]]
    out = {"experiment_id": EXPERIMENT_ID,
           "method": "online character WFSA with POS arcs and unmatched word-boundary buffer",
           "config": {"phases": PHASES, "lexicon": "checked-in POS arcs", "post_search_reversal": False,
                      "repair": False, "rlaif": False},
           "stats": {"transitions_checked": transitions, "rejected_transitions": rejects,
                     "rendered_candidates": len(rows), "prose_controls": len(controls),
                     "exact_gt38": len(exact), "max_control_letters": max(x["audit"]["letters"] for x in controls)},
           "rendered_candidates": rows[:20], "prose_controls": controls,
           "exact_candidates": exact, "novelty_preflight": pre,
           "provenance": {"independent_audit": ["two-pointer", "forward/reverse SHA-256"],
                          "catalogue_text": False, "mirrored_units": False, "word_order_symmetry": False,
                          "self_palindromic_units": False},
           "next_operator": "typed agreement arcs that carry number through the boundary state before lexical emission",
           "status": "fresh exact >38 requires human reading" if exact else "no exact >38 closure; grammatical controls retained"}
    RUN.write_text(json.dumps(out, indent=2) + "\n"); return out

if __name__ == "__main__": print(json.dumps(run(), indent=2))
