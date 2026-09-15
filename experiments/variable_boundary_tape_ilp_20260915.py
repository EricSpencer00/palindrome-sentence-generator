"""Variable-boundary grammar/tape ILP for exact readable palindromes.

Unlike a slot product or a reverse-language join, this model has one path
through a lexicalized grammar automaton.  A binary arc says that a particular
word starts at a particular character offset and moves the parser to the next
feature state.  Word lengths are therefore variables: the path chooses its
own token boundaries.  The same arc variables contribute characters directly
to both sides of every palindrome equation.

The model is solved with SciPy/HiGHS MILP.  It is deliberately bounded and
diagnostic; a solver solution still needs independent exact auditing and
blinded human readability evidence before promotion.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/variable-boundary-tape-ilp-20260915.json"
ID = "variable-boundary-tape-ilp"
SIGNATURE = (
    "character-tape-flow-ilp|variable-word-boundaries|lexicalized-feature-nfa|"
    "simultaneous-mirror-equations|verb-object-semantic-unification|"
    "length-sweep-not-slot-cross-product"
)
MIN_LETTERS = 39
MAX_LETTERS = 64


@dataclass(frozen=True)
class Word:
    text: str
    category: str
    number: str = ""
    sense: str = ""

    @property
    def tape(self) -> str:
        return self.text.lower()


# These are ordinary, task-authored lexical choices.  Subject number is a
# parser feature and `sense` is carried through the verb->object state.
WORDS = (
    *(Word(x, "DET") for x in ("a", "an", "the", "our", "my", "some", "this", "that")),
    *(Word(x, "ADJ") for x in ("quiet", "kind", "old", "new", "bright", "clear", "small", "brave", "calm", "young")),
    *(Word(x, "SUBJ", "sg") for x in ("man", "woman", "child", "sailor", "artist", "baker", "nurse", "farmer", "poet", "teacher", "writer", "reader", "captain", "friend", "pilot", "guard", "mother", "father")),
    *(Word(x, "SUBJ", "pl") for x in ("men", "women", "children", "sailors", "artists", "bakers", "nurses", "farmers", "poets", "teachers", "writers", "readers", "captains", "friends", "pilots", "guards", "mothers", "fathers")),
    *(Word(x, "VERB", "sg", "read") for x in ("reads",)),
    *(Word(x, "VERB", "pl", "read") for x in ("read",)),
    *(Word(x, "VERB", "sg", "write") for x in ("writes",)),
    *(Word(x, "VERB", "pl", "write") for x in ("write",)),
    *(Word(x, "VERB", "sg", "see") for x in ("sees",)),
    *(Word(x, "VERB", "pl", "see") for x in ("see",)),
    *(Word(x, "VERB", "sg", "carry") for x in ("carries",)),
    *(Word(x, "VERB", "pl", "carry") for x in ("carry",)),
    *(Word(x, "VERB", "sg", "hold") for x in ("holds",)),
    *(Word(x, "VERB", "pl", "hold") for x in ("hold",)),
    *(Word(x, "VERB", "sg", "send") for x in ("sends",)),
    *(Word(x, "VERB", "pl", "send") for x in ("send",)),
    *(Word(x, "VERB", "sg", "plant") for x in ("plants",)),
    *(Word(x, "VERB", "pl", "plant") for x in ("plant",)),
    *(Word(x, "VERB", "sg", "open") for x in ("opens",)),
    *(Word(x, "VERB", "pl", "open") for x in ("open",)),
    *(Word(x, "VERB", "sg", "close") for x in ("closes",)),
    *(Word(x, "VERB", "pl", "close") for x in ("close",)),
    *(Word(x, "VERB", "sg", "make") for x in ("makes",)),
    *(Word(x, "VERB", "pl", "make") for x in ("make",)),
    *(Word(x, "VERB", "sg", "keep") for x in ("keeps",)),
    *(Word(x, "VERB", "pl", "keep") for x in ("keep",)),
    *(Word(x, "VERB", "sg", "like") for x in ("likes",)),
    *(Word(x, "VERB", "pl", "like") for x in ("like",)),
    *(Word(x, "VERB", "sg", "help") for x in ("helps",)),
    *(Word(x, "VERB", "pl", "help") for x in ("help",)),
    *(Word(x, "VERB", "sg", "teach") for x in ("teaches",)),
    *(Word(x, "VERB", "pl", "teach") for x in ("teach",)),
    *(Word(x, "OBJ", sense="read") for x in ("book", "books", "letter", "letters", "note", "notes", "map", "maps", "story", "stories", "poem", "poems")),
    *(Word(x, "OBJ", sense="write") for x in ("book", "books", "letter", "letters", "note", "notes", "story", "stories", "poem", "poems", "plan", "plans")),
    *(Word(x, "OBJ", sense="see") for x in ("man", "woman", "child", "men", "women", "children", "sailor", "boat", "boats", "bird", "birds", "river", "rivers", "garden", "gardens")),
    *(Word(x, "OBJ", sense="carry") for x in ("book", "books", "map", "maps", "letter", "letters", "note", "notes", "box", "boxes", "bag", "bags", "stone", "stones", "lantern", "lanterns")),
    *(Word(x, "OBJ", sense="hold") for x in ("book", "books", "map", "maps", "letter", "letters", "note", "notes", "box", "boxes", "bag", "bags", "stone", "stones", "lantern", "lanterns")),
    *(Word(x, "OBJ", sense="send") for x in ("letter", "letters", "note", "notes", "map", "maps", "parcel", "parcels", "message", "messages")),
    *(Word(x, "OBJ", sense="plant") for x in ("seed", "seeds", "tree", "trees", "flower", "flowers", "rose", "roses")),
    *(Word(x, "OBJ", sense="open") for x in ("book", "books", "door", "doors", "box", "boxes", "gate", "gates", "letter", "letters")),
    *(Word(x, "OBJ", sense="close") for x in ("book", "books", "door", "doors", "box", "boxes", "gate", "gates", "letter", "letters")),
    *(Word(x, "OBJ", sense="make") for x in ("plan", "plans", "meal", "meals", "map", "maps", "poem", "poems", "boat", "boats", "art")),
    *(Word(x, "OBJ", sense="keep") for x in ("book", "books", "map", "maps", "letter", "letters", "note", "notes", "plan", "plans", "secret", "secrets")),
    *(Word(x, "OBJ", sense="like") for x in ("book", "books", "song", "songs", "meal", "meals", "garden", "gardens", "river", "rivers", "poem", "poems")),
    *(Word(x, "OBJ", sense="help") for x in ("child", "children", "friend", "friends", "farmer", "farmers", "sailor", "sailors", "nurse", "nurses")),
    *(Word(x, "OBJ", sense="teach") for x in ("child", "children", "friend", "friends", "reader", "readers", "artist", "artists", "farmer", "farmers")),
    *(Word(x, "PREP") for x in ("in", "on", "by", "near", "under", "with", "from", "at")),
    *(Word(x, "LOC") for x in ("harbor", "harbour", "garden", "window", "bridge", "station", "meadow", "room", "house", "river", "shore", "road", "village", "school", "field", "tower", "valley", "home", "table")),
    *(Word(x, "ADV") for x in ("now", "today", "often", "again", "quietly")),
)

# The first sweep is a small, held-out semantic menu rather than the large
# repository lexicon.  Keeping it explicit makes the ILP auditable and keeps
# the per-length solve bounded; importantly, boundaries remain variable.
LEXICAL_MENU = tuple(
    w for w in WORDS
    if (w.category == "DET" and w.text in {"a", "the", "our"})
    or (w.category == "ADJ" and w.text in {"quiet", "new"})
    or (w.category == "SUBJ" and w.text in {"man", "child", "men", "children"})
    or (w.category == "VERB" and w.sense in {"read", "write", "see", "carry", "send"})
    or (w.category == "OBJ" and w.text in {"book", "letter", "note", "child", "boat", "seed", "parcel"})
    or (w.category == "PREP" and w.text in {"in", "on"})
    or (w.category == "LOC" and w.text in {"garden", "room", "home"})
    or (w.category == "ADV" and w.text in {"now", "again"})
)


def norm(text: str) -> str:
    return "".join(c for c in text.lower() if "a" <= c <= "z")


def independent_audit(text: str) -> dict[str, object]:
    tape = norm(text)
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left": i, "right": j, "left_char": tape[i], "right_char": tape[j]})
        i += 1; j -= 1
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "normalized_tape": tape,
            "pairs_checked": len(tape) // 2, "mismatch_count": len(mismatches),
            "mismatches": mismatches[:3], "sha256": hashlib.sha256(tape.encode()).hexdigest()}


def states_and_transitions():
    """Return an NFA with explicit agreement and verb/object sense states."""
    # States are strings to make the semantic commitments inspectable.
    trans: dict[str, list[tuple[str, str]]] = {}
    def add(src, cat, dst): trans.setdefault(src, []).append((cat, dst))
    add("S", "DET", "SD")
    add("S", "SUBJ:sg", "VSG")
    add("S", "SUBJ:pl", "VPL")
    add("SD", "ADJ", "SA")
    add("SD", "SUBJ:sg", "VSG")
    add("SD", "SUBJ:pl", "VPL")
    add("SA", "SUBJ:sg", "VSG")
    add("SA", "SUBJ:pl", "VPL")
    for num in ("sg", "pl"):
        for sense in sorted({w.sense for w in WORDS if w.category == "VERB"}):
            add("V" + num.upper(), "VERB:" + num + ":" + sense, "O:" + sense)
    for sense in sorted({w.sense for w in WORDS if w.category == "VERB"}):
        add("O:" + sense, "DET", "OD:" + sense)
        add("O:" + sense, "OBJ:" + sense, "TAIL")
        add("OD:" + sense, "ADJ", "OA:" + sense)
        add("OD:" + sense, "OBJ:" + sense, "TAIL")
        add("OA:" + sense, "OBJ:" + sense, "TAIL")
    add("TAIL", "PREP", "P")
    add("P", "DET", "PD")
    add("P", "LOC", "TAILLOC")
    add("PD", "ADJ", "PA")
    add("PD", "LOC", "TAILLOC")
    add("PA", "LOC", "TAILLOC")
    add("TAILLOC", "ADV", "T")
    add("TAIL", "ADV", "T")
    return trans, {"TAIL", "TAILLOC", "T"}


def cat_matches(word: Word, label: str) -> bool:
    parts = label.split(":")
    if parts[0] == word.category:
        return True
    if parts[0] == "SUBJ": return word.category == "SUBJ" and word.number == parts[1]
    if parts[0] == "VERB": return word.category == "VERB" and word.number == parts[1] and word.sense == parts[2]
    if parts[0] == "OBJ": return word.category == "OBJ" and word.sense == parts[1]
    return False


def grammar_probe(length: int) -> str | None:
    """Find one complete ordinary sentence of exactly ``length`` letters.

    This is only a rendered control/probe.  It is intentionally separate from
    the MILP witness so that a timed-out optimization still leaves a concrete
    sentence and an independent audit rather than an empty result file.
    """
    transitions, accepts = states_and_transitions()
    memo: dict[tuple[int, str], tuple[Word, ...] | None] = {}
    def visit(pos: int, state: str):
        key = (pos, state)
        if key in memo: return memo[key]
        if pos == length and state in accepts:
            memo[key] = (); return ()
        if pos >= length:
            memo[key] = None; return None
        for label, dst in transitions.get(state, ()):
            choices = sorted((w for w in LEXICAL_MENU if cat_matches(w, label)), key=lambda w: (w.text, w.sense))
            for word in choices:
                end = pos + len(word.tape)
                if end > length: continue
                tail = visit(end, dst)
                if tail is not None:
                    memo[key] = (word,) + tail; return memo[key]
        memo[key] = None; return None
    answer = visit(0, "S")
    return " ".join(w.text for w in answer) if answer else None


def solve_length(length: int, time_limit: float = 3.0) -> dict[str, object]:
    transitions, accepts = states_and_transitions()
    states = sorted({"S", *accepts, *(q for q, pairs in transitions.items() for _, q in pairs), *(q for q in transitions for _ in [q])})
    sid = {q: i for i, q in enumerate(states)}
    arcs: list[tuple[int, str, Word, int, str]] = []
    # One variable per lexicalized NFA arc at one character offset.
    for start in range(length):
        for src, pairs in transitions.items():
            for label, dst in pairs:
                for word in LEXICAL_MENU:
                    if not cat_matches(word, label): continue
                    end = start + len(word.tape)
                    if end <= length:
                        arcs.append((start, src, word, end, dst))
    sinks = [(length, q) for q in accepts]
    n = len(arcs) + len(sinks)
    idx_sink = {node: len(arcs) + i for i, node in enumerate(sinks)}
    rows = []
    lbs = []
    ubs = []
    # Character equations: each selected lexical arc writes its characters
    # directly into the global tape, including across all word boundaries.
    for p in range(length // 2):
        q = length - 1 - p
        coeff: dict[int, float] = {}
        for k, (start, _, word, end, _) in enumerate(arcs):
            if start <= p < end: coeff[k] = coeff.get(k, 0) + ord(word.tape[p-start])
            if start <= q < end: coeff[k] = coeff.get(k, 0) - ord(word.tape[q-start])
        rows.append(coeff); lbs.append(0); ubs.append(0)
    # Grammar flow: one path starts at S@0 and ends in an accepting state@L.
    nodes = [(p, state) for p in range(length + 1) for state in states]
    for p, state in nodes:
        coeff: dict[int, float] = {}
        for k, (start, src, _, end, dst) in enumerate(arcs):
            if (start, src) == (p, state): coeff[k] = coeff.get(k, 0) + 1
            if (end, dst) == (p, state): coeff[k] = coeff.get(k, 0) - 1
        sink = idx_sink.get((p, state))
        if sink is not None: coeff[sink] = coeff.get(sink, 0) + 1
        rhs = 1 if (p, state) == (0, "S") else 0
        rows.append(coeff); lbs.append(rhs); ubs.append(rhs)
    # Exactly one sink is selected (flow equations already force the matched
    # sink, but this makes the terminal condition explicit and auditable).
    rows.append({i: 1 for i in idx_sink.values()}); lbs.append(1); ubs.append(1)
    A = lil_matrix((len(rows), n), dtype=float)
    for r, coeff in enumerate(rows):
        for c, value in coeff.items(): A[r, c] = value
    # Frequency-like objective is only a tie-break among exact solutions; it
    # never relaxes a character or grammar constraint.
    try:
        from wordfreq import zipf_frequency
        costs = np.array([-zipf_frequency(w.text, "en") for _, _, w, _, _ in arcs] + [0.0] * len(sinks))
    except Exception:
        costs = np.zeros(n)
    res = milp(c=costs, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
               constraints=LinearConstraint(A.tocsr(), np.array(lbs), np.array(ubs)),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.x is None or res.status not in (0, 1):
        probe = grammar_probe(length)
        probe_row = None
        if probe:
            probe_audit = independent_audit(probe)
            probe_row = {"text": probe, "letters": probe_audit["letters"], "normalized_tape": probe_audit["normalized_tape"],
                         "exact_letter_palindrome": probe_audit["exact"], "independent_exact_audit": probe_audit,
                         "provenance": {"method": "variable-boundary lexicalized grammar probe", "solver_status": str(res.message), "is_solver_witness": False},
                         "readability": {"status": "diagnostic_only", "blinded_human_reading_required": True}}
        return {"length": length, "status": str(res.message), "arc_count": len(arcs), "exact_rows": [], "rendered_probe": probe_row}
    chosen = [arcs[i] for i, value in enumerate(res.x[:len(arcs)]) if value > 0.5]
    chosen.sort(key=lambda x: x[0])
    if not chosen or chosen[0][0] != 0:
        return {"length": length, "status": "invalid-flow-witness", "arc_count": len(arcs), "exact_rows": []}
    text = " ".join(w.text for _, _, w, _, _ in chosen)
    audit = independent_audit(text)
    row = {"text": text, "letters": audit["letters"], "normalized_tape": audit["normalized_tape"],
           "exact_letter_palindrome": audit["exact"], "independent_exact_audit": audit,
           "provenance": {"method": "single variable-boundary lexicalized feature-flow ILP",
                          "selected_arcs": [{"start": s, "end": e, "state": src + "->" + dst, "word": w.text, "category": w.category, "sense": w.sense, "number": w.number} for s, src, w, e, dst in chosen],
                          "word_order_was_not_mirrored": True, "catalogue_text_used": False},
           "readability": {"status": "diagnostic_only", "word_count": len(chosen), "blinded_human_reading_required": True}}
    return {"length": length, "status": str(res.message), "arc_count": len(arcs), "exact_rows": [row] if audit["exact"] else [], "best_witness": row}


def run() -> dict[str, object]:
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    prior = [r for r in registry["entries"] if r["id"] != ID]
    assert SIGNATURE not in {r["signature"] for r in prior}
    assert "experiments/variable_boundary_tape_ilp_20260915.py" not in {r["artifact"] for r in prior}
    # Two representative lengths keep the initial evidence bounded.  The
    # method accepts every integer in the configured interval; widening the
    # sweep is the held-out repair after inspecting these first runs.
    lengths = [MIN_LETTERS]
    # Keep this first pass intentionally bounded; a later repair may widen
    # the sweep only after inspecting which lengths have feasible grammar
    # paths.  The model is still exact at every attempted length.
    results = [solve_length(n, time_limit=0.20) for n in lengths]
    exact = [r for r in results if r.get("exact_rows")]
    return {"status": "completed_variable_boundary_tape_ilp", "config": {"lengths": lengths, "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "solver": "scipy.optimize.milp/HiGHS", "variable_word_boundaries": True, "grammar_path_not_fixed_slots": True, "lexical_menu_size": len(LEXICAL_MENU), "semantic_constraints": ["subject number agreement", "verb valency/sense selects object lexicon", "complete NP/VP/PP grammar"]}, "solver_summary": {"length_runs": len(results), "arc_count_total": sum(r.get("arc_count", 0) for r in results), "exact_solution_count": len(exact)}, "results": results, "exact_candidates": exact, "reader_gate": "No programmatic score certifies readability; any exact row requires randomized blinded intact-prose/shuffled-control reading.", "novelty_audit": {"registry_entries_read_before_run": len(prior), "prior_signatures_overlap": [], "artifact": "experiments/variable_boundary_tape_ilp_20260915.py", "self_entry_present": False, "replayed_families": [], "registry_sha256": hashlib.sha256((ROOT / "docs/experiment-novelty-registry.json").read_bytes()).hexdigest()}, "reader_facing_next_operator": "If no exact closure, enlarge the lexicalized semantic automaton with a held-out content-word bank while retaining variable boundary flow; do not relax exact equations.", "scope": "Bounded construction experiment; exact rows are still unreviewed until blinded human readers."}


if __name__ == "__main__":
    if OUT.exists(): raise SystemExit(f"output exists: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = run()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "exact": len(payload["exact_candidates"]), "arc_count_total": payload["solver_summary"]["arc_count_total"]}, indent=2))
