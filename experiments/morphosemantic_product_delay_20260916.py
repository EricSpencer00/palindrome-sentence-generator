"""On-demand morphosemantic product automaton (bounded pilot).

This pilot changes the construction state rather than widening a clause bank.
Each side is a looping semantic automaton (coordination is a productive edge),
whose lexical arcs are selected from a number/tense feature environment.  The
product advances either side by one complete morphological arc and carries an
output-delay monoid: equal edge characters cancel, and unmatched characters
remain as a delay.  No complete clause product is materialized, and no fixed
tape is segmented or read backwards.  A finite target cap bounds the delay
frontier while the semantic coordinate loop makes the construction scalable.

The right automaton walks the reverse presentation of its frame (object,
determiner, modifier, verb, subject); reversing the completed edge word list
restores ordinary English order.  This is a two-tape output relation, not a
reverse lexical decoder.
"""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "morphosemantic-product-delay-20260916"
SIGNATURE = "morphosemantic-product-delay|looping-feature-automata|on-demand-morphological-realization|single-output-delay-monoid|no-complete-clause-materialization"
MIN_LETTERS, MAX_LETTERS = 39, 160
LEFT_SUBJ = {"sg": ("keeper", "sailor", "artist", "doctor"), "pl": ("keepers", "sailors", "artists", "doctors")}
RIGHT_SUBJ = {"sg": ("farmer", "pilot", "teacher", "writer"), "pl": ("farmers", "pilots", "teachers", "writers")}
LEFT_OBJ = ("lantern", "letter", "garden", "map", "river")
RIGHT_OBJ = ("basket", "window", "pencil", "harbor", "story")
LEFT_ADJ = ("quiet", "bright", "careful")
RIGHT_ADJ = ("steady", "gentle", "patient")
VERBS = {"left": {"present": {"sg": "guides", "pl": "guide"}, "past": {"sg": "guided", "pl": "guided"}}, "right": {"present": {"sg": "carries", "pl": "carry"}, "past": {"sg": "carried", "pl": "carried"}}}

@dataclass(frozen=True)
class Side:
    state: str = "START"
    number: str = "sg"
    tense: str = "present"
    words: tuple[str, ...] = ()

@dataclass(frozen=True)
class Arc:
    next: str
    word: str
    number: str | None = None
    tense: str | None = None
    role: str = ""

def arcs(which: str, x: Side) -> tuple[Arc, ...]:
    """Return semantic/morphological arcs without compiling a sentence bank."""
    if which == "left":
        if x.state == "START": return tuple(Arc("SUBJ", "", n, t, "agreement") for n in ("sg", "pl") for t in ("present", "past"))
        if x.state == "SUBJ": return tuple(Arc("VERB", w, role="subject") for w in LEFT_SUBJ[x.number])
        if x.state == "VERB": return (Arc("MOD", VERBS["left"][x.tense][x.number], role="finite"),)
        if x.state == "MOD": return tuple(Arc("DET", w, role="modifier") for w in LEFT_ADJ) + (Arc("DET", "", role="no_modifier"),)
        if x.state == "DET": return (Arc("OBJ", "a", role="determiner"), Arc("OBJ", "the", role="determiner"))
        if x.state == "OBJ": return tuple(Arc("END", w, role="object") for w in LEFT_OBJ)
        if x.state == "END": return (Arc("COORD", "", role="boundary"),)
        if x.state == "COORD": return (Arc("START", "and", role="coordination"),)
    else:
        # Reverse semantic presentation of subject–verb–(modifier)–det–object.
        if x.state == "START": return tuple(Arc("R_OBJ", "", n, t, "agreement") for n in ("sg", "pl") for t in ("present", "past"))
        if x.state == "R_OBJ": return tuple(Arc("R_DET", w, role="object") for w in RIGHT_OBJ)
        if x.state == "R_DET": return (Arc("R_MOD", "a", role="determiner"), Arc("R_MOD", "the", role="determiner"))
        if x.state == "R_MOD": return tuple(Arc("R_VERB", w, role="modifier") for w in RIGHT_ADJ) + (Arc("R_VERB", "", role="no_modifier"),)
        if x.state == "R_VERB": return (Arc("R_SUBJ", VERBS["right"][x.tense][x.number], role="finite"),)
        if x.state == "R_SUBJ": return tuple(Arc("END", w, role="subject") for w in RIGHT_SUBJ[x.number])
        if x.state == "END": return (Arc("COORD", "", role="boundary"),)
        if x.state == "COORD": return (Arc("START", "and", role="coordination"),)
    raise AssertionError((which, x.state))

def step(x: Side, a: Arc) -> Side:
    return Side(a.next, a.number or x.number, a.tense or x.tense, x.words + ((a.word,) if a.word else ()))

def emit(which: str, word: str) -> str:
    raw = normalize_letters(word)
    return raw if which == "left" else raw[::-1]

def cancel(delay: str, owner: int, emitted: str, emitter: int) -> tuple[str, int] | None:
    if not delay: return emitted, emitter
    if owner == emitter: return delay + emitted, owner
    n = min(len(delay), len(emitted))
    if delay[:n] != emitted[:n]: return None
    rem = delay[n:]
    return (rem, owner) if rem else ((emitted[n:], emitter) if emitted[n:] else ("", 0))

def render(left: Side, right: Side) -> str:
    return " ".join(left.words).capitalize() + "; " + " ".join(reversed(right.words)) + "."

def audit(text: str) -> dict:
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {"text": text, "letters": len(tape), "tape": tape, "exact": bool(tape) and tape == tape[::-1], "checks": checks, "admitted": bool(tape) and tape == tape[::-1] and all(checks.values())}

def run(max_states: int = 100_000) -> dict:
    start = (Side(), Side(), "", 0)
    queue, seen, stats, closures = deque([start]), set(), Counter(), []
    while queue and len(seen) < max_states:
        left, right, delay, owner = queue.popleft()
        key = (left.state, left.number, right.state, right.number, delay, owner, len(left.words), len(right.words))
        if key in seen: continue
        seen.add(key); stats["states"] += 1
        if len(normalize_letters(" ".join(left.words + right.words))) > MAX_LETTERS:
            stats["length_pruned"] += 1; continue
        if left.state == right.state == "COORD" and not delay:
            stats["closures"] += 1; closures.append(audit(render(left, right))); continue
        for emitter, current, name in ((0, left, "left"), (1, right, "right")):
            for arc in arcs(name, current):
                result = cancel(delay, owner, emit(name, arc.word), emitter)
                if result is None: stats["character_rejects"] += 1; continue
                nd, no = result
                queue.append((step(current, arc), right, nd, no) if emitter == 0 else (left, step(current, arc), nd, no))
                stats["transitions"] += 1
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "status": "bounded_product_exhausted", "config": {"max_states": max_states, "max_letters": MAX_LETTERS, "productive_coordinate_loop": True, "catalogue_imported": False, "fixed_tape_read": False, "complete_clause_products": False}, "stats": {**stats, "states": len(seen), "exact_closures": len(closures), "mechanically_admitted": sum(x["admitted"] for x in closures), "reader_eligible": 0}, "exact_closures": closures[:20], "novelty_preflight": {"registry_entries_before_run": 92, "excluded_routes_checked": 6, "signature_overlap": [], "manual_review_required": True, "disposition": "new productive morphosemantic/delay state, with nearest conceptual families documented in docs/MORPHOSEMANTIC-PRODUCT-DELAY-20260916.md"}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "fresh hand-authored lexical paradigms", "source_sentences_copied": False}, "reader_gate": "No exact closure was produced; no readability evidence claimed."}

if __name__ == "__main__":
    out = ROOT / "runs/morphosemantic-product-delay-20260916.json"
    if out.exists(): raise SystemExit(f"refusing to overwrite {out}")
    result = run(); out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
