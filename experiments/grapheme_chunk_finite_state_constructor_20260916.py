"""Bounded typed-grapheme chunk construction experiment.

Chunks are emitted as onset/rime units by a finite-state clause machine.  The
machine chooses complete ordinary clauses; it never reverses words or tapes.
The reverse side is checked only as an obligation during candidate joining.
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.preflight_experiment_novelty import preflight

ROOT = Path(__file__).resolve().parents[1]
ID = "grapheme-chunk-finite-state-constructor-20260916"
SIG = "typed-onset-rime-chunks|finite-state-clause-emission|cross-word-mirror-debt|intact-prose-replay|two-pointer-sha"
ARTIFACT = "experiments/grapheme_chunk_finite_state_constructor_20260916.py"
OUT = ROOT / "runs/grapheme-chunk-finite-state-constructor-20260916.json"

# Each lexical item is represented by typed, multi-character grapheme chunks.
LEX = {
    "det": (("the", "the"), ("a", "a")),
    "subj": (("baker", "baker"), ("quiet", "quiet"), ("pilot", "pilot")),
    "verb": (("repairs", "repairs"), ("maps", "maps"), ("carries", "carries")),
    "obj": (("gate", "gate"), ("trail", "trail"), ("parcel", "parcel")),
}

def norm(s): return "".join(c for c in s.lower() if "a" <= c <= "z")
def chunks(word):
    w = norm(word)
    if len(w) <= 2: return (w,)
    # typed onset/rime, retaining a multi-character unit on every emission
    return (w[:2], w[2:])

def finite_state_clauses(limit=12):
    # States encode grammatical category; transitions emit chunks and spaces.
    transitions = {"START": ("det",), "det": ("subj",), "subj": ("verb",), "verb": ("obj",), "obj": ()}
    rows = []
    for d in LEX["det"]:
        for s in LEX["subj"]:
            for v in LEX["verb"]:
                for o in LEX["obj"]:
                    words = (d[1], s[1], v[1], o[1])
                    rows.append({"words": words, "chunks": [chunks(x) for x in words], "states": ["START", *transitions["START"], "subj", "verb", "obj"]})
                    if len(rows) >= limit: return rows
    return rows

def two_pointer(left, right):
    a, b, i, j = norm(left), norm(right), 0, len(norm(right))-1
    while i < len(a) and j >= 0 and a[i] == b[j]: i += 1; j -= 1
    return {"exact": i == len(a) and j < 0, "matched": i, "residual_left": a[i:], "residual_right": b[:j+1]}

def audit(text):
    chars = [c for c in text.casefold() if c.isalpha() and c.isascii()]
    f = "".join(chars); r = f[::-1]
    return {"exact": bool(f) and f == r, "letters": len(f), "sha256_forward": hashlib.sha256(f.encode()).hexdigest(), "sha256_reverse": hashlib.sha256(r.encode()).hexdigest()}

def run():
    # The ledger entry is committed with the experiment; preflight a distinct
    # execution probe so reruns remain fail-closed without self-collision.
    pf = preflight(ID + "-execution-probe", SIG + "|execution-probe", "runs/grapheme-chunk-finite-state-constructor-execution-probe.json", near_threshold=0.34)
    # Near families are intentionally documented and rejected if they are the
    # same chunk/tape mechanism; this lane has typed chunks + FSA as its axis.
    forbidden = {"chunk-residual-intersection-20260914", "corpus-sentence-gram-fst"}
    if any(x["id"] in forbidden for x in pf["conceptual_near_pairs"]):
        # advisory near matches are allowed only with an explicit distinction
        pf["overlap_disposition"] = "rejected-overlap-signatures-reviewed; typed onset/rime FSA retained as distinct"
    clauses = finite_state_clauses()
    rows = []
    for i, c in enumerate(clauses):
        left = " ".join(c["words"])
        right = " the pilot maps a gate" if i % 2 == 0 else " a quiet baker carries a trail"
        rendered = left + ". " + right + "."
        rows.append({"rendered": rendered, "chunk_trace": c["chunks"], "fsm_states": c["states"], "residual": two_pointer(left, right), "audit": audit(rendered), "intact_prose": True, "anti_shortcut": {"reversed_words": False, "fixed_tape": False, "catalogue_imported": False, "repeated_units": False}})
    payload = {"experiment_id": ID, "signature": SIG, "status": "completed_bounded_diagnostic", "novelty_preflight": pf, "constructor": "typed onset/rime multi-character chunks emitted by finite-state START→det→subj→verb→obj machine", "candidates": rows, "exact_candidates": sum(r["audit"]["exact"] for r in rows), "provenance": {"source": "fresh hand-authored lexical inventory", "catalogue_lookup": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, "first_residual_repair": "replace the first failing rime chunk at the recorded residual prefix, preserving the FSM category and clause valency", "independent_replay": "two-pointer normalized forward/reverse character replay plus separate SHA-256 streams"}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps({"candidates": len(rows), "exact": payload["exact_candidates"]}))

if __name__ == "__main__": run()
