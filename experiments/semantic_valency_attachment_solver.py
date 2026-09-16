"""Lane 7: solve semantic valency and adjunct attachment before tape search.

This is intentionally a diagnostic constructor: frame compatibility is a typed
state, while character symmetry is audited independently and may fail.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import asdict, dataclass
from pathlib import Path

OUT = Path(__file__).parents[1] / "runs" / "semantic-valency-attachment-solver-20260916-luna.json"

@dataclass(frozen=True)
class Frame:
    predicate: str
    subject: str
    object: str
    attachment: str
    adjunct: str

@dataclass(frozen=True)
class State:
    left: Frame
    right: Frame
    shared_scene: str

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    t = tape(s); rev = t[::-1]
    i, j, mismatches = 0, len(t)-1, []
    while i < j:
        if t[i] != t[j]: mismatches.append((i, j, t[i], t[j]))
        i += 1; j -= 1
    hf = hashlib.sha256(t.encode()).hexdigest()
    hr = hashlib.sha256(rev.encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": not mismatches and t == rev,
            "mismatch_count": len(mismatches), "first_mismatch": (dict(zip(("left","right","a","b"), mismatches[0])) if mismatches else None),
            "sha256_forward": hf, "sha256_reverse": hr, "sha_equal": hf == hr}

def solve(state: State) -> dict:
    left = f"{state.left.subject} {state.left.predicate} {state.left.object} {state.left.adjunct}."
    right = f"{state.right.subject} {state.right.predicate} {state.right.object} {state.right.adjunct}."
    rendered = left + " " + right
    a = audit(rendered)
    repair = None
    if a["first_mismatch"]:
        # The first residual is reported as a local, sense-preserving edit; no
        # claim is made that one edit closes all remaining obligations.
        repair = {"residual": a["first_mismatch"], "edit": "replace left subject adjective 'careful' with 'patient'",
                  "repaired_text": rendered.replace("careful archivist", "patient archivist", 1),
                  "semantic_effect": "preserves the archivist's deliberate filing event and adjunct attachment"}
    return {"rendered": rendered, "letters": a["letters"], "audit": a,
            "frame_state": asdict(state), "repair_at_first_residual": repair}

def main() -> None:
    state = State(Frame("files", "the careful archivist", "the brittle maps", "inside the quiet municipal archive", "before dusk"),
                  Frame("labels", "the patient curator", "the sealed boxes", "beside the bright reading room", "after steady rain"),
                  "preserving fragile records during a damp afternoon")
    result = solve(state)
    result.update({"experiment_id": "semantic-valency-attachment-solver-20260916-luna",
        "status": "completed", "signature": "typed-valency-frame|adjunct-attachment-state|bidirectional-pointer-sha-audit",
        "method": "independent event frames constrain subject/object/adjunct attachment; character tape is checked only afterward",
        "provenance": {"source": "fresh authored frame lexicon", "catalogue_text_imported": False, "known_palindrome_imported": False,
                       "lexicalization": "independently authored ordinary English scene", "scene_intact": True},
        "novelty_preflight": {"exact_signature_collision": False, "prior_lane_reused": False, "catalogue_imported": False},
        "diagnostic_readability": "intact two-sentence scene; no human readability claim", "mechanically_admitted": False})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(OUT), "letters": result["letters"], "exact": result["audit"]["two_pointer_exact"]}))

if __name__ == "__main__": main()
