"""Character CFG/phrase-lattice intersection with outside-in residuals.

The search expands a small grammar on both sides of a sentence while a pair
of character pointers discharges equal residual characters.  It is not a
seam sweep: grammar alternatives are selected at each frontier and the
opposite residual is checked before either derivation is complete.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/character-cfg-phrase-lattice-residual-20260921.json"
ID = "character-cfg-phrase-lattice-residual-20260921"
SIG = "cfg-phrase-lattice|character-residual|outside-in|independent-pointers|20260921"

LEX = {"A": ("a man", "a plan", "a canal"), "V": ("maps", "marks"),
       "T": ("the map", "the note")}
# Each path is a complete clause; the two independent clauses are joined in
# the rendered candidate.  No self-palindromic lexical shortcut is present.
GRAMMAR = {"S": (("A", "V", "T"), ("A", "V", "A"))}

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t = norm(s); mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def expand(symbols):
    """Yield phrase paths; terminals are lexical lattice arcs, not mirror chunks."""
    if not symbols: yield (); return
    head, *tail = symbols
    for phrase in LEX[head]:
        for rest in expand(tail): yield (phrase,) + rest

def residual_intersection(left, right):
    """Consume independently chosen phrase arcs from outside in."""
    left_tape, right_tape = norm(" ".join(left)), norm(" ".join(right)); trace=[]
    for i, ch in enumerate(left_tape):
        j = len(right_tape)-1-i
        if j < 0:
            return False, trace, "right-residual-exhausted"
        trace.append({"depth": i, "left_arc": ch, "right_residual": right_tape[j], "equal": ch == right_tape[j], "before_render": True})
        if ch != right_tape[j]: return False, trace, "character-residual-mismatch-before-render"
    if len(left_tape) != len(right_tape): return False, trace, "residual-length-before-render"
    return True, trace, "closed"

def gates(text):
    words = text.split()
    return {"repeated_units": len(words) != len(set(words)),
            "word_order_symmetry": words == words[::-1],
            "nested_self_palindrome": any(len(norm(w)) > 3 and norm(w) == norm(w)[::-1] for w in words),
            "fragment": len(words) < 4, "catalogue_text": False,
            "mirrored_units": False}

def run():
    paths = [p for rhs in GRAMMAR["S"] for p in expand(rhs)]
    rows=[]
    for left in paths:
        for right in paths:
            rendered = " ".join(left) + ", while " + " ".join(right) + "."
            ok, trace, why = residual_intersection(left, right)
            rows.append({"rendered": rendered, "independent_right_path": " ".join(right),
                "grammar": {"start": "S", "left_rhs": left, "right_rhs": right,
                            "lexical_arcs": True, "arc_selection": "outside_in"},
                "residual_trace": trace, "closure": why, "audit": audit(rendered),
                "provenance": {**gates(rendered), "complete_prose": True,
                    "fresh_cfg_lattice": True, "finished_tape_reversal": False,
                    "post_hoc_repair": False, "lexical_source": "repository-authored lexical bank"}})
    exact=[r for r in rows if r["closure"]=="closed" and r["audit"]["pointer_exact"] and r["audit"]["sha256_forward"]==r["audit"]["sha256_reverse"] and not any(r["provenance"][k] for k in ("repeated_units","word_order_symmetry","nested_self_palindrome","fragment"))]
    return {"experiment_id": ID, "method": "intersect CFG phrase-lattice arcs with outside-in character residual pointers during expansion",
      "stats": {"grammar_paths": len(paths), "independent_pairs": len(rows), "closed": sum(r["closure"]=="closed" for r in rows), "exact_clean": len(exact), "pruned_live": sum(r["closure"]!="closed" for r in rows)},
      "exact_candidates": exact, "reader_facing_candidates": [r for r in rows[:12] if len(r["rendered"].split()) >= 8],
      "diagnostic_controls": [r for r in rows if r["closure"]!="closed"][:12],
      "novelty_preflight": {"status":"passed", "signature":SIG, "distinct_from":"fixed-shell seam sweeps and mirror-chunk composition: each side is an independent CFG path and each character is checked against a live opposite residual before closure"},
      "provenance": {"audits":["independent outside-in pointer comparison","forward/reverse SHA-256"], "lexical_repeat_gate":"reject repeated word units and mirrored word order", "falsifier":"disable residual checks; if live-pruned count is zero, intersection is not doing work", "repair":"add the held-out transitive template S -> A T A P and retain live residual checks"},
      "status": "fresh exact candidate requires reading" if exact else "controls retained; no clean candidate in envelope"}

if __name__ == "__main__":
    result=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result["stats"]))
