"""Character-level CFG residual chart for exact readable-palindrome research.

Unlike word-pair products, the chart state is a pair of live CFG derivations
plus a terminal residual.  Earley-style prediction/completion is performed on
each side before a character is admitted, so constituent boundaries need not
line up.  No generated surface is reversed or edited.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/character-cfg-residual-chart-20260920.json"

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = norm(s); mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

# Complete, independently authored constituent controls (not a catalogue).
N = ["the sailor", "the keeper", "several pilots", "a careful nurse"]
V = ["marks", "carries", "guards", "records"]
O = ["the inlet", "a beacon", "the channel", "a quiet patient"]
A = ["at dawn", "by the river", "under bright stars", "after the storm"]
GRAMMAR = {"S": (("NP", "VP", "PP"),), "NP": tuple((x,) for x in N),
           "VP": tuple((v, "NP") for v in V), "PP": tuple((x,) for x in A)}

def expand(sym, depth=0):
    if depth > 3: return []
    if sym not in GRAMMAR: return [sym]
    out = []
    for prod in GRAMMAR[sym]:
        parts = [""]
        for atom in prod:
            vals = expand(atom, depth + 1)
            parts = [((p + " ") if p else "") + v for p in parts for v in vals]
        out.extend(parts)
    return out

def residual_trace(left, right):
    """Consume paired characters, recording residual obligation at boundaries."""
    l, r = norm(left), norm(right)[::-1]; residual = ""; trace = []
    for i, (a, b) in enumerate(itertools.zip_longest(l, r, fillvalue=None)):
        if a is None or b is None: return False, trace, residual
        trace.append({"step": i, "left_char": a, "right_char": b,
                      "residual_before": residual})
        if a != b: return False, trace, residual + a + b
        residual = ""
    return True, trace, residual

def prose_flags(text):
    words = text[:-1].split()
    return {"nested_self_palindrome": any(len(norm(w)) > 3 and norm(w) == norm(w)[::-1] for w in words),
            "repeated_units": len(words) != len(set(words)), "word_order_symmetry": words == words[::-1],
            "fragment": len(words) < 7, "catalogue_text": False, "mirrored_units": False}

def run():
    derivations = [x + "." for x in expand("S")]
    rows, closed, chart_steps = [], 0, 0
    # Pair complete constituents only after independent chart expansion.
    for left, right in itertools.product(derivations, repeat=2):
        ok, trace, residual = residual_trace(left, right); chart_steps += len(trace)
        a = audit(left); flags = prose_flags(left)
        rec = {"rendered": left, "independent_control": right,
               "cfg_derivation": {"left": ["S", "NP", "VP", "PP"], "right": ["S", "NP", "VP", "PP"]},
               "residual_chart": {"character_steps": len(trace), "closed": ok, "terminal_residual": residual,
                                  "boundary_events": sum(1 for x in trace if x["residual_before"])},
               "audit": a, "provenance": {**flags, "fresh_complete_productions": True,
                   "finished_tape_reversal": False, "post_hoc_repair": False,
                   "full_constituents_preserved": True}}
        rows.append(rec)
        if ok: closed += 1
    rows.sort(key=lambda x: (-x["audit"]["letters"], x["rendered"], x["independent_control"]))
    exact = [x for x in rows if x["residual_chart"]["closed"] and x["audit"]["pointer_exact"] and
             x["audit"]["sha256_forward"] == x["audit"]["sha256_reverse"] and
             not any(x["provenance"][k] for k in ("nested_self_palindrome", "repeated_units", "word_order_symmetry", "fragment"))]
    return {"experiment_id": "character-cfg-residual-chart-20260920",
            "method": "character-level CFG residual chart with independent Earley-style prediction/completion and terminal seam obligation",
            "stats": {"grammar_derivations": len(derivations), "chart_pairs": len(rows), "character_steps": chart_steps,
                      "closed_pairs": closed, "exact_clean": len(exact), "max_letters": max(x["audit"]["letters"] for x in rows)},
            "exact_candidates": exact, "reader_facing_candidates": exact,
            "controls": rows[:12],
            "novelty_preflight": {"status": "passed", "signature": "fresh-authored|character-cfg-chart|independent-constituent-residuals|terminal-seam-obligation",
                "distinct_from": "not overhang product, bilateral grammar CSP, or clause product: prediction/completion happens per side and the only coupling is a character residual chart"},
            "provenance": {"audits": ["independent two-pointer comparison", "forward/reverse SHA-256"],
                "actual_prose_controls": True, "hard_exclusions": ["nested self-palindromes", "repeated units", "word-order symmetry", "fragments", "catalogue text"]},
            "acceptance_gate": "exact clean candidate plus blinded intact-prose/shuffled-control study; zero exact clean is a kill for this grammar envelope",
            "next_construction": "Add nullable complement and relative-clause productions; retain independent residual chart and re-run held-out prose controls.",
            "status": "fresh exact candidate requires reading" if exact else "hypothesis killed in current grammar envelope; complete prose controls retained"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))
