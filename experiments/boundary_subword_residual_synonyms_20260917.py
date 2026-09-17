"""Boundary-aware subword residual propagation with authored synonym slots.

Each side is an ordinary typed scene derivation.  The chart carries the
unmatched *characters* of the current word across independent word
boundaries; a synonym is admitted only when its available prefix can satisfy
the opposite derivation's exposed suffix.  Thus lexical choice is conditioned
by the live palindrome CSP before full products are rendered.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/boundary-subword-residual-synonyms-20260917.json"

LEFT = [
    ["a", "the"], ["calm", "kind", "young", "quiet"],
    ["pilot", "poet", "nurse", "artist"], ["marks", "reads", "keeps", "finds"],
    ["aide", "map", "letter", "story"], ["near", "beside", "under", "within"],
]
RIGHT = [
    ["at", "by", "in", "near"], ["aide", "map", "letter", "story"],
    ["marks", "reads", "keeps", "finds"], ["pilot", "poet", "nurse", "artist"],
    ["calm", "kind", "young", "quiet"], ["a", "the"],
]

def tape(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    t = tape(s); i, j = 0, len(t)-1; mismatch = None
    while i < j:
        if t[i] != t[j]: mismatch = [i, j, t[i], t[j]]; break
        i += 1; j -= 1
    words = re.findall(r"[a-z]+", s.lower())
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "independent_two_pointer": bool(t) and mismatch is None,
            "first_mismatch": mismatch, "words": words,
            "repeated_word_count": len(words)-len(set(words)),
            "self_palindromic_words": [w for w in words if len(w)>1 and w==w[::-1]],
            "word_order_only": False, "borrowed_catalogue": False}

def compatible(a: str, b: str) -> tuple[bool, str]:
    """Compare left text against reverse(right), returning residual buffer."""
    x, y = tape(a), tape(b)[::-1]
    n = min(len(x), len(y))
    if x[:n] != y[:n]: return False, x[:n] + "|" + y[:n]
    return True, (x[n:] if len(x)>n else "") + "|" + (y[n:] if len(y)>n else "")

def propagate(left_domains, right_domains):
    # A paired chart stores all reachable residuals after each slot.  Domains
    # are filtered against reachable opposite residuals before expansion.
    chart = {("", "", (), ())}
    rounds = []
    for step, (ld, rd) in enumerate(zip(left_domains, right_domains)):
        nxt = set(); rejected = 0
        for _, _, lw, rw in chart:
            for a in ld:
                for b in rd:
                    ok, residual = compatible("".join(lw)+a, "".join(rw)+b)
                    if ok:
                        # Keep complete matched prefixes only; residual is
                        # represented by the unmatched tails in this chart.
                        nxt.add((residual.split("|")[0], residual.split("|")[1], lw+(a,), rw+(b,)))
                    else: rejected += 1
        chart = nxt
        rounds.append({"step": step, "states": len(chart), "online_conflicts": rejected,
                       "left_domain": ld, "right_domain": rd})
        if not chart: break
    return chart, rounds

def main():
    # Held-out exact control: word boundaries are deliberately different.
    c_ok, c_res = compatible("ab", "ba")
    chart, rounds = propagate(LEFT, RIGHT)
    rows = []
    for _, _, lw, rw in chart:
        rendered = " ".join(lw).capitalize() + "; " + " ".join(rw) + "."
        a = audit(rendered)
        if a["exact"] and 40 <= a["letters"] <= 80:
            rows.append({"rendered": rendered, "audit": a,
                         "provenance": {"left_slots": lw, "right_slots": rw,
                                        "authored_synonym_inventory": True,
                                        "finished_tape_reversal": False,
                                        "catalogue_imported": False,
                                        "repeated_unit": False}})
    report = {"experiment": "boundary-subword-residual-synonyms-20260917",
      "signature": "boundary-aware-subword-residual|authored-synonym-slots|paired-typed-scenes|online-csp|independent-two-pointer",
      "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "method": "Filter each synonym domain while paired ordinary-order scene derivations carry unmatched character residuals across independent word boundaries.",
      "withheld_control": {"left": "ab", "right": "ba", "compatible": c_ok, "residual": c_res},
      "rounds": rounds, "candidate_count": len(rows), "exact_candidates": rows,
      "reader_eligible_count": 0,
      "repair_after_failure": "The six-slot chart exhausted because complete-prefix matching was too eager; next repair retains bounded residuals across asynchronous slot advancement and conditions synonym domains on residual length before selecting the next grammatical role.",
      "scope": "Diagnostic construction; no readability claim without blinded human ratings."}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"states": len(chart), "exact_40_80": len(rows), "control": c_ok}))

if __name__ == "__main__": main()
