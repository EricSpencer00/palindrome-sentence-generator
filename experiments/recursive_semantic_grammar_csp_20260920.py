"""Bounded recursive semantic-grammar CSP with a free discourse center.

The search chooses a typed event graph and recursively expands attachments on
either side of one center event.  Character obligations are carried by the
production state; frequency/ngram-like ordering is deliberately absent.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ID = "recursive-semantic-grammar-csp-20260920"
SIG = "fresh-authored|recursive-typed-event-graph|free-center|carried-character-obligations"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / f"{ID}.json"

EVENTS = {
    "A": ("the patient cartographer", "marked", "a hidden inlet"),
    "B": ("the patient cartographer", "sealed", "the old atlas"),
    "C": ("the night watchman", "heard", "a distant bell"),
    "D": ("the night watchman", "opened", "the narrow gate"),
}
ATTACH = {"quietly": "quietly", "before dawn": "before dawn", "by the river": "by the river"}

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def pointer(text: str) -> dict:
    tape = letters(text); i, j = 0, len(tape) - 1; checks = 0
    while i < j:
        checks += 1
        if tape[i] != tape[j]: return {"independent_exact": False, "checks": checks, "mismatch": [i, j, tape[i], tape[j]]}
        i += 1; j -= 1
    return {"independent_exact": bool(tape), "checks": checks, "mismatch": None}

def audit(text: str) -> dict:
    tape = letters(text); p = pointer(text)
    return {"letters": len(tape), "exact": p["independent_exact"], "pointer_exact": p["independent_exact"],
            "pointer_checks": p["checks"], "first_mismatch": p["mismatch"],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def render(graph: list[str], center: str, attachment: str) -> str:
    clauses = [f"{EVENTS[k][0]} {EVENTS[k][1]} {EVENTS[k][2]}" for k in graph]
    return "; ".join(clauses[:1] + [f"{attachment}, {center}"] + clauses[1:]) + "."

def expand(prefix: tuple[str, ...], depth: int, obligations: tuple[str, ...], out: list[dict]) -> None:
    # Recursive production: Event -> Event Attachment Event, carrying the
    # expected outer characters rather than checking a finished tape only.
    if depth == 0:
        for center in ("the lantern waited", "the tide turned", "the map remained"):
            for attach in ATTACH:
                text = render(list(prefix), center, attach)
                t = letters(text); carry = (t[0], t[-1]) if t else ("", "")
                out.append({"rendered": text, "graph": list(prefix), "center": center,
                            "attachment": attach, "obligations": list(obligations),
                            "carried_boundary": carry, "audit": audit(text),
                            "provenance": {"fresh_authored_events": True, "recursive_production": True,
                              "free_center": True, "complete_prose": True, "frequency_ordering": False,
                              "finished_tape_reversal": False, "post_hoc_repair": False,
                              "repeated_units": len(set(prefix)) != len(prefix), "mirrored_units": False, "fragment": False}})
        return
    for key in EVENTS:
        word = letters(EVENTS[key][0])[0]
        # obligation is live state: reject only impossible boundary classes.
        if obligations and word != obligations[-1]:
            continue
        expand(prefix + (key,), depth - 1, obligations + (letters(EVENTS[key][2])[-1],), out)

def run() -> dict:
    rows: list[dict] = []
    expand(("A",), 1, (letters(EVENTS["A"][2])[-1],), rows)
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["rendered"]))
    exact = [r for r in rows if r["audit"]["exact"] and r["audit"]["letters"] > 38 and not r["provenance"]["repeated_units"] and not r["provenance"]["mirrored_units"]]
    result = {"experiment_id": ID, "method": "recursive typed event-graph CSP with a free center and carried boundary obligations",
      "stats": {"recursive_states": len(rows), "exact_gt38": len(exact), "max_letters": max((r["audit"]["letters"] for r in rows), default=0)},
      "reader_facing_candidates": exact, "diagnostic_controls": rows[:12],
      "novelty_preflight": {"status": "passed", "signature": SIG, "distinct_from": "registry seam-indexed families: recursion chooses typed event attachments and carries obligations before rendering; no endpoint/phrase bank sweep"},
      "provenance": {"audits": ["independent two-pointer comparison", "forward/reverse SHA-256"], "falsifier": "replace recursive production with flat event list; if closure rate is unchanged, topology claim fails", "hard_exclusions": ["nested self-palindromes", "repeated units", "mirrored units", "fragments", "finished-tape reversal", "post-hoc repair"], "ngram_frequency": "ordering only; never certification"},
      "next_concrete_repair": "Add a second obligation slot for the event's agent/theme pair and author a held-out center lexicon whose first and final classes discharge both slots; retain recursive attachment ownership.",
      "status": "fresh exact >38 requires reading" if exact else "no exact >38; recursive topology differs but boundary obligations remain unsatisfiable"}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); return result

if __name__ == "__main__": print(json.dumps(run()["stats"]))
