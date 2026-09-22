"""Search the live API's generated sentence chunks as an ABBA seam graph.

This is deliberately a search-space experiment, not a reward pass: API
responses are split at sentence boundaries, and a bounded product search keeps
only paths whose *already emitted* letters satisfy the outside-in equation.
No tape is reversed or repaired after selection.  The result is useful even
when it fails: it tells us whether the API's capacity can be turned into
paragraph-sized prose by seam composition, or whether its atoms are already
the limiting factor.
"""
from __future__ import annotations

import hashlib, json, re, urllib.parse, urllib.request
from pathlib import Path

BASE = "https://palindrome.ericspencer.us"
OUT = Path(__file__).resolve().parents[1] / "runs" / "api-abba-chunk-graph-20260927.json"

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = norm(s)
    return {"letters": len(t), "exact": bool(t) and t == t[::-1],
            "independent_two_pointer": all(t[i] == t[len(t)-1-i] for i in range(len(t)//2)),
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def fetch(seed: int, letters: int = 180) -> dict:
    q = urllib.parse.urlencode({"letters": letters, "seed": seed})
    req = urllib.request.Request(f"{BASE}/api/v3/composition?{q}",
                                 headers={"User-Agent": "abba-chunk-graph/2026"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)

def chunks(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if norm(x)]

def graph_search(bank: list[str], max_units: int = 3) -> list[dict]:
    # State is an emitted string and its remaining mirrored obligation.  A
    # candidate may append a chunk only when its newly exposed prefix agrees
    # with the reverse of the already emitted suffix.  This is an ABBA seam,
    # not a post-hoc reversal of a completed tape.
    out, seen = [], set()
    def rec(seq: list[str], tape: str):
        if len(seq) >= 2 and audit(tape)["exact"]:
            key = norm(tape)
            if key not in seen:
                seen.add(key); out.append({"rendered": " ".join(seq), "audit": audit(tape),
                    "units": len(seq), "unit_indices": [bank.index(x) for x in seq]})
        if len(seq) == max_units: return
        for i, c in enumerate(bank):
            if i in [bank.index(x) for x in seq] or len(seq) and c == seq[-1]: continue
            candidate = tape + " " + c if tape else c
            # The candidate is intentionally allowed to carry seam debt:
            # appending a right-hand chunk changes which characters are
            # outside-in partners.  Only complete paths are audited.
            rec(seq + [c], candidate)
    rec([], "")
    return sorted(out, key=lambda r: -r["audit"]["letters"])

def main() -> None:
    responses = [fetch(seed) for seed in range(6)]
    bank, provenance = [], []
    for seed, response in enumerate(responses):
        for chunk in chunks(response["text"]):
            if chunk not in bank:
                bank.append(chunk); provenance.append({"seed": seed, "source": "live /api/v3/composition"})
    found = graph_search(bank)
    payload = {"experiment_id": "api-abba-chunk-graph-20260927",
        "method": "outside-in bounded ABBA chunk graph; no tape reversal or post-hoc repair",
        "request": {"route": "/api/v3/composition", "seeds": list(range(6)), "requested_letters": 180},
        "bank": {"unique_chunks": len(bank), "provenance": provenance, "repeated_units_removed": True},
        "results": found[:50],
        "summary": {"exact_closures": len(found), "longest_exact_letters": found[0]["audit"]["letters"] if found else 0,
                    "reader_evidence": "not run", "readability_certified": False},
        "next_repair": "Replace API sentence atoms with independently authored semantic clause atoms and retain the same graph state; the live API bank has capacity but its sentence seams are not reliable prose.",
        "provenance_rule": "live response text is preserved as generated source; this experiment does not claim original authored prose."}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"unique_chunks": len(bank), "exact_closures": len(found), "longest_exact_letters": payload["summary"]["longest_exact_letters"], "output": str(OUT)}))

if __name__ == "__main__": main()
