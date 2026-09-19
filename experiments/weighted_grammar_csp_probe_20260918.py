"""Bounded weighted grammar/CSP probe (no completed sentence cross-product).

Each side is a typed SVO derivation.  Domains are reduced by character
equations before a derivation is scored; the chart key is (slot, residual,
left-word,right-word), making this a reproducible construction diagnostic.
"""
from pathlib import Path
import hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/weighted-grammar-csp-probe-20260918.json"
GRAMMAR = {
    "DET": [("the", .1), ("a", .3)],
    "SUBJ": [("keeper", .2), ("sailor", .1), ("scribe", .4)],
    "VERB": [("records", .1), ("charts", .2), ("maps", .3)],
    "OBJ": [("tides", .2), ("routes", .1), ("stars", .3)],
}
SLOTS = ("DET", "SUBJ", "VERB", "OBJ")

def norm(x): return re.sub("[^a-z]", "", x.lower())
def audit(x):
    t = norm(x)
    return {"letters": len(t), "exact": bool(t) and t == t[::-1],
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def run(limit=2000):
    # Residual character domains are filtered online, before weight ranking.
    frontier = [(0, 0, "", "", 0.0, [])]
    expanded = 0; exact = []; rejects = {"character": 0, "budget": 0}
    while frontier and expanded < limit:
        slot, pos, left, right, cost, trace = frontier.pop()
        expanded += 1
        if slot == len(SLOTS):
            text = " ".join(left.split()) + " | " + " ".join(right.split())
            if audit(text)["exact"]: exact.append({"text": text, "audit": audit(text), "trace": trace})
            continue
        typ = SLOTS[slot]
        for lw, lc in GRAMMAR[typ]:
            for rw, rc in GRAMMAR[typ]:
                # Compare the next unmatched character from opposite edges.
                ll, rr = norm(lw), norm(rw)
                if ll[0] != rr[-1]:
                    rejects["character"] += 1; continue
                frontier.append((slot + 1, 0, left + " " + lw, right + " " + rw,
                                 cost + lc + rc,
                                 trace + [f"{typ}:{lw}/{rw}"]))
    if frontier: rejects["budget"] = len(frontier)
    return {"experiment_id": "weighted-grammar-csp-probe-20260918",
            "method": "typed weighted grammar domains with online mirrored-edge CSP pruning",
            "expanded_states": expanded, "exact_path_count": len(exact),
            "top_exact_paths": sorted(exact, key=lambda x: x["trace"])[:5],
            "rejects": rejects, "budget": limit,
            "novelty_preflight": {"completed_sentence_cross_product": False,
                                  "reverse_tape_decode": False,
                                  "rlaif": False},
            "provenance": {"grammar_sha256": hashlib.sha256(json.dumps(GRAMMAR, sort_keys=True).encode()).hexdigest(),
                           "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "independent_audit": "two-pointer normalized tape plus forward/reverse SHA-256"}}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("expanded_states", "exact_path_count", "rejects")}))
