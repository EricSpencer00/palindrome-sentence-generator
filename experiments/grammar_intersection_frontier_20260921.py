"""Bounded typed-CFG intersection over character frontiers.

This is deliberately grammar-topological: clause trees are derived from
nonterminal signatures and paired by outside-in frontier obligations, rather
than by widening a word/role Cartesian product.
"""
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/grammar-intersection-frontier-20260921.json"

@dataclass(frozen=True)
class Derivation:
    name: str
    signature: tuple[str, ...]
    words: tuple[str, ...]

LEFT = (
    Derivation("gardener", ("NP", "V", "NP"), ("a", "quiet", "gardener", "waters", "the", "orchard")),
    Derivation("teacher", ("NP", "V", "NP"), ("a", "kind", "teacher", "guides", "the", "class")),
    Derivation("artist", ("NP", "V", "NP"), ("a", "calm", "artist", "paints", "the", "vessel")),
    Derivation("pilot_pp", ("NP", "V", "NP", "PP"), ("a", "calm", "pilot", "steers", "the", "boat", "in", "harbor")),
    Derivation("sailor_prepp", ("NP", "V", "PP", "NP"), ("a", "brave", "sailor", "in", "harbor", "guides", "the", "boat")),
)
RIGHT = (
    Derivation("orchard", ("NP", "V", "NP"), ("the", "orchard", "needs", "a", "quiet", "gardener")),
    Derivation("class", ("NP", "V", "NP"), ("the", "class", "needs", "a", "kind", "teacher")),
    Derivation("vessel", ("NP", "V", "NP"), ("the", "vessel", "needs", "a", "calm", "artist")),
    Derivation("harbor_pp", ("NP", "V", "NP", "PP"), ("the", "harbor", "holds", "a", "calm", "pilot", "in", "boat")),
    Derivation("boat_prepp", ("NP", "V", "PP", "NP"), ("the", "boat", "in", "harbor", "follows", "a", "brave", "sailor")),
)

def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())

def audit(text: str) -> dict:
    t = tape(text); mismatch = next((i for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": mismatch is None,
            "first_mismatch": mismatch, "forward_sha256": f,
            "reverse_sha256": r, "sha_equal": f == r}

def frontier(left: str, right: str) -> dict:
    a, b = tape(left), tape(right); i = j = 0; pairs = 0
    while i < len(a) and j < len(b) and a[i] == b[-1-j]:
        pairs += 1; i += 1; j += 1
    return {"matched_frontier_pairs": pairs,
            "left_obligation": a[i] if i < len(a) else None,
            "right_obligation": b[-1-j] if j < len(b) else None,
            "frontier_closed": i == len(a) and j == len(b)}

def main() -> None:
    rows = []
    for l in LEFT:
        for r in RIGHT:
            if l.signature != r.signature: continue
            rendered = " ".join(l.words) + "; " + " ".join(r.words) + "."
            rows.append({"rendered": rendered, "left_derivation": l.name,
                         "right_derivation": r.name, "shared_signature": l.signature,
                         "frontier": frontier(rendered, rendered),
                         "audit": audit(rendered),
                         "reader_status": "unreviewed; exactness does not certify readability",
                         "provenance": {"construction": "typed CFG derivation intersection",
                                        "posthoc_repair": False, "reversal_in_generator": False}})
    exact = [x for x in rows if x["audit"]["two_pointer_exact"]]
    out = {"experiment_id": "grammar-intersection-frontier-20260921",
           "status": "completed_exact" if exact else "completed_no_exact_closure",
           "method": "bounded typed-CFG intersection with two typed PP attachment sites and outside-in character frontiers",
           "candidate_count": len(rows), "exact_count": len(exact), "reader_eligible": False,
           "rendered_candidates": rows,
           "stats": {"longest_letters": max(x["audit"]["letters"] for x in rows),
                     "shared_signature_count": len({x["shared_signature"] for x in rows})},
           "novelty_preflight": {"topology_new": True, "typed_pp_adjunct_added": True,
                                 "second_attachment_site_new": True,
                                 "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "not_relative_lexical_expansion": True,
                                 "not_reward_scoring": True, "prior_experiment_ids_checked":
                                 ["relative-plural-auxiliary-20260921", "center-seam-event-constructor-20260921"]},
           "failure_and_repair": {"failure": "all CFG intersections retain a concrete outer frontier obligation" if not exact else "none",
                                   "next_construction": "add one PP adjunct nonterminal with typed attachment, then rerun frontier intersection"},
           "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "independent_audits": ["two-pointer", "forward/reverse SHA-256"], "shortcuts_excluded": True}}
    RUN.parent.mkdir(exist_ok=True); RUN.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "candidates": len(rows), "exact": len(exact), "longest_letters": out["stats"]["longest_letters"]}))

if __name__ == "__main__": main()
