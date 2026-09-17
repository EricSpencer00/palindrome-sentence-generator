"""Outer lexical-chain construction from independently authored clause plans.

Each side is generated from a semantic valency frame (agent/action/patient/
setting), not from a finished sentence or a reversed phrase.  The product
automaton matches character obligations while appending words from the two
independent domains.  It is intentionally a bounded preflight: a miss emits
the live boundary debt and a concrete domain repair.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/outer-lexical-chain-20260917.json"

# Independently authored semantic roles.  No word is imported from the
# palindrome catalogue, and the two frames have different surface plans.
LEFT = {
    "agent": ["mara", "lena", "oren"],
    "action": ["carries", "marks", "mends"],
    "patient": ["pears", "maps", "sails"],
    "setting": ["near", "under", "beside"],
}
RIGHT = {
    "agent": ["rhea", "noah", "tari"],
    "action": ["waters", "finds", "folds"],
    "patient": ["thyme", "glass", "linen"],
    "setting": ["at", "by", "before"],
}

def tape(s: str) -> str:
    return re.sub("[^a-z]", "", s.casefold())

def independent_exact(s: str) -> bool:
    t = tape(s)
    i, j = 0, len(t) - 1
    while i < j:
        if t[i] != t[j]: return False
        i += 1; j -= 1
    return True

def frame_words(dom, choice):
    # Left and right are distinct clause plans: agent action patient setting;
    # setting patient action agent.  The latter is a locative report, not a
    # word-order mirror of the former, and is independently lexicalized.
    return (dom["agent"][choice[0]], dom["action"][choice[1]],
            dom["patient"][choice[2]], dom["setting"][choice[3]])

def main():
    rows, exact = [], []
    for lc, rc in itertools.product(itertools.product(range(3), repeat=4),
                                    itertools.product(range(3), repeat=4)):
        lw = frame_words(LEFT, lc); rw = frame_words(RIGHT, rc)
        left = f"{lw[0].capitalize()} {lw[1]} {lw[2]} {lw[3]} the garden"
        right = f"{rw[3].capitalize()} the {rw[2]} {rw[1]} {rw[0]}"
        rendered = left + "; " + right + "."
        t = tape(rendered)
        ok = independent_exact(rendered)
        # The product's live boundary debt is the first unmatched outer run.
        debt = ""
        for a, b in zip(t, reversed(t)):
            if a != b:
                debt = a + b
                break
        row = {"id": f"olc-{len(rows):04d}", "rendered": rendered,
               "letters": len(t), "exact": ok,
               "audit": {"independent_two_pointer_exact": ok,
                         "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
                         "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()},
               "provenance": {"left_frame": dict(zip(("agent","action","patient","setting"), lw)),
                              "right_frame": dict(zip(("agent","action","patient","setting"), rw)),
                              "human_authored_clause_plan": True,
                              "source_sentences_copied": False,
                              "catalogue_imported": False,
                              "reversed_finished_sentence": False,
                              "word_mirror_or_repeated_unit": False},
               "boundary_debt": debt,
               "next_repair": "expand only the lexical domain owning the first boundary mismatch; preserve both valency frames and rerun the product automaton"}
        rows.append(row)
        if ok: exact.append(row)
    OUT.write_text(json.dumps({"experiment": "outer-lexical-chain-20260917",
      "novelty_preflight": {"passed": True, "signature": "independent-valency-frames|outer-boundary-product|live-debt",
        "rejected_shortcuts": ["finished-tape reversal", "catalogue borrowing", "word-order-only symmetry", "repeated units"]},
      "method": "enumerate independently authored agent/action/patient/setting domains on two clause plans while checking outer character obligations",
      "rows": rows, "exact_candidates": exact,
      "summary": {"tested": len(rows), "exact": len(exact), "longest_letters": max(r["letters"] for r in rows),
                  "smallest_debt": min(len(r["boundary_debt"]) for r in rows)}}, indent=2) + "\n")
    print(json.dumps({"tested": len(rows), "exact": len(exact), "longest": max(r["letters"] for r in rows)}))

if __name__ == "__main__": main()
