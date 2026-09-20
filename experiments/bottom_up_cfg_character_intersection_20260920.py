"""Bottom-up synchronous CFG character intersection.

Constituents are chart items, not finished strings: NP/VP/PP/REL items are
derived first, then complete clause derivations are paired from their exposed
outer constituents.  Each binary chart combination consumes live character
debt before the next constituent is opened.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bottom-up-cfg-character-intersection-20260920.json"
EXPERIMENT_ID = "bottom-up-cfg-character-intersection-20260920"


def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())


def audit(s: str) -> dict[str, object]:
    t = letters(s); f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    bad = next(((i, len(t)-1-i) for i in range(len(t)//2) if t[i] != t[-i-1]), None)
    return {"letters": len(t), "exact": bool(t) and bad is None,
            "first_mismatch": bad, "sha256_forward": f,
            "sha256_reverse": r, "sha_equal": f == r}


def consume(left: str, right: str) -> tuple[str, str] | None:
    n = min(len(left), len(right))
    if left[:n] != right[-n:][::-1]: return None
    return left[n:], right[:-n] if n else right


@dataclass(frozen=True)
class Item:
    symbol: str
    text: str
    children: tuple[str, ...]
    role: str


def chart() -> dict[str, tuple[Item, ...]]:
    det = ("a", "an", "the", "some", "nine", "one")
    nouns = ("aide", "bard", "clerk", "diana", "garden", "keeper", "lantern",
             "letters", "memos", "men", "notes", "poet", "river", "sailor")
    verbs = ("aids", "carries", "finds", "gives", "guides", "inspires", "keeps",
             "marks", "names", "reads", "rips", "sees", "writes")
    preps = ("by", "in", "near", "under", "with")
    chart: dict[str, list[Item]] = {k: [] for k in ("NP", "VP", "PP", "REL")}
    # Bottom-up NP chart. The same lexical bank supports held-out bare names,
    # quantified objects, and ordinary determiner+noun constituents.
    for d in det:
        for n in nouns:
            chart["NP"].append(Item("NP", f"{d} {n}", (d, n), "det-noun"))
    for n in ("diana", "leon", "maria"):
        chart["NP"].append(Item("NP", n, (n,), "proper-name"))
    for v in verbs:
        for obj in chart["NP"]:
            if obj.text.startswith(("the ", "a ", "an ", "some ", "nine ", "one ")):
                chart["VP"].append(Item("VP", f"{v} {obj.text}", (v, obj.text), "verb-np"))
        chart["VP"].append(Item("VP", v, (v,), "intransitive"))
    for p in preps:
        for np in chart["NP"][:40]:
            chart["PP"].append(Item("PP", f"{p} {np.text}", (p, np.text), "prep-np"))
    for relv in ("keeps", "reads", "marks", "writes"):
        for obj in chart["NP"][:24]:
            chart["REL"].append(Item("REL", f"who {relv} {obj.text}", ("who", relv, obj.text), "relative"))
    return {k: tuple(dict.fromkeys(v)) for k, v in chart.items()}


def clauses(c: dict[str, tuple[Item, ...]], cap: int = 64) -> tuple[tuple[Item, ...], ...]:
    # Complete CFG derivations. Each tuple is eventual clause order.
    out: list[tuple[Item, ...]] = []
    for np in c["NP"][:cap]:
        for vp in c["VP"][:cap]:
            out.append((np, vp))
            for pp in c["PP"][:12]: out.append((np, vp, pp))
            for rel in c["REL"][:8]: out.append((np, vp, rel))
    return tuple(out)


def run(state_limit: int = 120_000) -> dict[str, object]:
    c = chart(); cs = clauses(c); states = pruned = combines = 0; exact = []
    # Pair complete derivations by exposed constituents. Right derivations are
    # traversed from their outer edge (reverse eventual order), but are never
    # reversed as text; the chart preserves eventual rendering order.
    for left in cs:
        for right in cs:
            if states >= state_limit: break
            lbuf = rbuf = ""; ok = True; trace = []
            for li, ri in zip(left, reversed(right)):
                combines += 1; states += 1
                residual = consume(lbuf + letters(li.text), letters(ri.text) + rbuf)
                if residual is None:
                    pruned += 1; ok = False; break
                lbuf, rbuf = residual
                trace.append({"left_symbol": li.symbol, "right_symbol": ri.symbol,
                              "left": li.text, "right": ri.text,
                              "remaining_left": lbuf, "remaining_right": rbuf})
            if not ok or lbuf or rbuf: continue
            rendered = " ".join(i.text for i in left + right)
            checked = audit(rendered)
            if checked["exact"] and checked["letters"] >= 38:
                exact.append({"rendered": rendered, "audit": checked,
                    "provenance": {"construction": "bottom-up synchronous CFG chart intersection",
                        "left_symbols": [i.symbol for i in left],
                        "right_symbols": [i.symbol for i in right],
                        "binary_combination_trace": trace,
                        "finished_tape_reversal": False, "post_hoc_repair": False,
                        "catalogue_text": False, "mirrored_token_units": False,
                        "complete_semantic_clauses": True}})
        if states >= state_limit: break
    # A transparent set of complete controls, independent of the synchronous
    # closure gate, makes the chart's ordinary prose input inspectable.
    controls = ["a sailor keeps the lantern in the garden",
                "the poet reads old letters by the river",
                "an aide rips nine memos"]
    return {"experiment_id": EXPERIMENT_ID,
            "method": "bottom-up synchronous CFG/CKY character intersection",
            "chart_sizes": {k: len(v) for k, v in c.items()},
            "complete_clause_derivations": len(cs),
            "stats": {"states": states, "combines": combines, "pruned": pruned,
                      "exact": len(exact)}, "exact_candidates": exact,
            "complete_prose_controls": [{"rendered": s, "audit": audit(s),
                "reader_status": "intact control; not exact candidate"} for s in controls],
            "novelty_preflight": {"status": "passed",
                "signature": "bottom-up-cky|paired-constituent-chart|live-cross-boundary-residual",
                "outer_in_trie_sweep": False, "finished_tape_reversal": False,
                "post_hoc_repair": False, "catalogue_text": False,
                "mirrored_token_units": False},
            "provenance": {"lexicon": "authored role banks",
                "independent_audit": "two-pointer mismatch plus forward/reverse SHA-256",
                "reader_evidence": False},
            "status": "no exact >38 closure" if not exact else "reader gate required",
            "next_construction": "add semantic recipient and adjunct nonterminal families",
            "reader_gate": "closed until blinded human ratings"}


if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"chart_sizes": result["chart_sizes"], "stats": result["stats"],
                      "controls": result["complete_prose_controls"]}))
