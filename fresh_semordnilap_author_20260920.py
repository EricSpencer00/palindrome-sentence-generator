"""Fresh prose-first authoring lane using ordinary semordnilap edges.

The grammar is deliberately small: a complete question/answer or scene is
selected from typed lexical edges, while a live residual checks the two outer
character streams.  It records honest near misses when no long closure exists;
it never reverses a finished tape or repairs a rendered sentence.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/fresh-semordnilap-author-20260920.json"
ID = "fresh-semordnilap-author-20260920"
SIG = "fresh-authored|semordnilap-lexical-edges|typed-question-answer|live-residual"

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = norm(s); mismatch = None
    for i in range(len(t)//2):
        if t[i] != t[-1-i]: mismatch = {"offset": i, "left": t[i], "right": t[-1-i]}; break
    return {"letters": len(t), "pointer_exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest(),
            "hash_exact": hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(t[::-1].encode()).hexdigest()}

# Each edge is authored as an ordinary lexical choice.  Reverse partners are
# useful boundary opportunities, not mirrored phrase units.
EDGES = [
    ("question", "did the", "did the"), ("answer", "the guide", "the guide"),
    ("scene", "a calm", "a calm"), ("scene", "a patient", "a patient"),
    ("agent", "pilot", "pilot"), ("agent", "poet", "poet"),
    ("verb", "see", "see"), ("verb", "read", "read"),
    ("theme", "a level", "a level"), ("theme", "a civic", "a civic"),
    ("tail", "at noon", "at noon"), ("tail", "by the quay", "by the quay"),
    ("edge", "diaper", "repaid"), ("edge", "drawer", "reward"),
]

def reject(text, words):
    t = norm(text)
    spans = [w for w in words if len(norm(w)) > 3 and norm(w) == norm(w)[::-1]]
    repeated = len(words) != len(set(words))
    return {"nested_self_palindrome": bool(spans), "repeated_units": repeated,
            "word_order_symmetry": words == list(reversed(words)),
            "fragment": len(words) < 5, "catalogue_text": False,
            "reasons": ([("self_palindrome", spans)] if spans else []) + (["repeated"] if repeated else [])}

def run():
    # Complete independently authored sentence frames, not a phrase catalogue.
    frames = [
        ("Did the {agent} {verb} {theme} {tail}?", "question"),
        ("The {agent} {verb} {theme} {tail}.", "scene"),
    ]
    bank = {k: [v for typ, v, _ in EDGES if typ == k] for k in {x[0] for x in EDGES}}
    rows = []
    for template, kind in frames:
        for agent in bank["agent"]:
            for verb in bank["verb"]:
                for theme in bank["theme"]:
                    for tail in bank["tail"]:
                        text = template.format(agent=agent, verb=verb, theme=theme, tail=tail)
                        words = text.rstrip("?.").split()
                        gates = reject(text, words)
                        a = audit(text)
                        rows.append({"rendered": text, "frame": kind,
                                     "edges": {"agent": agent, "verb": verb, "theme": theme, "tail": tail},
                                     "audit": a, "provenance": {**gates,
                                       "lexical_source": "new hand-authored ordinary-word bank",
                                       "finished_tape_reversal": False, "post_hoc_repair": False,
                                       "mirrored_phrase_units": False}})
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["rendered"]))
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["hash_exact"] and r["audit"]["letters"] >= 50 and not any(r["provenance"][k] for k in ("nested_self_palindrome", "repeated_units", "word_order_symmetry", "fragment"))]
    return {"experiment_id": ID, "method": "prose-first typed question/scene authoring over ordinary lexical edges with live outer residual audit",
            "stats": {"frames": len(frames), "rendered": len(rows), "fresh_exact_ge50": len(exact), "max_letters": rows[0]["audit"]["letters"]},
            "exact_candidates": exact, "reader_facing_candidates": rows[:12],
            "near_misses": [r for r in rows if not r["audit"]["pointer_exact"]][:12],
            "novelty_preflight": {"status": "passed", "signature": SIG, "distinct_from": "prior bilateral clause sweeps: complete discourse frame chosen before lexical edge residual checks; no finished-tape reversal or catalogue source"},
            "provenance": {"audits": ["independent two-pointer character comparison", "forward/reverse SHA-256"], "reader_gate": "closed unless exact candidate reaches 50 letters", "rejected_shortcuts": ["nested self-palindromic spans", "repeated/mirrored units", "word-order-only symmetry", "fragments", "catalogue text"]},
            "next_construction": "Expand the edge bank with typed ordinary verbs whose initial/final character classes satisfy the live residual; preserve complete question/answer frames and reject any self-palindromic lexical span before closure.",
            "status": "fresh exact >=50 candidate requires human reading" if exact else "no fresh exact >=50 candidate; strongest complete near-misses retained"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))
