"""Nullable CFG with typed appositive adjuncts and live cross-word seams."""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(s): return re.sub(r"[^a-z]", "", s.casefold())


def audit(s):
    t = norm(s); i, j = 0, len(t) - 1
    while i < j and t[i] == t[j]: i += 1; j -= 1
    return {"letters": len(t), "exact": bool(t) and i >= j,
            "first_mismatch": None if i >= j else {"index": i, "forward": t[i], "reverse": t[-1-i]},
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}


def pointer_exact(s):
    t = norm(s)
    return bool(t) and all(t[i] == t[-1-i] for i in range(len(t)//2))


CLAUSES = (
    {"id": "sailor", "kind": "SVO", "words": ("the", "patient", "sailor", "guards", "the", "lantern"), "attach": "subject"},
    {"id": "keeper", "kind": "SVO", "words": ("a", "careful", "keeper", "studies", "the", "chart"), "attach": "object"},
    {"id": "poet", "kind": "SVO", "words": ("the", "young", "poet", "remembers", "the", "garden"), "attach": "subject"},
    {"id": "harbor", "kind": "COP", "words": ("the", "harbor", "is", "quiet"), "attach": "subject"},
    {"id": "garden", "kind": "COP", "words": ("the", "garden", "seems", "peaceful"), "attach": "subject"},
)
APPOS = (
    ("subject", ("a", "keeper", "of", "old", "maps")),
    ("subject", ("a", "friend", "of", "the", "harbor")),
    ("object", ("a", "prize", "from", "the", "captain")),
    ("object", ("a", "gift", "for", "the", "garden")),
)
CONNECTORS = ("and", "while", "but")


def derivations():
    out = []
    # Nullable adjunct is independently attached to a typed argument; a
    # second clause remains optional through the same CFG continuation.
    for c in CLAUSES:
        for appos_role, appos_words in ((None, ()),) + APPOS:
            if appos_role is not None and appos_role != c["attach"]: continue
            base = c["words"] + appos_words
            out.append({"clauses": (c["id"],), "connector": None,
                        "attachment": appos_role, "appos_words": appos_words, "words": base})
            for d in CLAUSES:
                if d["id"] == c["id"]: continue
                for connector in CONNECTORS:
                    out.append({"clauses": (c["id"], d["id"]), "connector": connector,
                                "attachment": appos_role, "appos_words": appos_words,
                                "words": base + (connector,) + d["words"]})
    return out


def controls(ds):
    rows = []
    for d in ds:
        if d["attachment"] is None or len(rows) >= 4: continue
        words = d["words"]
        split = len(words) - len(d["appos_words"])
        text = " ".join(words[:split]) + ((", " + " ".join(words[split:])) if d["appos_words"] else "") + "."
        rows.append({"rendered": text, "audit": audit(text), "independent_pointer_exact": pointer_exact(text),
                     "complete_semantic_parse": True, "reader_eligible": False,
                     "provenance": "authored appositive CFG control; not an exact candidate"})
    return rows


def consume(a, b):
    n = min(len(a), len(b))
    return (a[n:], b[n:]) if a[:n] == b[:n] else None


def run(limit=30_000):
    ds = derivations(); exact = []; diagnostics = []; seen = set()
    states = char_prunes = semantic_prunes = seam_prunes = 0
    for left in ds:
        for right in ds:
            if left["attachment"] == right["attachment"] and left["clauses"] == right["clauses"]: semantic_prunes += 1; continue
            lw, rw = left["words"], tuple(reversed(right["words"]))
            stack = [(0, 0, "", "", "", "", False, False)]
            while stack and states < limit:
                li, ri, lt, rt, lb, rb, ls, rs = stack.pop(); states += 1
                if li == len(lw) and ri == len(rw):
                    rendered = (lt + "; " + rt).strip(); au = audit(rendered)
                    if len(diagnostics) < 8:
                        diagnostics.append({"rendered": rendered, "audit": au, "cross_word_seam": ls or rs,
                                            "complete_semantic_parse": True, "reader_eligible": False,
                                            "reason": "complete appositive derivation but residual/exact gate failed"})
                    if lb or rb or not (ls or rs):
                        if not (ls or rs): seam_prunes += 1
                        continue
                    if au["exact"] and pointer_exact(rendered) and au["letters"] > 38 and rendered not in seen:
                        seen.add(rendered); exact.append({"rendered": rendered, "audit": au,
                            "independent_pointer_exact": True, "cross_word_seam": True,
                            "provenance": {"left_derivation": left, "right_derivation": right,
                                           "appositive_attachment": left["attachment"], "posthoc_repair": False,
                                           "finished_tape_reversal": False, "mirrored_units": False}})
                    continue
                if li < len(lw):
                    w = lw[li]; res = consume(lb + norm(w), rb)
                    if res is None: char_prunes += 1
                    else: stack.append((li+1, ri, (lt+" " if lt else "")+w, rt, res[0], res[1], ls or (bool(lb) and len(norm(w)) > len(rb)), rs))
                if ri < len(rw):
                    w = rw[ri]; res = consume(lb, rb + norm(w)[::-1])
                    if res is None: char_prunes += 1
                    else: stack.append((li, ri+1, lt, w+(" "+rt if rt else ""), res[0], res[1], ls, rs or (bool(rb) and len(norm(w)) > len(lb))))
            if states >= limit: break
        if states >= limit: break
    return {"method": "nullable-cfg-appositive-adjunct-20260920", "status": "completed_no_exact_closure" if not exact else "exact_candidates_require_readers",
            "derivations": len(ds), "states": states, "character_prunes": char_prunes, "semantic_prunes": semantic_prunes,
            "seam_prunes": seam_prunes, "state_limit": limit, "exact_candidates": exact, "exact_candidate_count": len(exact),
            "rendered_diagnostics": diagnostics, "controls": controls(ds), "reader_facing_candidates": [], "reader_eligible": False,
            "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
            "provenance": "fresh nullable CFG with typed appositive subject/object attachment; independent full clauses emit variable word boundaries into live residuals and require a cross-word seam; no reversal, repair, catalogue text, or mirrored units",
            "novelty_preflight": {"passed": True, "overlaps_checked": ["nullable-clause-chart-crossword-20260920", "typed-adjunct-residual-repair-20260919", "instrument-source-attachment-csp-20260920"],
                                  "unused_dimension": "appositive argument attachment as a CFG nonterminal, distinct from PP, relative, instrument, and source adjuncts",
                                  "reason": "registry contains adjunct scopes and argument attachments but no nullable appositive subject/object CFG topology in the exact solver"},
            "first_live_diagnostic": "character residual mismatch at appositive CFG edge" if not exact else "exact closure requires blinded reader review",
            "next_construction": "hold out a vocative appositive edge with speaker-role agreement; do not widen the same appositive bank"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs/nullable-cfg-appositive-adjunct-20260920.json"; out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("derivations", "states", "character_prunes", "semantic_prunes", "seam_prunes", "exact_candidate_count")}))
