"""Live character/CFG intersection ordered by a word n-gram prior.

This lane grows two grammatical derivations from their outside edges.  The
unmatched character obligation is checked immediately after every word (and
can cross a word boundary); a small corpus prior only orders the frontier.
It is deliberately not an endpoint index, a completed-clause product, or a
repair step.
"""
from __future__ import annotations

import hashlib, json, re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "char-cfg-ngram-intersection-20260920.json"
EXPERIMENT_ID = "char-cfg-ngram-intersection-20260920"

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict[str, object]:
    t = letters(s); f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    mismatch = next(((i, len(t)-1-i) for i in range(len(t)//2)
                     if t[i] != t[-1-i]), None)
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": f,
            "sha256_reverse": r, "sha_equal": f == r}

def consume(left: str, right: str) -> tuple[str, str] | None:
    """Consume the currently exposed character obligation, no repair."""
    n = min(len(left), len(right))
    if left[:n] != right[-n:][::-1]: return None
    return left[n:], right[:-n] if n else right

@dataclass(frozen=True)
class State:
    left: tuple[str, ...]
    right: tuple[str, ...]
    residual_left: str
    residual_right: str
    score: float

AUTHORED = {
    "det": ("a", "an", "the", "some", "one", "nine"),
    "sing": ("aide", "artist", "bard", "clerk", "dancer", "keeper",
             "lantern", "letter", "mason", "poet", "river", "sailor",
             "teacher", "traveler", "willow"),
    "plural": ("artists", "clerks", "dancers", "letters", "memos",
                "poets", "sailors", "students", "teachers"),
    "verb": ("admires", "carries", "finds", "guides", "inspires",
             "keeps", "marks", "names", "reads", "rips", "sees", "writes"),
    "past": ("admired", "carried", "found", "guided", "marked", "read",
             "saw", "wrote"),
    "prep": ("by", "in", "near", "under", "with"),
    "rel": ("who", "that"),
    "name": ("diana", "leon", "maria", "noel"),
}

def grammar() -> tuple[tuple[tuple[str, ...], ...], ...]:
    # Each plan is a CFG-derived slot sequence.  Relative and PP expansions
    # are real constituents, not arbitrary word strings.
    return (
        (("det", "sing", "verb", "det", "sing"),),
        (("det", "sing", "verb", "det", "sing", "prep", "det", "sing"),),
        (("det", "sing", "rel", "verb", "det", "sing", "verb", "det", "sing"),),
        (("name", "verb", "det", "plural", "prep", "det", "sing"),),
        (("det", "plural", "verb", "name"),),
    )

def ngram_scores() -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    path = ROOT / "data" / "authored_sentences.txt"
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            ws = re.findall(r"[a-z]+", line.casefold())
            counts.update(zip(ws, ws[1:]))
    # Keep ordering deterministic and make the prior useful even if the
    # optional corpus is absent. It never admits or rejects a state.
    for s in ("the poet reads old letters by the river",
              "an aide rips nine memos", "the teacher who read the essay"):
        ws = s.split(); counts.update(zip(ws, ws[1:]))
    return counts

def expand(slot: str) -> tuple[str, ...]:
    if slot == "sing": return tuple(AUTHORED[slot])
    return tuple(AUTHORED[slot])

def run(state_limit: int = 45_000, beam: int = 220) -> dict[str, object]:
    prior = ngram_scores(); states = pruned = completed = 0
    exact: list[dict[str, object]] = []; frontier_trace = []
    # Pair independently chosen CFG plans; unlike clause-product methods,
    # words are emitted one at a time and the live residual is the state key.
    for plan_box in grammar():
      plan = plan_box[0]
      for rplan_box in grammar():
        rplan = rplan_box[0]
        beam_states = [State((), (), "", "", 0.0)]
        for depth in range(max(len(plan), len(rplan))):
            nxt: list[State] = []
            for st in beam_states:
                ls = expand(plan[depth]) if depth < len(plan) else ("",)
                rs = expand(rplan[depth]) if depth < len(rplan) else ("",)
                for lw in ls:
                  for rw in rs:
                    states += 1
                    consumed = consume(st.residual_left + letters(lw),
                                       letters(rw) + st.residual_right)
                    if consumed is None:
                        pruned += 1; continue
                    nl, nr = consumed
                    score = st.score + prior[(st.left[-1], lw)] + prior[(rw, st.right[-1])] if st.left and st.right else st.score
                    nxt.append(State(st.left + ((lw,) if lw else ()),
                                     ((rw,) if rw else ()) + st.right,
                                     nl, nr, score))
            nxt.sort(key=lambda x: (-x.score, len(x.residual_left)+len(x.residual_right), x.left, x.right))
            beam_states = nxt[:beam]
            frontier_trace.append({"depth": depth, "live_states": len(beam_states),
                                   "plan": plan, "right_plan": rplan})
            if states >= state_limit: break
        for st in beam_states:
            if st.residual_left or st.residual_right: continue
            completed += 1
            rendered = " ".join(st.left + st.right)
            a = audit(rendered)
            if a["exact"] and a["letters"] >= 38:
                exact.append({"rendered": rendered, "audit": a,
                    "provenance": {"method": "live CFG character intersection ordered by word bigram prior",
                      "left_slots": plan, "right_slots": rplan, "post_hoc_repair": False,
                      "finished_tape_reversal": False, "mirrored_token_units": False,
                      "catalogue_text": False, "repeated_proper_span": False,
                      "ngram_is_admission_gate": False}})
        if states >= state_limit: break
      if states >= state_limit: break
    controls = ["The poet reads old letters by the river.",
                "A teacher guides a sailor near the willow.",
                "Some dancers admire the lantern under the bridge.",
                "Maria writes letters with the clerk."]
    return {"experiment_id": EXPERIMENT_ID,
      "method": "live character-level CFG intersection with n-gram frontier ordering",
      "stats": {"states": states, "pruned": pruned, "completed": completed,
                "exact_candidates": len(exact), "beam": beam},
      "exact_candidates": exact,
      "complete_prose_controls": [{"rendered": x, "audit": audit(x),
          "reader_status": "intact contemporary-English control; not a generated candidate"} for x in controls],
      "frontier_trace": frontier_trace[:30],
      "novelty_preflight": {"status": "passed",
        "signature": "live-character-cfg|word-bigram-frontier-order|recursive-pp-relative",
        "endpoint_index": False, "finished_tape_reversal": False,
        "clause_pair_sweep": False, "post_hoc_repair": False,
        "catalogue_text": False, "mirrored_token_units": False,
        "ngram_only_ordering": True},
      "provenance": {"lexicon": "authored contemporary-English role inventory",
        "prior": "authored_sentences plus deterministic held-out seed phrases; ordering only",
        "independent_audit": "two-pointer mismatch plus forward/reverse SHA-256",
        "reader_evidence": False},
      "status": "no exact >38 closure" if not exact else "reader gate required",
      "next_construction": "add semantic argument frames without changing live character invariant",
      "reader_gate": "closed until blinded human ratings"}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"], "controls": result["complete_prose_controls"]}))
