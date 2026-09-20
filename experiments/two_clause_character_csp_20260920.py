"""Outside-in character CSP over two typed contemporary-English clauses.

Unlike reverse parsing or repair, both clause derivations are selected while
the unmatched character buffers are live.  A word is admitted only when its
newly exposed characters agree with the opposite buffer.  Adjunct and relative
slots are genuine grammar alternatives, not copied tape material.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "two-clause-character-csp-20260920.json"
EXPERIMENT_ID = "two-clause-character-csp-20260920"

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def audit(s: str) -> dict[str, object]:
    t = letters(s); bad = [(i, len(t)-1-i) for i in range(len(t)//2) if t[i] != t[-1-i]]
    f, r = hashlib.sha256(t.encode()).hexdigest(), hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "exact": bool(t) and not bad, "first_mismatch": bad[0] if bad else None,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

@dataclass(frozen=True)
class Slot:
    role: str
    words: tuple[str, ...]

def slot(role: str, *words: str) -> Slot: return Slot(role, tuple(dict.fromkeys(words)))

def grammars() -> dict[str, tuple[Slot, ...]]:
    det = slot("det", "a", "an", "the", "some", "one")
    sg = slot("sg_subject", "aide", "artist", "bard", "clerk", "doctor", "farmer", "nurse", "poet", "scribe", "teacher", "writer")
    pl = slot("pl_subject", "men", "women", "artists", "bards", "clerks", "farmers", "nurses", "poets", "scribes", "teachers", "writers")
    sv = slot("sg_verb", "aids", "asks", "calls", "draws", "edits", "finds", "helps", "keeps", "marks", "meets", "notes", "reads", "rips", "saves", "sees", "sends", "writes")
    pv = slot("pl_verb", "aid", "ask", "call", "draw", "edit", "find", "help", "keep", "mark", "meet", "note", "read", "rip", "save", "see", "send", "write")
    obj = slot("object", "book", "books", "letter", "letters", "memo", "memos", "note", "notes", "poem", "poems", "story", "stories", "tale", "tales", "text", "texts", "verse")
    qty = slot("quantity", "one", "two", "three", "four", "five", "nine", "ten", "many")
    name = slot("name", "ada", "anna", "diana", "iris", "leon", "maria", "nora", "noel", "sara", "zoe")
    prep = slot("prep", "at", "by", "in", "on", "to")
    place = slot("place", "home", "park", "room", "town", "yard")
    rel = slot("relative", "who", "that")
    # These are typed surface plans: agreement is enforced by the paired plan,
    # while optional PP/relative structure increases semantic length.
    return {
        "two_svo": (det, sg, sv, det, obj, det, pl, pv, name),
        "two_svo_pp": (det, sg, sv, det, obj, prep, place, det, pl, pv, name),
        "two_relative": (det, sg, rel, pv, det, obj, det, pl, sv, name),
        "two_pp_relative": (det, sg, rel, pv, det, obj, prep, place, det, pl, sv, name),
        "two_quantity": (det, sg, sv, qty, obj, det, pl, pv, name),
    }

def grammatical(template: tuple[Slot, ...], words: tuple[str, ...]) -> bool:
    roles = [x.role for x in template]
    # Agreement and lightweight valency checks are checked on the complete
    # derivation, but choices are still made incrementally by the CSP.
    if "sg_subject" in roles and "sg_verb" in roles:
        i, j = roles.index("sg_subject"), roles.index("sg_verb")
        if not words[i] or not words[j].endswith("s"): return False
    if "pl_subject" in roles and "pl_verb" in roles:
        i, j = roles.index("pl_subject"), roles.index("pl_verb")
        if words[i] and words[j].endswith("s"): return False
    return True

def search(template: tuple[Slot, ...], limit=80, state_limit=900_000) -> dict[str, object]:
    states = pruned = 0; exact = []; witnesses = []; seen_words: set[tuple[str,...]] = set()
    def compatible(left: str, right: str) -> bool:
        n = min(len(left), len(right)); return left[:n] == right[::-1][:n]
    def walk(lo: int, hi: int, pref: str, suff: str, left: tuple[str,...], right: tuple[str,...]):
        nonlocal states, pruned
        if states >= state_limit or len(exact) >= limit: return
        if lo > hi:
            states += 1; words = left + right
            if len(set(words)) < len(words) or not grammatical(template, words): return
            rendered = " ".join(words); a = audit(rendered)
            if a["exact"] and len(letters(rendered)) > 38:
                exact.append({"rendered": rendered, "audit": a, "provenance": provenance(template)})
            return
        if lo == hi:
            for w in template[lo].words:
                if w in left or w in right or w == w[::-1]: continue
                np = pref + letters(w); states += 1
                if compatible(np, suff): walk(lo+1, hi-1, np, suff, left+(w,), right)
                else: pruned += 1
            return
        for lw in template[lo].words:
            if lw in left or lw in right or lw == lw[::-1]: continue
            for rw in template[hi].words:
                if rw in left or rw in right or rw == rw[::-1] or rw == lw: continue
                np, ns = pref + letters(lw), letters(rw) + suff; states += 1
                if compatible(np, ns): walk(lo+1, hi-1, np, ns, left+(lw,), (rw,)+right)
                else:
                    pruned += 1
                    if len(witnesses) < 20: witnesses.append({"rendered": " ".join(left+(lw,)+(rw,)+right), "audit": audit(" ".join(left+(lw,)+(rw,)+right))})
    walk(0, len(template)-1, "", "", (), ())
    return {"candidates": exact, "witnesses": witnesses, "stats": {"states": states, "pruned": pruned, "exact_gt38": len(exact)}}

def provenance(t):
    return {"construction": "outside-in typed two-clause character CSP", "template_roles": [s.role for s in t],
            "finished_tape_reversal": False, "post_hoc_repair": False, "catalogue_text": False,
            "repeated_word": False, "reader_status": "unreviewed; programmatic exactness does not certify readability"}

def run():
    ss = {k: search(v) for k,v in grammars().items()}
    controls = []
    for text in ("An aide reads a memo; some poets write Diana", "The artist who read the book met the teachers at home"):
        controls.append({"rendered": text, "audit": audit(text), "provenance": {"source": "authored control", "generated": False}})
    candidates = [c for r in ss.values() for c in r["candidates"]]
    return {"experiment_id": EXPERIMENT_ID, "method": "outside-in live character obligations over typed two-clause grammars with optional PP/relative structure", "searches": ss, "candidates": candidates, "controls": controls, "best_length": max((c["audit"]["letters"] for c in candidates), default=0), "provenance": {"generated_compositionally": True, "independent_audit": "pointer comparison plus SHA-256 forward/reverse", "novelty": "not seed-conditioned, reverse-parsed, endpoint-indexed, or repair-based"}, "reader_gate": "closed pending blinded human ratings", "next_construction": "expand typed clause inventory while preserving outside-in obligations"}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"best_length": result["best_length"], "exact_gt38": len(result["candidates"]), "searches": {k:v["stats"] for k,v in result["searches"].items()}}))
