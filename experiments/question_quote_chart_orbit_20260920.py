"""Typed question complements inside a live character-synchronous chart."""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "question-quote-chart-orbit-20260920.json"
EXPERIMENT_ID = "question-quote-chart-orbit-20260920"

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())

def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    bad = [(i, len(tape)-i-1) for i in range(len(tape)//2) if tape[i] != tape[-i-1]]
    fwd = hashlib.sha256(tape.encode()).hexdigest(); rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not bad,
            "first_mismatch": bad[0] if bad else None, "sha256_forward": fwd,
            "sha256_reverse": rev, "sha_equal": fwd == rev}

def consume(left: str, right: str) -> tuple[str, str] | None:
    n = min(len(left), len(right))
    if n and left[:n] != right[-n:][::-1]: return None
    return left[n:], right[:-n] if n else right

@dataclass(frozen=True)
class Item:
    symbol: str; text: str; valency: str
    number: str | None = None; agreement: str | None = None; question: bool = False

def it(symbol: str, text: str, valency: str, *, number=None, agreement=None, question=False) -> Item:
    return Item(symbol, text, valency, number, agreement, question)

def expand(symbol: str, *, depth=0, number=None) -> list[tuple[Item, ...]]:
    if depth > 6: return []
    if symbol == "SPEAKER":
        return [(it("SPEAKER", "the bard", "speaker", number="singular"),),
                (it("SPEAKER", "the guards", "speaker", number="plural"),),
                (it("SPEAKER", "a raven", "speaker", number="singular"),)]
    if symbol == "QSUBJ":
        return [(it("QSUBJ", "the king", "question-subject", number="singular", question=True),),
                (it("QSUBJ", "the queens", "question-subject", number="plural", question=True),),
                (it("QSUBJ", "a friend", "question-subject", number="singular", question=True),)]
    if symbol == "QOBJ":
        return [(it("QOBJ", "the letter", "question-object", question=True),),
                (it("QOBJ", "a bright sign", "question-object", question=True),),
                (it("QOBJ", "one true word", "question-object", question=True),)]
    if symbol == "QUESTION":
        rows=[]
        for subj in expand("QSUBJ", depth=depth+1):
            plural = subj[0].number == "plural"
            forms = (("do", "guard", "plural"), ("do", "read", "plural")) if plural else (("does", "keep", "singular"), ("does", "read", "singular"))
            for a, v, agreement in forms:
                for obj in expand("QOBJ", depth=depth+1):
                    rows.append(subj + (it("AUX", a, "question-aux", agreement=agreement, question=True),
                                        it("QVERB", v, "question-predicate", agreement=agreement, question=True)) + obj)
        return rows
    if symbol == "DIALOGUE":
        rows=[]
        for sp in expand("SPEAKER", depth=depth+1):
            for pred in ("asks", "wonders"):
                for q in expand("QUESTION", depth=depth+1):
                    rows.append(sp + (it("SPEECH", pred, "speech-valency"), it("QCOMP", "whether", "question-complement")) + q)
        return rows
    if symbol == "SCENE":
        rows=list(expand("DIALOGUE", depth=depth+1))
        for left in expand("DIALOGUE", depth=depth+1):
            for bridge in ("and", "while"):
                for right in expand("DIALOGUE", depth=depth+1):
                    rows.append(left + (it("BRIDGE", bridge, "coordination"),) + right)
        return rows
    return []

def complete_derivations() -> list[tuple[Item, ...]]:
    seen=set(); rows=[]
    for path in expand("SCENE"):
        key=tuple(x.text for x in path)
        if key not in seen: seen.add(key); rows.append(path)
    return rows

def run(*, state_limit=300_000) -> dict[str, object]:
    paths=complete_derivations(); states=pruned=advances=0; candidates=[]; witnesses=[]
    def pair(lp, rp):
        nonlocal states, pruned, advances
        def walk(li, ri, left, right, ls, rs, env):
            nonlocal states, pruned, advances
            if states >= state_limit: return
            if li >= len(lp) and ri < 0:
                if left or right: return
                ordered=ls+tuple(reversed(rs)); rendered=" ".join(x.text for x in ordered); checked=audit(rendered)
                if checked["exact"]:
                    candidates.append({"rendered": rendered, "audit": checked,
                        "provenance": {"construction": "typed question complement chart",
                            "symbols": [x.symbol for x in ordered], "valencies": [x.valency for x in ordered],
                            "question_features": [x.question for x in ordered], "variable_word_boundaries": True,
                            "finished_tape_reversal": False, "post_hoc_repair": False,
                            "catalogue_text": False, "aligned_token_mirror": False},
                        "reader_status": "unreviewed; exactness does not certify readability"})
                return
            states += 1
            if li >= len(lp) or ri < 0: return
            l, r=lp[li], rp[ri]; e=dict(env)
            if l.valency == "question-subject": e["ln"] = l.number or ""
            if r.valency == "question-subject": e["rn"] = r.number or ""
            if l.valency == "question-aux" and l.agreement != e.get("ln"): pruned += 1; return
            if r.valency == "question-aux" and r.agreement != e.get("rn"): pruned += 1; return
            residual=consume(left+letters(l.text), letters(r.text)+right)
            if residual is None:
                pruned += 1
                if len(witnesses)<20:
                    witnesses.append({"rendered": " ".join(x.text for x in ls+(l,)+(r,)+tuple(reversed(rs))), "depth": li, "audit": audit(" ".join(x.text for x in ls+(l,)+(r,)+tuple(reversed(rs)))), "reader_status": "diagnostic chart witness"})
                return
            advances += 1; walk(li+1, ri-1, residual[0], residual[1], ls+(l,), (r,)+rs, e)
        walk(0, len(rp)-1, "", "", (), (), {})
    controls=[{"rendered": " ".join(x.text for x in p), "audit": audit(" ".join(x.text for x in p)), "reader_status": "complete generated question-dialogue control; not exact"} for p in paths[:8]]
    for lp in paths:
        for rp in paths:
            pair(lp,rp)
            if states>=state_limit: break
        if states>=state_limit: break
    candidates.sort(key=lambda x:x["audit"]["letters"], reverse=True)
    result={"experiment": EXPERIMENT_ID, "method": "typed question complements in character chart", "complete_prose_controls": controls, "candidates": candidates, "witnesses": witnesses,
            "stats": {"grammar_paths": len(paths), "states": states, "pruned": pruned, "chart_advances": advances, "exact": len(candidates)},
            "provenance": {"complete_question_clauses": True, "subject_agreement_before_auxiliary": True, "speech_valency": True, "variable_word_boundaries": True, "independent_pointer_sha_audit": True,
                "finished_tape_reversal": False, "post_hoc_repair": False, "catalogue_text": False, "aligned_token_mirror": False,
                "novelty_preflight": "new interrogative complement grammar family", "next_construction": "add wh-question complements with typed extraction sites"}}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2)+"\n"); return result

if __name__ == "__main__": print(json.dumps(run(), indent=2))
