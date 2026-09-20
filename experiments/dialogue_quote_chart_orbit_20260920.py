"""Typed dialogue-quotation grammar with character-synchronous seam chart."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dialogue-quote-chart-orbit-20260920.json"
EXPERIMENT_ID = "dialogue-quote-chart-orbit-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    bad = [(i, len(tape) - i - 1) for i in range(len(tape) // 2)
           if tape[i] != tape[-i - 1]]
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not bad,
            "first_mismatch": bad[0] if bad else None,
            "sha256_forward": fwd, "sha256_reverse": rev,
            "sha_equal": fwd == rev}


def consume(left: str, right: str) -> tuple[str, str] | None:
    n = min(len(left), len(right))
    if n and left[:n] != right[-n:][::-1]:
        return None
    return left[n:], right[:-n] if n else right


@dataclass(frozen=True)
class Item:
    symbol: str
    text: str
    valency: str
    number: str | None = None
    agreement: str | None = None
    quote_depth: int = 0


def it(symbol: str, text: str, valency: str, *, number: str | None = None,
       agreement: str | None = None, quote_depth: int = 0) -> Item:
    return Item(symbol, text, valency, number, agreement, quote_depth)


def expand(symbol: str, *, depth: int = 0, number: str | None = None,
           quote_depth: int = 0) -> list[tuple[Item, ...]]:
    if depth > 6:
        return []
    if symbol == "SPEAKER":
        return [(it("SPEAKER", "the bard", "speaker", number="singular"),),
                (it("SPEAKER", "the guards", "speaker", number="plural"),),
                (it("SPEAKER", "a raven", "speaker", number="singular"),)]
    if symbol == "QNP":
        return [(it("QNP", "the king", "quote-subject", number="singular", quote_depth=quote_depth),),
                (it("QNP", "the queens", "quote-subject", number="plural", quote_depth=quote_depth),),
                (it("QNP", "a friend", "quote-subject", number="singular", quote_depth=quote_depth),),
                (it("QNP", "the old book", "quote-object", number="singular", quote_depth=quote_depth),)]
    if symbol == "QOBJ":
        return [(it("QOBJ", "the letter", "quote-object", quote_depth=quote_depth),),
                (it("QOBJ", "a bright sign", "quote-object", quote_depth=quote_depth),),
                (it("QOBJ", "one true word", "quote-object", quote_depth=quote_depth),)]
    if symbol == "QVP":
        verbs = (("keeps", "singular"), ("reads", "singular")) if number != "plural" else (("keep", "plural"), ("read", "plural"))
        rows: list[tuple[Item, ...]] = []
        for verb, agreement in verbs:
            for obj in expand("QOBJ", depth=depth + 1, quote_depth=quote_depth):
                rows.append((it("QVP", verb, "quote-transitive", agreement=agreement, quote_depth=quote_depth),) + obj)
        return rows
    if symbol == "QUOTE":
        rows: list[tuple[Item, ...]] = []
        for subject in expand("QNP", depth=depth + 1, quote_depth=quote_depth):
            for vp in expand("QVP", depth=depth + 1, number=subject[0].number, quote_depth=quote_depth):
                rows.append(subject + vp)
        return rows
    if symbol == "DIALOGUE":
        rows: list[tuple[Item, ...]] = []
        for speaker in expand("SPEAKER", depth=depth + 1):
            for predicate in (it("SAY", "says", "speech-predicate"),
                              it("SAY", "asks", "speech-predicate")):
                for quote in expand("QUOTE", depth=depth + 1, quote_depth=1):
                    rows.append(speaker + (predicate, it("QUOTE_OPEN", "that", "quote-complement"),) + quote)
        return rows
    if symbol == "SCENE":
        rows = list(expand("DIALOGUE", depth=depth + 1))
        for left in expand("DIALOGUE", depth=depth + 1):
            for bridge in (it("BRIDGE", "and", "coordination"), it("BRIDGE", "while", "finite-complement")):
                for right in expand("DIALOGUE", depth=depth + 1):
                    rows.append(left + (bridge,) + right)
        return rows
    return []


def complete_derivations() -> list[tuple[Item, ...]]:
    seen: set[tuple[str, ...]] = set()
    rows: list[tuple[Item, ...]] = []
    for path in expand("SCENE"):
        key = tuple(x.text for x in path)
        if key not in seen:
            seen.add(key)
            rows.append(path)
    return rows


def run(*, state_limit: int = 300_000) -> dict[str, object]:
    paths = complete_derivations()
    states = pruned = advances = 0
    candidates: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []

    def pair(lp: tuple[Item, ...], rp: tuple[Item, ...]) -> None:
        nonlocal states, pruned, advances
        def walk(li: int, ri: int, left: str, right: str,
                 ls: tuple[Item, ...], rs: tuple[Item, ...], env: dict[str, str]) -> None:
            nonlocal states, pruned, advances
            if states >= state_limit:
                return
            if li >= len(lp) and ri < 0:
                if left or right:
                    return
                ordered = ls + tuple(reversed(rs))
                rendered = " ".join(x.text for x in ordered)
                checked = audit(rendered)
                if checked["exact"]:
                    candidates.append({"rendered": rendered, "audit": checked,
                        "provenance": {"construction": "typed dialogue quotation chart",
                            "symbols": [x.symbol for x in ordered],
                            "valencies": [x.valency for x in ordered],
                            "quote_depths": [x.quote_depth for x in ordered],
                            "variable_word_boundaries": True,
                            "finished_tape_reversal": False, "post_hoc_repair": False,
                            "catalogue_text": False, "aligned_token_mirror": False},
                        "reader_status": "unreviewed; exactness does not certify readability"})
                return
            states += 1
            if li >= len(lp) or ri < 0:
                return
            litem, ritem = lp[li], rp[ri]
            next_env = dict(env)
            if litem.valency == "quote-subject":
                next_env["left_quote_number"] = litem.number or ""
            if ritem.valency == "quote-subject":
                next_env["right_quote_number"] = ritem.number or ""
            if litem.valency == "quote-transitive" and litem.agreement != next_env.get("left_quote_number"):
                pruned += 1
                return
            if ritem.valency == "quote-transitive" and ritem.agreement != next_env.get("right_quote_number"):
                pruned += 1
                return
            residual = consume(left + letters(litem.text), letters(ritem.text) + right)
            if residual is None:
                pruned += 1
                if len(witnesses) < 20:
                    rendered = " ".join(x.text for x in ls + (litem,) + (ritem,) + tuple(reversed(rs)))
                    witnesses.append({"rendered": rendered, "depth": li, "audit": audit(rendered), "reader_status": "diagnostic chart witness"})
                return
            advances += 1
            walk(li + 1, ri - 1, residual[0], residual[1], ls + (litem,), (ritem,) + rs, next_env)
        walk(0, len(rp) - 1, "", "", (), (), {})

    controls = [{"rendered": " ".join(x.text for x in path),
                 "audit": audit(" ".join(x.text for x in path)),
                 "reader_status": "complete generated dialogue control; not an exact candidate"}
                for path in paths[:8]]
    for lp in paths:
        for rp in paths:
            pair(lp, rp)
            if states >= state_limit:
                break
        if states >= state_limit:
            break
    candidates.sort(key=lambda x: x["audit"]["letters"], reverse=True)
    result = {"experiment": EXPERIMENT_ID,
              "method": "typed dialogue quotation complements in a character chart",
              "complete_prose_controls": controls, "candidates": candidates,
              "witnesses": witnesses,
              "stats": {"grammar_paths": len(paths), "states": states,
                        "pruned": pruned, "chart_advances": advances,
                        "exact": len(candidates)},
              "provenance": {"complete_quote_clauses": True, "quote_agreement_carried": True,
                  "quote_valency_carried": True, "variable_word_boundaries": True,
                  "independent_pointer_sha_audit": True, "finished_tape_reversal": False,
                  "post_hoc_repair": False, "catalogue_text": False,
                  "aligned_token_mirror": False,
                  "novelty_preflight": "new dialogue-complement grammar family",
                  "next_construction": "allow a typed question complement while retaining quote agreement state"}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
