"""Character-synchronous Earley-style chart for complete English scenes.

The chart first expands a small recursive grammar into complete derivation
items, then advances left and right items together while consuming whatever
character overlap is currently available.  Phrase boundaries are incidental
to the character state; no token mirror or finished-tape reversal is used.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "earley-scene-chart-orbit-20260920.json"
EXPERIMENT_ID = "earley-scene-chart-orbit-20260920"


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
    mode: str | None = None


def item(symbol: str, text: str, valency: str, *, number: str | None = None,
         agreement: str | None = None, mode: str | None = None) -> Item:
    return Item(symbol, text, valency, number, agreement, mode)


def expand(symbol: str, *, depth: int = 0, number: str | None = None) -> list[tuple[Item, ...]]:
    """Small recursive grammar; returned tuples are complete chart paths."""
    if depth > 3:
        return []
    if symbol == "NP":
        return [
            (item("NP", "the bard", "subject", number="singular"),),
            (item("NP", "the guards", "subject", number="plural"),),
            (item("NP", "a raven", "subject", number="singular"),),
            (item("NP", "the old book", "object", number="singular"),),
        ]
    if symbol == "LOC":
        return [(item("LOC", "in the hall", "locative"),),
                (item("LOC", "under the elm", "locative"),)]
    if symbol == "VP":
        if number == "plural":
            verbs = ("guard", "read")
            agreement = "plural"
        else:
            verbs = ("keeps", "reads")
            agreement = "singular"
        rows = [(item("VP", verb, "transitive", agreement=agreement, mode="finite"),
                 *np) for verb in verbs for np in expand("NP", depth=depth + 1)]
        rows += [(item("VP", "is", "copular", agreement=agreement, mode="finite"), *loc)
                 for loc in expand("LOC", depth=depth + 1)]
        return rows
    if symbol == "IMP":
        return [(item("IMP", "read the letter", "imperative", mode="imperative"),),
                (item("IMP", "guard the gate", "imperative", mode="imperative"),)]
    if symbol == "CLAUSE":
        rows: list[tuple[Item, ...]] = []
        for np in expand("NP", depth=depth + 1):
            n = np[0].number
            for vp in expand("VP", depth=depth + 1, number=n):
                # Complete subject/verb/object or subject/copula/location.
                rows.append(np + vp)
        rows.extend(expand("IMP", depth=depth + 1))
        return rows
    if symbol == "SCENE":
        rows = list(expand("CLAUSE", depth=depth + 1))
        for left in expand("CLAUSE", depth=depth + 1):
            for bridge in (item("BRIDGE", "while", "finite-complement"),
                           item("BRIDGE", "and", "coordination")):
                for right in expand("CLAUSE", depth=depth + 1):
                    rows.append(left + (bridge,) + right)
        return rows
    return []


def complete_derivations() -> list[tuple[Item, ...]]:
    # Deduplicate by rendered grammar path while preserving separate feature
    # items.  This is a chart inventory, not a catalogue-text input.
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
    states = pruned = chart_advances = 0
    candidates: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []

    # Chart state: (left path cursor, right path cursor, residual buffers,
    # feature environment).  Cursors advance over variable-length phrase
    # items; the seam only consumes characters, never whole tokens.
    def pair(left_path: tuple[Item, ...], right_path: tuple[Item, ...]) -> None:
        nonlocal states, pruned, chart_advances
        left_text = "".join(letters(x.text) for x in left_path)
        right_text = "".join(letters(x.text) for x in right_path)
        # Structural chart pairing: consume incrementally one item at a time
        # from opposite ends, retaining unequal character debt.
        def advance(li: int, ri: int, left: str, right: str,
                    selected_l: tuple[Item, ...], selected_r: tuple[Item, ...],
                    env: dict[str, str]) -> None:
            nonlocal states, pruned, chart_advances
            if states >= state_limit:
                return
            if li >= len(left_path) and ri < 0:
                if left or right:
                    return
                ordered = selected_l + tuple(reversed(selected_r))
                rendered = " ".join(x.text for x in ordered)
                checked = audit(rendered)
                if checked["exact"]:
                    candidates.append({"rendered": rendered, "audit": checked,
                        "provenance": {"construction": "character-synchronous recursive chart",
                            "symbols": [x.symbol for x in ordered],
                            "valencies": [x.valency for x in ordered],
                            "features": [{"number": x.number, "agreement": x.agreement,
                                          "mode": x.mode} for x in ordered],
                            "variable_word_boundaries": True, "finished_tape_reversal": False,
                            "post_hoc_repair": False, "catalogue_text": False,
                            "aligned_token_mirror": False},
                        "reader_status": "unreviewed; exactness does not certify readability"})
                return
            states += 1
            if li >= len(left_path) or ri < 0:
                return
            litem, ritem = left_path[li], right_path[ri]
            next_env = dict(env)
            if litem.symbol == "NP" and litem.valency == "subject":
                next_env["left_number"] = litem.number or ""
            if ritem.symbol == "NP" and ritem.valency == "subject":
                next_env["right_number"] = ritem.number or ""
            if litem.agreement and litem.agreement != next_env.get("left_number"):
                pruned += 1
                return
            if ritem.agreement and ritem.agreement != next_env.get("right_number"):
                pruned += 1
                return
            nl = left + letters(litem.text)
            nr = letters(ritem.text) + right
            residual = consume(nl, nr)
            if residual is None:
                pruned += 1
                if len(witnesses) < 20:
                    rendered = " ".join(x.text for x in selected_l + (litem,) + (ritem,) + tuple(reversed(selected_r)))
                    witnesses.append({"rendered": rendered, "depth": li,
                                      "audit": audit(rendered), "reader_status": "diagnostic chart witness"})
                return
            chart_advances += 1
            advance(li + 1, ri - 1, residual[0], residual[1],
                    selected_l + (litem,), (ritem,) + selected_r, next_env)
        advance(0, len(right_path) - 1, "", "", (), (), {})

    # Hold out every other complete derivation as a reader-facing control;
    # all are ordinary generated grammar paths, none borrowed text.
    controls = [{"rendered": " ".join(x.text for x in path),
                 "audit": audit(" ".join(x.text for x in path)),
                 "reader_status": "complete prose control; not an exact candidate"}
                for path in paths[:8]]
    for left_path in paths:
        for right_path in paths:
            pair(left_path, right_path)
            if states >= state_limit:
                break
        if states >= state_limit:
            break
    candidates.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    result = {"experiment": EXPERIMENT_ID,
              "method": "character-synchronous Earley-style recursive scene chart",
              "complete_prose_controls": controls, "candidates": candidates,
              "witnesses": witnesses,
              "stats": {"grammar_paths": len(paths), "states": states,
                        "pruned": pruned, "chart_advances": chart_advances,
                        "exact": len(candidates)},
              "provenance": {"recursive_complete_grammar": True,
                  "imperative_dialogue_alternative": True,
                  "copular_locative_alternative": True,
                  "variable_word_boundaries": True, "independent_pointer_sha_audit": True,
                  "finished_tape_reversal": False, "post_hoc_repair": False,
                  "catalogue_text": False, "aligned_token_mirror": False,
                  "novelty_preflight": "new recursive chart geometry; no prior lane imported",
                  "next_construction": "add recursive dialogue quotation complements to the chart while keeping clause features typed"}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
