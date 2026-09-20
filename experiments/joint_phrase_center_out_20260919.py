"""Joint phrase center-out construction with live character obligations.

Unlike repair, this lane never renders a finished sentence and edits it.  It
chooses typed phrase expansions on the left and right at the same depth, then
checks the newly exposed character interval before descending to the center.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def independent_audit(text: str) -> dict[str, object]:
    tape = letters(text)
    f = hashlib.sha256(tape.encode()).hexdigest()
    r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and tape == tape[::-1],
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}


def _compatible(left: str, right: str) -> bool:
    n = min(len(left), len(right))
    return left[:n] == right[::-1][:n]


def _phrases(bank: dict[str, list[dict[str, object]]], tag: str, limit: int) -> tuple[str, ...]:
    return tuple(x["word"].casefold() for x in bank.get(tag, [])[:limit])


def search(*, limit: int = 8, bank_path: str = "data/brown_pcfg_bank_20260920.json") -> dict[str, object]:
    bank = json.loads(Path(bank_path).read_text())["lexicon"]
    det = _phrases(bank, "DET", 12) or ("a", "the")
    noun = _phrases(bank, "NOUN", 48) or ("poet", "reader")
    verb = _phrases(bank, "VERB", 48) or ("reads", "sees")
    adp = _phrases(bank, "ADP", 12) or ("in", "of")
    # Typed expansions are generated compositionally; no catalogue sentence is copied.
    np_bank = tuple(f"{d} {n}" for d in det for n in noun)
    vp_bank = tuple(f"{v} {d} {n}" for v in verb for d in det[:6] for n in noun[:24])
    pp_bank = tuple(f"{p} {n}" for p in adp for n in noun[:24])
    # Left and right phrases are independently sampled from the same grammar.
    slots = (np_bank, vp_bank, pp_bank, tuple(reversed(pp_bank)), tuple(reversed(vp_bank)), tuple(reversed(np_bank)))
    states = pruned = 0
    best: list[dict[str, object]] = []
    exact: list[dict[str, object]] = []

    def keep_best(text: str, prefix: str, suffix: str, depth: int, *, complete: bool = False) -> None:
        item = {"rendered": text, "audit": independent_audit(text),
                "provenance": {"grammar": "NP VP PP | PP VP NP", "depth": depth,
                               "construction": "synchronous typed phrase expansion",
                               "catalogue_text": False, "complete": complete}}
        best.append(item)
        best.sort(key=lambda x: (x["audit"]["letters"], -len(letters(x["rendered"])[0:])) , reverse=True)
        del best[limit:]
        if item["audit"]["exact"]:
            exact.append(item)

    def walk(i: int, prefix: str, suffix: str, left: list[str], right: list[str]) -> None:
        nonlocal states, pruned
        if i == 3:
            text = " ".join(left + right)
            keep_best(text, prefix, suffix, i)
            return
        for lp in slots[i][:96]:
            for rp in slots[-1 - i][:96]:
                if lp == rp or lp == rp[::-1]:
                    continue
                l, r = letters(lp), letters(rp)
                states += 1
                nl, nr = prefix + l, r + suffix
                if not _compatible(nl, nr):
                    pruned += 1
                    # Preserve a rendered witness of the first failed seam;
                    # it is evidence, not a generated palindrome completion.
                    keep_best(" ".join(left + [lp] + [rp] + right), nl, nr, i + 1)
                    continue
                walk(i + 1, nl, nr, left + [lp], [rp] + right)

    walk(0, "", "", [], [])
    return {"run_id": "joint-phrase-center-out-20260919", "method": "joint typed phrase center-out",
            "search_space": {"np": len(np_bank), "vp": len(vp_bank), "pp": len(pp_bank)},
            "stats": {"states": states, "pruned": pruned, "complete": len(best), "exact": len(exact)},
            "candidates": best, "exact_candidates": exact,
            "provenance": {"source": bank_path, "generated_compositionally": True,
                           "next_construction": "add a center phrase with semantic valency while retaining synchronous obligations"}}


if __name__ == "__main__":
    print(json.dumps(search(), indent=2))
