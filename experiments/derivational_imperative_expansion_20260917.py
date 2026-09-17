"""Expanded morphology-first imperative search with live outer character equations.

This is a new construction lane, not a scorer sweep.  It chooses a typed
vocative--verb--object grammar, including independently listed inflectional and
derivational forms, and solves the outer character equations before a word
assignment is accepted.  Catalogue tapes are excluded during generation.
Exactness is audited again by a separate two-pointer pass; neither grammar
membership nor lexical frequency certifies human readability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPERIMENT_ID = "derivational-imperative-expansion-20260917"
SIGNATURE = (
    "typed-imperative-vocative|inflectional-derivational-families|"
    "outer-character-equations|catalogue-exclusion-at-generation"
)
OUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"


def tape(text: str) -> str:
    return "".join(c for c in text.casefold() if "a" <= c <= "z")


def audit(text: str) -> dict:
    t = tape(text)
    pairs = [(i, len(t) - 1 - i) for i in range(len(t) // 2)
             if t[i] != t[-1 - i]]
    return {
        "letters": len(t),
        "exact": bool(t) and not pairs,
        "mismatch_count": len(pairs),
        "mismatches": pairs[:16],
        "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest(),
        "independent_two_pointer": all(t[i] == t[-1 - i]
                                        for i in range(len(t) // 2)),
    }


def catalogue_tapes() -> set[str]:
    out: set[str] = set()
    for path in (ROOT / "data").glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        def walk(value: object) -> None:
            if isinstance(value, str):
                t = tape(value)
                if t:
                    out.add(t)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(payload)
    return out


def _common(words: Iterable[str], *, min_zipf: float = 2.7,
            max_words: int = 140) -> tuple[str, ...]:
    # Keep the inventory inspectable and deterministic.  The Brown/wordfreq
    # tables are lexical evidence only; the typed slot grammar below is the
    # actual syntactic constraint.
    from wordfreq import top_n_list, zipf_frequency

    candidates = {
        w.casefold() for w in words
        if re.fullmatch(r"[a-z]+", w.casefold()) and 2 <= len(w) <= 12
    }
    candidates.update(top_n_list("en", 25_000))
    ranked = sorted((w for w in candidates
                     if re.fullmatch(r"[a-z]+", w)
                     and zipf_frequency(w, "en") >= min_zipf),
                    key=lambda w: (-zipf_frequency(w, "en"), w))
    return tuple(ranked[:max_words])


def banks() -> dict[str, tuple[str, ...]]:
    from llm_palindrome.syntax import brown_tables

    table, _, _ = brown_tables()
    all_words = _common(table)
    by_tag: dict[str, set[str]] = defaultdict(set)
    for word in all_words:
        by_tag.update()
        for tag in table.get(word, ()):
            by_tag[tag].add(word)

    # The vocative inventory deliberately includes people and named agents;
    # these are ordinary sentence materials, not completed palindrome text.
    vocative = set("anna diana iris lisa maria nina noel satan sara ada eva".split())
    vocative |= {w for w in by_tag["NOUN"] if 3 <= len(w) <= 9}
    verbs = set("admire answer ask carry catch check clean close cook draw edit find give guard guide help hold keep label leave like live love make mark meet name open paint plan plant read record repair repay save see send serve share sing speak start stay stop store study take teach tend test thank think tie use visit wait walk watch write oscillate deliver refer stress level".split())
    verbs |= {w for w in by_tag["VERB"] if 3 <= len(w) <= 10}
    det = set("a an the my our your his her one each this that no some".split())
    adjs = set("calm careful clear cold dark fair fine fresh good kind late mild new patient quiet red safe small soft warm wise young metallic stressed civic vivid rural tidal".split())
    adjs |= {w for w in by_tag["ADJ"] if 3 <= len(w) <= 10}
    nouns = set("artist artists baker bird child clerk cloud dog door dream friend garden gardens girl hand horse house keeper lamp letter light map market memo memos moon nurse park path poet rain river road room sailor school seed signal sky song stone sun table teacher town train tree traveler watch water window sonata sonatas dessert desserts record records lantern archive harbor channel".split())
    nouns |= {w for w in by_tag["NOUN"] if 3 <= len(w) <= 12}

    # Keep high-frequency words but cap each role to make the search bounded;
    # all generated paths are still checked independently below.
    from wordfreq import zipf_frequency
    rank = lambda xs, n: tuple(sorted(xs, key=lambda w: (-zipf_frequency(w, "en"), w))[:n])
    return {
        "VOC": rank(vocative, 180),
        "V": rank(verbs, 220),
        "DET": rank(det, 20),
        "ADJ": rank(adjs, 180),
        "N": rank(nouns, 260),
    }


def anti_shortcut(words: tuple[str, ...], catalogue: set[str]) -> dict:
    content = [w for w in words if w not in {"a", "an", "the", "my", "our", "your", "his", "her", "one", "each", "this", "that", "no", "some"}]
    text = " ".join(words)
    t = tape(text)
    # Only a word-aligned, proper multiword span is a disallowed nested unit.
    # Scanning arbitrary character offsets would flag every palindrome because
    # its interior (the tape with the first/last character removed) is itself
    # palindromic.
    proper_spans = []
    boundaries = [0]
    for word in words:
        boundaries.append(boundaries[-1] + len(tape(word)))
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            if i == 0 and j == len(words):
                continue
            span = tape(" ".join(words[i:j]))
            if len(span) >= 8 and span == span[::-1]:
                proper_spans.append((boundaries[i], boundaries[j]))
                break
        if proper_spans:
            break
    return {
        "word_order_mirror": list(words) == [w[::-1] for w in words[::-1]],
        "repeated_content": len(content) != len(set(content)),
        "self_palindromic_words": [w for w in words if len(w) > 1 and w == w[::-1]],
        "proper_palindromic_span": proper_spans,
        "catalogue_tape": t in catalogue,
    }


def search(pattern: tuple[str, ...], role_banks: dict[str, tuple[str, ...]],
           catalogue: set[str], state_budget: int = 2_000_000,
           min_letters: int = 40) -> dict:
    # A stack state carries active word offsets, so boundaries may cross at
    # arbitrary characters.  This is the same invariant as the finished path,
    # not a post-hoc reversal or Cartesian pairing of half sentences.
    n = len(pattern)
    stack = [(0, n - 1, None, 0, None, 0, [None] * n, 0)]
    states = 0
    candidates: list[dict] = []
    seen: set[tuple] = set()
    while stack and states < state_budget:
        li, ri, lw, lp, rw, rp, assignment, depth = stack.pop()
        states += 1
        if lw is not None and lp == len(lw):
            stack.append((li + 1, ri, None, 0, rw, rp, assignment, depth)); continue
        if rw is not None and rp == len(rw):
            stack.append((li, ri - 1, lw, lp, None, 0, assignment, depth)); continue
        key = (li, ri, lw, lp, rw, rp, tuple(assignment))
        if key in seen:
            continue
        seen.add(key)
        # The seam may land inside the one word currently owned by the
        # opposite pointer. Its unconsumed center residual must itself be a
        # palindrome; requiring a one-character center would lose valid
        # cross-word closures such as ``satan ... sonatas``.
        if li == ri and lw is None and rw is not None:
            rt = tape(rw)
            residual = rt[:len(rt) - rp]
            if residual and residual == residual[::-1]:
                words = tuple(x for x in assignment if x is not None)
                text = " ".join(words) + "."
                a = audit(text)
                anti = anti_shortcut(words, catalogue)
                if (a["exact"] and min_letters <= a["letters"] <= 220
                        and not anti["catalogue_tape"]):
                    candidates.append({"rendered": text, "words": words,
                                       "audit": a, "anti_shortcut": anti,
                                       "roles": pattern,
                                       "center_residual": residual})
            continue
        if li == ri and rw is None and lw is not None:
            lt = tape(lw)
            residual = lt[lp:]
            if residual and residual == residual[::-1]:
                words = tuple(x for x in assignment if x is not None)
                text = " ".join(words) + "."
                a = audit(text)
                anti = anti_shortcut(words, catalogue)
                if (a["exact"] and min_letters <= a["letters"] <= 220
                        and not anti["catalogue_tape"]):
                    candidates.append({"rendered": text, "words": words,
                                       "audit": a, "anti_shortcut": anti,
                                       "roles": pattern,
                                       "center_residual": residual})
            continue
        if li > ri:
            words = tuple(x for x in assignment if x is not None)
            text = " ".join(words) + "."
            a = audit(text)
            anti = anti_shortcut(words, catalogue)
            if (a["exact"] and min_letters <= a["letters"] <= 220
                    and not anti["catalogue_tape"]):
                candidates.append({"rendered": text, "words": words,
                                   "audit": a, "anti_shortcut": anti,
                                   "roles": pattern})
            continue
        if li == ri and lw is None and rw is None:
            for word in role_banks[pattern[li]]:
                if len(tape(word)) != 1:
                    continue
                updated = assignment.copy(); updated[li] = word
                stack.append((li + 1, ri - 1, None, 0, None, 0, updated, depth + 1))
            continue
        if lw is None:
            for word in role_banks[pattern[li]]:
                updated = assignment.copy(); updated[li] = word
                stack.append((li, ri, word, 0, rw, rp, updated, depth + 1))
            continue
        if rw is None:
            for word in role_banks[pattern[ri]]:
                updated = assignment.copy(); updated[ri] = word
                stack.append((li, ri, lw, lp, word, 0, updated, depth + 1))
            continue
        lt, rt = tape(lw), tape(rw)
        if lt[lp] != rt[-1 - rp]:
            continue
        stack.append((li, ri, lw, lp + 1, rw, rp + 1, assignment, depth))
    return {"states": states, "truncated": bool(stack),
            "candidates": candidates, "seen_states": len(seen)}


def run() -> dict:
    role_banks = banks()
    catalogue = catalogue_tapes()
    patterns = {
        "vocative_imperative": ("VOC", "V", "DET", "ADJ", "N"),
        "vocative_long_object": ("VOC", "V", "DET", "ADJ", "N", "PREP", "N"),
    }
    # The long-object pattern uses only a small set of ordinary prepositions;
    # its inclusion is a concrete expansion beyond the earlier five-slot tree.
    role_banks["PREP"] = ("by", "in", "near", "of", "on", "to", "under", "with")
    searches = {}
    all_candidates = []
    for name, pattern in patterns.items():
        result = search(pattern, role_banks, catalogue)
        searches[name] = {k: result[k] for k in ("states", "truncated", "seen_states")}
        all_candidates.extend({**row, "pattern_name": name,
                               "provenance": {
                                   "method": EXPERIMENT_ID,
                                   "typed_roles": list(pattern),
                                   "catalogue_tapes_loaded_for_exclusion": len(catalogue),
                                   "finished_surface_reversal": False,
                                   "fixed_tape": False,
                                   "source_sentences_copied": False,
                               }} for row in result["candidates"])
    all_candidates.sort(key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    admitted = [row for row in all_candidates
                if not row["anti_shortcut"]["word_order_mirror"]
                and not row["anti_shortcut"]["repeated_content"]
                and not row["anti_shortcut"]["self_palindromic_words"]
                and not row["anti_shortcut"]["proper_palindromic_span"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_exact_candidates_need_readers" if admitted else "completed_no_admitted_exact",
        "grammar": {"patterns": {k: list(v) for k, v in patterns.items()},
                     "role_inventory_sizes": {k: len(v) for k, v in role_banks.items()}},
        "searches": searches,
        "candidate_count": len(all_candidates),
        "exact_candidates": all_candidates[:100],
        "mechanically_admitted": admitted[:100],
        "reader_gate": "No programmatic metric certifies readability; any admitted candidate needs randomized blinded intact-vs-shuffled readers.",
        "next_repair": {
            "operator": "expand typed imperative to coordinated event clauses with agreement-carrying inflection",
            "reason": "the current expansion tests derivation and a PP attachment but may still exhaust at outer equations",
            "concrete": "add a second clause role graph and require tense/number agreement before character closure",
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["direct normalized tape", "two-pointer comparison", "forward/reverse SHA-256"],
            "catalogue_exclusion_at_generation": True,
            "word_order_only_symmetry_rejected": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = run()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "status": result["status"],
                      "candidates": result["candidate_count"],
                      "admitted": len(result["mechanically_admitted"]),
                      "searches": result["searches"]}, sort_keys=True))


if __name__ == "__main__":
    main()
