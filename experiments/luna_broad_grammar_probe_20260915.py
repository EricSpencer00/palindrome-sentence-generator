"""Broad but compact POS-shape residual search.

This probe keeps grammatical shape selection in the exact cancellation loop.
It is intentionally a diagnostic: Brown supplies POS *types* and wordfreq
orders ordinary lexical choices, but no survivor is readability evidence.
The admission gate is applied to every exact closure before it is retained.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

from wordfreq import top_n_list, zipf_frequency
from nltk.corpus import brown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
MIN_LETTERS = 39


@dataclass(frozen=True)
class Plan:
    name: str
    tags: tuple[str, ...]


# These are complete, ordinary clause shapes rather than arbitrary POS salad.
# A few alternate openings/attachments are the intentionally small frontier.
PLANS = (
    Plan("pron_svo", ("PRON", "VERB", "DET", "NOUN")),
    Plan("det_svo", ("DET", "NOUN", "VERB", "DET", "NOUN")),
    Plan("pron_pp", ("PRON", "VERB", "ADP", "DET", "NOUN")),
    Plan("det_adj_svo", ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN")),
    Plan("noun_svo", ("NOUN", "VERB", "DET", "NOUN")),
    Plan("adv_pron_svo", ("ADV", "PRON", "VERB", "DET", "NOUN")),
    Plan("det_svo_pp", ("DET", "NOUN", "VERB", "ADP", "DET", "NOUN")),
    Plan("pron_vv", ("PRON", "VERB", "VERB", "DET", "NOUN")),
    # Discourse/reporting shapes observed in ordinary edited prose.  They
    # deliberately permit a complete thought to span punctuation while the
    # character matcher still operates on one uninterrupted tape.
    Plan("reporting_discourse", ("NOUN", "NOUN", "PRON", "VERB", "DET", "ADJ", "ADV")),
    Plan("verb_report", ("VERB", "DET", "NOUN", "PRON", "NOUN", "ADP", "NOUN")),
    Plan("noted_report", ("NOUN", "NOUN", "PRON", "VERB", "DET", "ADJ", "ADV")),
    Plan("report_tail", ("VERB", "DET", "NOUN", "PRON", "NOUN", "ADP", "NOUN")),
)


def lexical_table() -> dict[str, frozenset[str]]:
    table: dict[str, set[str]] = defaultdict(set)
    for sentence in brown.tagged_sents(tagset="universal"):
        for word, tag in sentence:
            if word.isascii() and word.isalpha():
                table[word.casefold()].add(tag)
    # A tiny hand-audited supplement covers ordinary inflections that Brown's
    # historical sample happens not to contain.  These are lexical *types*,
    # not borrowed sentence material.
    table["fatness"].add("NOUN")
    table["prevents"].add("VERB")
    return {word: frozenset(tags) for word, tags in table.items()}


def pools(table: dict[str, frozenset[str]], size: int = 650) -> dict[str, tuple[str, ...]]:
    # Keep only common lexical items but restore the closed-class words needed
    # by these shapes.  Self-palindromic content words are excluded up front.
    words = [w for w in top_n_list("en", 30000)
             if w.isascii() and w.isalpha() and w in table and len(w) >= 2
             and w == w.casefold() and w != w[::-1]
             and zipf_frequency(w, "en") >= 3.0]
    words = words[:size]
    out: dict[str, tuple[str, ...]] = {}
    for tag in {tag for plan in PLANS for tag in plan.tags}:
        vals = [w for w in words if tag in table[w]]
        out[tag] = tuple(vals)
    return out


def cancel(residual: str, emitted: str, owner: int):
    n = min(len(residual), len(emitted))
    if residual[:n] != emitted[:n]:
        return None
    if len(residual) > n:
        return residual[n:], owner
    if len(emitted) > n:
        return emitted[n:], -owner
    return "", 0


class RoleIndex:
    """Prefix index for role-specific words.

    The residual solver never needs a word whose first characters disagree
    with the live debt.  Indexing prefixes keeps the broad POS probe cheap
    enough to use a genuinely large ordinary vocabulary.
    """

    def __init__(self, words: Iterable[str]):
        self.words = tuple(words)
        self.by_prefix: dict[str, tuple[str, ...]] = {}
        prefixes: dict[str, list[str]] = defaultdict(list)
        for word in self.words:
            for width in range(1, len(word) + 1):
                prefixes.setdefault(word[:width], []).append(word)
        self.by_prefix = {key: tuple(vals) for key, vals in prefixes.items()}

    def matches(self, residual: str) -> tuple[str, ...]:
        if not residual:
            return self.words
        out: list[str] = []
        # Words swallowed by the debt, plus words that overrun it.
        for width in range(1, len(residual) + 1):
            out.extend(self.by_prefix.get(residual[:width], ()))
        return tuple(dict.fromkeys(out))


def search_pair(left: Plan, right: Plan, role_pools: dict[str, tuple[str, ...]],
                budget: int = 250_000) -> tuple[list[dict], dict]:
    results: list[dict] = []
    stack = [(0, len(right.tags) - 1, "", 0, (), ())]
    states = 0
    deepest = {"matched": 0, "left": (), "right": (), "residual": "", "owner": 0}
    indexes = {tag: RoleIndex(words) for tag, words in role_pools.items()}
    reverse_indexes = {tag: RoleIndex(word[::-1] for word in words)
                       for tag, words in role_pools.items()}
    while stack and states < budget:
        li, ri, residual, owner, lwords, rrev = stack.pop()
        states += 1
        matched = sum(len(w) for w in lwords + rrev) - len(residual)
        if matched > deepest["matched"]:
            deepest = {"matched": matched, "left": lwords,
                       "right": tuple(reversed(rrev)), "residual": residual,
                       "owner": owner}
        if li == len(left.tags) and ri < 0:
            if not residual:
                text = " ".join(lwords).capitalize() + "; " + " ".join(reversed(rrev)) + "."
                tape = normalize_letters(text)
                if len(tape) >= MIN_LETTERS:
                    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=180)
                    results.append({"text": text, "left_plan": left.name, "right_plan": right.name,
                                    "left_words": lwords, "right_words": tuple(reversed(rrev)),
                                    "letters": len(tape), "exact": tape == tape[::-1],
                                    "mechanical_checks": checks,
                                    "mechanically_eligible": all(checks.values())})
            continue
        # Break the initial symmetry deterministically: emit the left edge
        # first.  Allowing both edges while owner==0 would create unrelated
        # halves and falsely report a closure.
        if owner in (0, -1):
            if li >= len(left.tags):
                continue
            tag = left.tags[li]
            # reverse for deterministic DFS while retaining frequency ordering
            candidates = role_pools[tag] if owner == 0 else indexes[tag].matches(residual)
            for word in reversed(candidates):
                emitted = word
                if owner == 0:
                    stack.append((li + 1, ri, emitted, 1, lwords + (word,), rrev))
                else:
                    outcome = cancel(residual, emitted, -1)
                    if outcome is not None:
                        rem, new_owner = outcome
                        stack.append((li + 1, ri, rem, new_owner, lwords + (word,), rrev))
        if owner == 1:
            if ri < 0:
                continue
            tag = right.tags[ri]
            # Match the emitted reverse spelling, then recover its source word.
            emitted_matches = reverse_indexes[tag].matches(residual)
            for emitted in reversed(emitted_matches):
                word = emitted[::-1]
                outcome = cancel(residual, emitted, 1)
                if outcome is not None:
                    rem, new_owner = outcome
                    stack.append((li, ri - 1, rem, new_owner, lwords, rrev + (word,)))
    return results, {"states": states, "budget_exhausted": bool(stack), "deepest": deepest}


def run(size: int = 650, budget: int = 250_000) -> dict:
    table = lexical_table()
    role_pools = pools(table, size)
    exact: list[dict] = []
    searches = []
    for left, right in product(PLANS, repeat=2):
        rows, stats = search_pair(left, right, role_pools, budget)
        searches.append({"left": left.name, "right": right.name, **stats, "exact": len(rows)})
        exact.extend(rows)
    unique = {row["text"]: row for row in exact}
    return {
        "status": "broad_grammar_residual_probe_complete",
        "config": {"plans": len(PLANS), "pool_size": size, "state_budget_per_pair": budget,
                   "min_letters": MIN_LETTERS, "grammar_during_search": True},
        "pool_counts": {tag: len(vals) for tag, vals in role_pools.items()},
        "exact_closures": len(exact), "unique_exact_closures": len(unique),
        "mechanically_eligible": [row for row in unique.values() if row["mechanically_eligible"]],
        "exact_records": list(unique.values()), "searches": searches,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "material": "wordfreq-ranked Brown-tagged lexical items; no catalogue text"},
        "reader_gate": "Any mechanically eligible output still requires randomized blinded intact-prose and shuffled-control readers.",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=650)
    parser.add_argument("--budget", type=int, default=250_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite output")
    result = run(args.pool_size, args.budget)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exact": result["unique_exact_closures"],
                      "eligible": len(result["mechanically_eligible"])}, indent=2))
