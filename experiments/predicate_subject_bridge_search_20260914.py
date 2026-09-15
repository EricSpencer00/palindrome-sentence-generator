"""Large exact phrase bridge over grammatical subject/name endpoint factors.

The search varies determiner-person openings and terminal names; its one known
productive path is ``Diana -> an aide`` with a live ``e`` debt.  It jointly
searches a singular transitive predicate plus object NP against a plural
subject NP plus verb, with Brown supplying POS inventories but never complete
output text. Word and phrase boundaries may stagger during exact cancellation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dual_plan_residual_search_20260914 import cancel
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


PEOPLE = tuple("aide artist author baker captain clerk doctor driver editor farmer friend gardener leader maker mother father neighbor nurse owner pilot player reader sailor student teacher visitor worker writer".split())
PEOPLE_PL = tuple("aides artists authors bakers captains clerks doctors drivers editors farmers friends gardeners leaders makers mothers fathers neighbors nurses owners pilots players readers sailors students teachers visitors workers writers men women people".split())
NAMES = tuple("Aidan Alan Anna Anne Ari Ava Diana Eva Ira Leon Liam Mara Mia Nadia Nina Noel Nora Otis Ron Sara Tessa".split())
DETERMINERS = tuple("a an the this that my our".split())
HEADS = tuple("a an the this that my our her his their some many few several one two three four five six seven eight nine ten".split())


def _brown_top(tags: set[str], limit: int, min_zipf: float = 3.2) -> tuple[str, ...]:
    from nltk.corpus import brown
    from wordfreq import zipf_frequency

    counts = Counter()
    for word, raw_tag in brown.tagged_words():
        word = word.casefold()
        tag = raw_tag.split("-")[0]
        if (tag in tags and word.isascii() and word.isalpha() and len(word) > 1
                and zipf_frequency(word, "en") >= min_zipf):
            counts[word] += 1
    return tuple(word for word, _ in counts.most_common(limit))


def build_pools(*, noun_limit: int = 350, adjective_limit: int = 60,
                verb_limit: int = 180) -> tuple[tuple[tuple[str, ...], ...],
                                                 tuple[tuple[str, ...], ...], dict]:
    singular_nouns = _brown_top({"NN"}, noun_limit)
    plural_nouns = _brown_top({"NNS"}, noun_limit)
    adjectives = _brown_top({"JJ"}, adjective_limit)
    singular_verbs = tuple(dict.fromkeys(("rips",) + _brown_top({"VBZ"}, verb_limit)))
    plural_verbs = tuple(dict.fromkeys(("inspire",) + _brown_top({"VB", "VBP"}, verb_limit)))

    objects = {f"{head} {noun}" for head in HEADS for noun in singular_nouns + plural_nouns}
    objects.update(f"{adjective} {noun}" for adjective in adjectives
                   for noun in singular_nouns + plural_nouns)
    objects.update(f"{det} {adjective} {noun}"
                   for det in ("a", "an", "the", "my", "our", "her", "his", "their")
                   for adjective in adjectives for noun in singular_nouns[:120] + plural_nouns[:120])
    objects.update(("nine memos", "nine notes", "ten letters", "some papers", "the old letter"))

    subjects = {f"{head} {person}" for head in
                "the these those some many few several two three four five six seven eight nine ten".split()
                for person in PEOPLE_PL}
    subjects.update(f"{adjective} {person}" for adjective in adjectives for person in PEOPLE_PL)
    subjects.update(f"{det} {adjective} {person}"
                    for det in ("the", "my", "our", "her", "his", "their")
                    for adjective in adjectives for person in PEOPLE_PL)
    subjects.add("some men")

    left = (DETERMINERS, PEOPLE, singular_verbs, tuple(sorted(objects)))
    right = (tuple(sorted(subjects)), plural_verbs, NAMES)
    metadata = {"singular_nouns": len(singular_nouns), "plural_nouns": len(plural_nouns),
                "adjectives": len(adjectives), "singular_verbs": len(singular_verbs),
                "plural_verbs": len(plural_verbs), "object_phrases": len(objects),
                "subject_phrases": len(subjects)}
    return left, right, metadata


def indexed_search(left_pools: tuple[tuple[str, ...], ...],
                   right_pools: tuple[tuple[str, ...], ...], *,
                   state_budget: int = 2_000_000) -> tuple[list[tuple[tuple[str, ...], tuple[str, ...]]], dict]:
    left_index, right_index = [], []
    for pool in left_pools:
        index = defaultdict(list)
        for word in pool:
            index[normalize_letters(word)[0]].append(word)
        left_index.append(index)
    for pool in right_pools:
        index = defaultdict(list)
        for word in pool:
            index[normalize_letters(word)[-1]].append(word)
        right_index.append(index)

    stack = [(0, len(right_pools) - 1, "", 0, (), (), 0)]
    results = []
    states = 0
    deepest = {"matched_letters": 0, "left_words": (), "right_words": (),
               "residual": "", "owner": 0}
    while stack and states < state_budget:
        li, ri, residual, owner, left_words, right_reversed, matched = stack.pop()
        states += 1
        if matched > deepest["matched_letters"]:
            deepest = {"matched_letters": matched, "left_words": left_words,
                       "right_words": tuple(reversed(right_reversed)),
                       "residual": residual, "owner": owner}
        if li == len(left_pools) and ri < 0:
            if not residual:
                results.append((left_words, tuple(reversed(right_reversed))))
            continue
        if owner == 0:
            if li < len(left_pools):
                for word in left_pools[li]:
                    stack.append((li + 1, ri, normalize_letters(word), 1,
                                  left_words + (word,), right_reversed, matched))
        elif owner == 1:
            if ri >= 0:
                for word in right_index[ri].get(residual[0], ()):
                    emitted = normalize_letters(word)[::-1]
                    outcome = cancel(residual, emitted, 1)
                    if outcome is not None:
                        debt, debt_owner = outcome
                        stack.append((li, ri - 1, debt, debt_owner, left_words,
                                      right_reversed + (word,),
                                      matched + min(len(residual), len(emitted))))
        elif li < len(left_pools):
            for word in left_index[li].get(residual[0], ()):
                emitted = normalize_letters(word)
                outcome = cancel(residual, emitted, -1)
                if outcome is not None:
                    debt, debt_owner = outcome
                    stack.append((li + 1, ri, debt, debt_owner,
                                  left_words + (word,), right_reversed,
                                  matched + min(len(residual), len(emitted))))
    return results, {"states": states, "budget_exhausted": bool(stack), "deepest": deepest}


def independent_audit(text: str) -> dict:
    tape = "".join(char.casefold() for char in text if char.isascii() and char.isalpha())
    bad = [[i, len(tape) - 1 - i] for i in range(len(tape) // 2)
           if tape[i] != tape[-1 - i]]
    return {"normalized": tape, "letters": len(tape), "mismatches": bad,
            "exact": bool(tape) and not bad}


def run() -> dict:
    left, right, inventory = build_pools()
    pairs, stats = indexed_search(left, right)
    records = []
    for left_words, right_words in pairs:
        text = " ".join(left_words).capitalize() + "; " + " ".join(right_words) + "."
        audit = independent_audit(text)
        checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
        records.append({"text": text, "audit": audit, "mechanical_checks": checks,
                        "mechanically_eligible": all(checks.values()),
                        "reader_status": "human-unreviewed"})
    return {"status": "predicate_subject_phrase_bridge_complete",
            "config": {"opening_endpoint_combinations": len(DETERMINERS) * len(PEOPLE) * len(NAMES),
                       "known_productive_factor": "Diana -> an aide with live one-letter debt",
                       "state_budget": 2_000_000, "grammar_during_search": True,
                       "brown_complete_sentences_copied": False},
            "inventory": inventory, "search": stats, "exact_records": records,
            "eligible_closures": [row for row in records if row["mechanically_eligible"]],
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "material": "Brown POS word inventories plus authored semantic person/endpoints"},
            "next_operator_if_no_novel": "Generalize endpoint roles beyond determiner-person openings and proper-name objects: add typed adjectival/name/pronoun subjects and multiword terminal object NPs, then reuse this indexed predicate bridge for every live residual.",
            "reader_next": "Only a novel mechanically eligible closure enters blinded intact-prose versus shuffled-control ratings."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite output")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": result["inventory"],
                      "states": result["search"]["states"],
                      "exact": len(result["exact_records"]),
                      "eligible": len(result["eligible_closures"])}, indent=2))


if __name__ == "__main__":
    main()
