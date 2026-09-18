"""Exhaustive whole-sentence search with typed semantic slots.

Each template supplies agreement and a restricted verb/object selection. The
solver assigns the outermost remaining slot opposite the unmatched letter
debt. Prefix contradiction is an exact impossibility proof, not a beam score.
The centre may cross any word or phrase. No catalogue text seeds construction.
Template membership remains a construction witness, never reader evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def letters(text: str) -> str:
    return re.sub("[^a-z]", "", text.lower())


def words(text: str) -> tuple[str, ...]:
    return tuple(re.findall("[a-z]+", text.lower()))


def solve(domains: list[list[str]], *, min_letters: int = 30,
          max_letters: int = 60) -> tuple[list[str], dict]:
    """Enumerate every palindrome in a finite concatenative sentence language.

    'debt' is read from the outside toward the centre on its owning side.
    Thus right-hand slot text is reversed before cancellation. Once no slots
    remain, the uncancelled central substring must itself be a palindrome.
    """
    options = [[(text, letters(text), words(text)) for text in domain]
               for domain in domains]
    bounds = [(min((len(t) for _, t, _ in domain), default=0),
               max((len(t) for _, t, _ in domain), default=0))
              for domain in options]
    stats = {"cartesian_derivations": math.prod(map(len, domains)),
             "states": 0, "prefix_rejections": 0, "unit_rejections": 0,
             "length_rejections": 0, "closed_exact": 0}
    found: set[str] = set()
    chosen = [""] * len(domains)

    def visit(lo: int, hi: int, debt: str, owner: int,
              used: frozenset[str], length: int) -> None:
        stats["states"] += 1
        minimum = sum(bounds[i][0] for i in range(lo, hi + 1))
        maximum = sum(bounds[i][1] for i in range(lo, hi + 1))
        if length + minimum > max_letters or length + maximum < min_letters:
            stats["length_rejections"] += 1
            return
        if lo > hi:
            if debt == debt[::-1]:
                text = " ".join(chosen)
                assert letters(text) == letters(text)[::-1]
                stats["closed_exact"] += 1
                found.add(text)
            return
        side = -owner if debt else 1
        index = lo if side == 1 else hi
        for text, tape, units in options[index]:
            if (len(units) != len(set(units)) or used.intersection(units)
                    or any(unit == unit[::-1] for unit in units)):
                stats["unit_rejections"] += 1
                continue
            incoming = tape if side == 1 else tape[::-1]
            width = min(len(debt), len(incoming))
            if debt[:width] != incoming[:width]:
                stats["prefix_rejections"] += 1
                continue
            if len(debt) > len(incoming):
                new_debt, new_owner = debt[width:], owner
            else:
                new_debt, new_owner = incoming[width:], side
            chosen[index] = text
            visit(lo + (side == 1), hi - (side == -1), new_debt,
                  new_owner, used.union(units), length + len(tape))

    visit(0, len(domains) - 1, "", 1, frozenset(), 0)
    return sorted(found), stats


def sentence_templates() -> list[dict]:
    """Finite, transparent grammar with plural agreement and selection classes."""
    determiners = "the my your our his her their these those some several".split()
    people = "bakers chefs cooks nurses doctors teachers artists dancers singers poets writers drivers sailors workers farmers students friends parents brothers sisters children guards clerks readers editors reporters visitors tourists soldiers pilots scientists engineers researchers builders carpenters painters strangers neighbours guests owners residents".split()
    animals = "dogs cats rats birds horses rabbits foxes wolves bears goats sheep cows pigs deer ducks geese hens chickens mice fish ants bees flies tigers lions monkeys apes".split()
    human_adj = "young old tired busy calm quiet hungry curious worried nervous careful cheerful polite honest clever stressed eager sleepy serious excited patient friendly gentle".split()
    animal_adj = "young old hungry wild tame sleepy frightened thirsty injured tired quiet gentle tiny large small nervous".split()
    thing_adj = "new old red blue green brown white black yellow small large heavy light broken wooden metal expensive cheap useful unusual ordinary delicate beautiful missing stolen".split()
    things = "books letters notes papers reports pictures paintings songs poems stories maps signs tools tables chairs boxes bags baskets plates cups glasses bottles doors windows keys locks toys lamps boats cars trucks bicycles trains watches clocks shoes coats hats shirts beds blankets mirrors desks radios cameras machines devices computers phones pencils pens brushes stones ropes sticks".split()
    food_adj = "fresh stale cold hot warm sweet bitter delicious plain frozen raw cooked dry ripe".split()
    foods = "food bread rice pasta spaghetti noodles meat beef pork fish cod trout salmon chicken fruit cheese soup stew salad butter cream milk water tea coffee juice honey sugar salt pepper flour jam toast pie cake chocolate wine beer corn wheat barley oats lettuce cabbage spinach broccoli garlic ginger yogurt ice".split()
    plural_foods = "apples pears plums peaches grapes berries cherries oranges lemons limes melons bananas nuts beans peas lentils carrots potatoes tomatoes onions peppers eggs cakes cookies biscuits sandwiches pancakes waffles desserts sweets snacks meals".split()
    templates = []
    groups = [
        ("human_artifact", people, human_adj,
         "find see notice inspect examine buy sell collect carry move lift drop hide keep use need want lose recover repair replace clean wash damage break build make paint photograph measure test check count sort arrange pack unpack store steal return deliver send bring hold remove share display touch take leave open close".split(),
         things, thing_adj, determiners),
        ("human_mass_food", people, human_adj,
         "find see buy sell carry keep need want prepare cook serve taste consume share deliver bring smell spill store waste request order".split(),
         foods, food_adj, "the my your our his her their some".split()),
        ("human_plural_food", people, human_adj,
         "find see buy sell carry keep need want prepare cook serve taste eat share deliver bring store waste request order count".split(),
         plural_foods, food_adj, determiners),
        ("animal_food", animals, animal_adj,
         "find see smell eat consume need want seek".split(),
         plural_foods, food_adj, "the my your our his her their some".split()),
        ("human_person", people, human_adj,
         "see notice meet greet help teach guide lead follow trust respect admire praise blame fear question visit invite call thank welcome watch hear remember recognize".split(),
         people, human_adj, determiners),
    ]
    for name, subjects, subj_adj, verbs, objects, obj_adj, obj_det in groups:
        for question, subject_modifier, object_modifier in itertools.product((False, True), repeat=3):
            # Bare plural subjects are grammatical and increase endpoint diversity.
            for subject_determiner in (False, True):
                domains = []
                if question:
                    domains.append(["do"])
                if subject_determiner:
                    domains.append(determiners)
                if subject_modifier:
                    domains.append(subj_adj)
                domains.extend([subjects, verbs, obj_det])
                if object_modifier:
                    domains.append(obj_adj)
                domains.append(objects)
                templates.append({"id": f"{name}:q{int(question)}:sd{int(subject_determiner)}:sm{int(subject_modifier)}:om{int(object_modifier)}",
                                  "selection_class": name, "question": question,
                                  "domains": domains})
    # A separate grammatical family changes the sentence endpoint from an
    # object noun to a finite predicate. Coordinated subjects agree in plural;
    # past-tense and auxiliary predicates are self-contained phrase domains.
    predicates = ["work", "rest", "sleep", "talk", "speak", "sing", "dance",
                  "run", "walk", "swim", "laugh", "smile", "cry", "wait",
                  "agree", "disagree", "arrive", "leave", "return", "depart",
                  "worked", "rested", "slept", "talked", "spoke", "sang",
                  "danced", "ran", "walked", "swam", "laughed", "smiled",
                  "cried", "waited", "agreed", "disagreed", "arrived", "left",
                  "returned", "departed", "have arrived", "have departed",
                  "have returned", "have agreed", "have rested", "have slept",
                  "will work", "will rest", "will sleep", "will return"]
    for coordination, subject_modifier, subject_determiner in itertools.product((False, True), repeat=3):
        domains = []
        if subject_determiner:
            domains.append(determiners)
        if subject_modifier:
            domains.append(human_adj)
        domains.append(people)
        if coordination:
            domains.extend([["and"], people])
        domains.append(predicates)
        templates.append({"id": f"human_intransitive:c{int(coordination)}:sd{int(subject_determiner)}:sm{int(subject_modifier)}",
                          "selection_class": "human_intransitive", "question": False,
                          "domains": domains})
    return templates


def audit(text: str, catalogue: set[str]) -> dict:
    from server.v3 import real_words
    from llm_palindrome.admission import mechanical_admission_checks
    units = words(text)
    shared = mechanical_admission_checks(
        text, local_catalogue=catalogue, min_letters=30, max_letters=60
    )
    return shared | {"exact_palindrome": shared["exact_letter_palindrome"],
            "lexicon_words": bool(units) and real_words(list(units)),
            "no_repeated_words": shared["distinct_words"],
            "no_self_palindromic_words": shared["no_self_palindromic_word"],
            "absent_from_local_catalogue": shared["local_catalogue_absent"]}


def run() -> dict:
    catalogue_path = ROOT / "data/known_palindromes.json"
    catalogue = set(json.loads(catalogue_path.read_text()))
    templates = sentence_templates()
    records, witnesses, template_runs = {}, {}, []
    for template in templates:
        texts, stats = solve(template["domains"])
        template_runs.append({"id": template["id"], **stats})
        for text in texts:
            rendered = text[0].upper() + text[1:] + ("?" if template["question"] else ".")
            witnesses.setdefault(rendered, []).append(template["id"])
            records[rendered] = {"text": rendered, "letters": len(letters(rendered)),
                                 "checks": audit(rendered, catalogue),
                                 "reader_status": "unreviewed; do not claim readable"}
    for text, row in records.items():
        row["construction_witnesses"] = witnesses[text]
    return {"status": "exhaustive_finite_semantic_template_intersection_complete",
            "template_sha256": hashlib.sha256(json.dumps(templates, sort_keys=True).encode()).hexdigest(),
            "catalogue_sha256": hashlib.sha256(catalogue_path.read_bytes()).hexdigest(),
            "templates": templates, "template_runs": template_runs,
            "totals": {key: sum(row[key] for row in template_runs)
                       for key in template_runs[0] if key != "id"},
            "records": list(records.values()),
            "mechanically_admitted": [row for row in records.values() if all(row["checks"].values())],
            "scope": "An exhaustive negative result applies only to this finite grammar. Semantic selection restricts combinations but does not prove every derivation natural. Local catalogue absence is not global novelty."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "totals": result["totals"],
                      "records": result["records"]}, indent=2))
