"""Search lexicalized dependency trees under an exact palindrome constraint.

This operator is deliberately narrower than free language generation but richer
than a linear POS template.  Each candidate is a complete dependency tree: a
person lets another person perform a licensed infinitival action on a named
person's concrete object.  Character equality is imposed outside-in, while a
prefix index makes every rejection an exact constraint contradiction rather
than a beam-score decision.

The vocabulary is typed and local.  It does not contain catalogue sentences
or use a corpus span as an output.  A surviving string has construction
provenance and hard mechanical validation only; it is not called readable
until it passes the separate blinded-reader protocol.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def letters(text: str) -> str:
    return re.sub("[^a-z]", "", text.lower())


def word_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall("[a-z]+", text.lower()))


# A frozen, typed vocabulary.  The names are intentionally ordinary names, not
# a list of known palindrome components.  Possessives are rendered as one word
# with an apostrophe only after exactness has been checked on letters.
PEOPLE = tuple("""
aaron abigail adam adrian alan alana albert alex alexander alexis alice alicia
amanda amber amelia amy andrea andrew angela anna anthony ariel arthur ashley
barbara ben benjamin beth betty brenda brian bridget bruce carl carla carol
caroline catherine charles chris christine claire dan daniel diana donna doris
edward elena elizabeth ellen emily eric erika eva frank fred gabriel george
grace greg hannah harry helen henry holly irene jack james jane janet jason jean
jennifer jeremy jessica joan john jonathan joseph julia julie justin karen kate
katherine kevin laura lauren leo leon linda lisa louise lucas lucy luke mark
maria marie mary matthew maureen megan michael michelle natalie neil nicholas
nicole nina noel nora oliver olivia pamela patricia paul peter rachel rebecca
richard robert rose ruby ruth sam samantha sarah scott sean sophia stephen steve
susan sylvia thomas timothy valerie victor victoria vincent walter wendy william
""".split())

OBJECTS = tuple("""
album anchor apron bag basket bell bicycle blanket book bottle box bracelet
brush camera candle canvas car carpet chair clock coat compass computer cup
diary door drawing envelope flag folder frame game garden gate guitar hammer
hat jacket key lamp letter map medal mirror model necklace notebook painting
paper parcel pencil photograph picture pillow postcard radio recipe report
ring rope rug scarf schedule sketch song suitcase table ticket toolbox toy
train vase wallet watch window
""".split())


@dataclass(frozen=True)
class Template:
    identifier: str
    finite_verb: str
    infinitive: str
    object_class: str

    def slots(self) -> tuple[tuple[str, ...], ...]:
        return (
            PEOPLE,
            (self.finite_verb,),
            PEOPLE,
            (self.infinitive,),
            tuple(f"{name}s" for name in PEOPLE),
            OBJECTS,
        )

    def render(self, chosen: tuple[str, ...]) -> str:
        subject, finite, recipient, action, owner, object_word = chosen
        owner_rendered = owner[:-1] + "'s"
        return f"{subject.capitalize()} {finite} {recipient} {action} {owner_rendered} {object_word}."

    def dependency_witness(self, chosen: tuple[str, ...]) -> dict:
        subject, finite, recipient, action, owner, object_word = chosen
        return {
            "root": finite,
            "nsubj": subject,
            "obj": recipient,
            "xcomp": action,
            "xcomp_obj": object_word,
            "possessor": owner,
            "selection": (
                f"{finite} licenses a human object and bare infinitive; "
                f"{action} licenses a concrete {self.object_class} object"
            ),
        }


TEMPLATES = (
    Template("let-see-possessive-object", "lets", "see", "visible object"),
    Template("let-find-possessive-object", "lets", "find", "concrete object"),
    Template("let-carry-possessive-object", "lets", "carry", "portable object"),
    Template("let-open-possessive-object", "lets", "open", "openable object"),
    Template("let-inspect-possessive-object", "lets", "inspect", "concrete object"),
)


@dataclass(frozen=True)
class Option:
    text: str
    tape: str
    tokens: tuple[str, ...]


class DirectionalIndex:
    """Exact lookup of domain values compatible with a current letter debt."""

    def __init__(self, values: tuple[str, ...], *, reverse: bool) -> None:
        self.options = tuple(
            Option(value, letters(value)[::-1] if reverse else letters(value), word_tokens(value))
            for value in values
        )
        self.by_prefix: dict[str, list[Option]] = defaultdict(list)
        self.by_tape: dict[str, list[Option]] = defaultdict(list)
        for option in self.options:
            self.by_tape[option.tape].append(option)
            for end in range(1, len(option.tape) + 1):
                self.by_prefix[option.tape[:end]].append(option)

    def compatible(self, debt: str) -> tuple[Option, ...]:
        """Return precisely the values whose leading letters match ``debt``."""
        if not debt:
            return self.options
        found = list(self.by_prefix.get(debt, ()))
        for end in range(1, len(debt)):
            found.extend(self.by_tape.get(debt[:end], ()))
        # A value can enter both branches only if its tape has zero length,
        # which is disallowed.  Deduplicate defensively for future vocab edits.
        return tuple(dict.fromkeys(found))


def solve_template(template: Template, *, min_letters: int = 30,
                   max_letters: int = 70) -> tuple[list[tuple[str, ...]], dict]:
    """Exhaust the template language with literal, outside-in cancellation."""
    domains = template.slots()
    forward = tuple(DirectionalIndex(domain, reverse=False) for domain in domains)
    backward = tuple(DirectionalIndex(domain, reverse=True) for domain in domains)
    lengths = tuple((min(len(option.tape) for option in index.options),
                     max(len(option.tape) for option in index.options)) for index in forward)
    stats = {
        "cartesian_derivations": _product(len(domain) for domain in domains),
        "states": 0,
        "prefix_rejections": 0,
        "repetition_rejections": 0,
        "length_rejections": 0,
        "closed_exact": 0,
    }
    selected: list[str | None] = [None] * len(domains)
    found: set[tuple[str, ...]] = set()

    def visit(lo: int, hi: int, debt: str, owner: int,
              used: frozenset[str], current_length: int) -> None:
        stats["states"] += 1
        possible_min = sum(lengths[index][0] for index in range(lo, hi + 1))
        possible_max = sum(lengths[index][1] for index in range(lo, hi + 1))
        if current_length + possible_min > max_letters or current_length + possible_max < min_letters:
            stats["length_rejections"] += 1
            return
        if lo > hi:
            if debt == debt[::-1]:
                row = tuple(value for value in selected if value is not None)
                rendered = template.render(row)
                assert letters(rendered) == letters(rendered)[::-1]
                found.add(row)
                stats["closed_exact"] += 1
            return
        side = -owner if debt else 1
        index = lo if side == 1 else hi
        candidates = (forward[index] if side == 1 else backward[index]).compatible(debt)
        if not candidates:
            stats["prefix_rejections"] += 1
            return
        for option in candidates:
            if (not option.tape or any(token == token[::-1] for token in option.tokens)
                    or used.intersection(option.tokens)):
                stats["repetition_rejections"] += 1
                continue
            shared = min(len(debt), len(option.tape))
            if debt[:shared] != option.tape[:shared]:
                raise AssertionError("prefix index admitted an incompatible option")
            if len(debt) > len(option.tape):
                new_debt, new_owner = debt[shared:], owner
            else:
                new_debt, new_owner = option.tape[shared:], side
            selected[index] = option.text
            visit(lo + (side == 1), hi - (side == -1), new_debt, new_owner,
                  used.union(option.tokens), current_length + len(option.tape))
            selected[index] = None

    visit(0, len(domains) - 1, "", 1, frozenset(), 0)
    return sorted(found), stats


def _product(values) -> int:
    answer = 1
    for value in values:
        answer *= value
    return answer


def checks(text: str, catalogue: set[str]) -> dict[str, bool]:
    from llm_palindrome.admission import mechanical_admission_checks

    return mechanical_admission_checks(
        text, local_catalogue=catalogue, min_letters=30, max_letters=70
    )


def run() -> dict:
    catalogue = set(json.loads((ROOT / "data" / "known_palindromes.json").read_text()))
    records, template_runs = [], []
    for template in TEMPLATES:
        solutions, stats = solve_template(template)
        template_runs.append({"template": template.identifier, **stats})
        for solution in solutions:
            text = template.render(solution)
            record_checks = checks(text, catalogue)
            records.append({
                "text": text,
                "letters": len(letters(text)),
                "template": template.identifier,
                "slots": solution,
                "dependency_witness": template.dependency_witness(solution),
                "checks": record_checks,
                "reader_status": "unreviewed; mechanics and a dependency witness do not establish readability",
            })
    admitted = [record for record in records if all(record["checks"].values())]
    return {
        "status": "complete_lexicalized_dependency_tree_intersection",
        "operator": "typed dependency trees plus exact outside-in character cancellation",
        "vocabulary_sha256": hashlib.sha256(
            json.dumps({"people": PEOPLE, "objects": OBJECTS, "templates": [t.identifier for t in TEMPLATES]},
                       sort_keys=True).encode()).hexdigest(),
        "template_runs": template_runs,
        "records": records,
        "mechanically_admitted": admitted,
        "scope": (
            "Exhaustive only for the frozen lexicalized dependency trees. A mechanically admitted "
            "record remains a reader-study candidate, not a readability claim."
        ),
        "next_construction_operator_if_empty": (
            "Expand a dependency tree through character-crossing relative-clause attachment: "
            "select a licensed object head first, then jointly lexicalize its left modifier and "
            "the reflected subject boundary. This changes the tree topology instead of widening "
            "the current word inventory or beam."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "templates": len(result["template_runs"]),
        "mechanically_admitted": len(result["mechanically_admitted"]),
    }, indent=2))


if __name__ == "__main__":
    main()
