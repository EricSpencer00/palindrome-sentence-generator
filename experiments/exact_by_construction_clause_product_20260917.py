"""Exact-by-construction search over an optional-slot clause automaton.

The earlier slot products paired positions in a fixed template.  This lane
uses a finite acyclic grammar automaton instead: adjective, adverb, and
prepositional-phrase edges are optional, so the left and right character walks
can cross word boundaries at different places.  Character equality is checked
before a state is pushed; a completed path is a complete clause, not a prose
candidate filtered after the fact.

This is a construction experiment.  Programmatic checks can reject a shortcut
but cannot certify that a surviving sentence is readable to a person.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "exact-by-construction-clause-product-20260917.json"
MIN_LETTERS, MAX_LETTERS = 40, 100


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    char: str
    completed_word: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class Grammar:
    start: int
    end: int
    edges: tuple[Edge, ...]
    role_words: dict[str, tuple[str, ...]]
    role_transitions: tuple[tuple[int, str, int], ...]


def _word_edges(edges: list[Edge], next_node: list[int], source: int,
                target: int, word: str, role: str) -> None:
    cursor = source
    for index, char in enumerate(word):
        final = index == len(word) - 1
        if final:
            nxt = target
        else:
            nxt = next_node[0]
            next_node[0] += 1
        edges.append(Edge(cursor, nxt, char, word if final else None,
                          role if final else None))
        cursor = nxt


def build_clause_grammar() -> Grammar:
    """Build a fresh, typed clause grammar with optional modifiers."""
    # States are word-boundary states.  Character nodes are allocated above
    # this range, so the role automaton remains easy to inspect in the run.
    START, SUBJ_DET, SUBJ_ADJ, SUBJ_NOUN = 0, 1, 2, 3
    ADV, VERB, OBJ_DET, OBJ_ADJ, OBJ_NOUN = 4, 5, 6, 7, 8
    PREP, LOC_DET, LOC_ADJ, LOC_NOUN, END = 9, 10, 11, 12, 13
    next_node = [100]
    edges: list[Edge] = []
    role_words = {
        "subject_det": ("a", "an", "the"),
        "subject_adj": ("calm", "careful", "patient", "quiet", "young"),
        "subject": ("artist", "gardener", "keeper", "reader", "sailor", "teacher"),
        "adverb": ("carefully", "quietly", "slowly", "often"),
        "verb": ("carries", "draws", "marks", "opens", "plants", "records", "visits", "watches"),
        "object_det": ("a", "an", "the"),
        "object_adj": ("blue", "bright", "clean", "old", "small", "warm"),
        # Final letters deliberately cover the opening determiners so the
        # product can cross the outer seam instead of failing trivially at the
        # first character pair.
        "object": ("chart", "data", "garden", "lantern", "letter", "parcel", "path", "record", "window"),
        "prep": ("beside", "by", "near", "under"),
        "location_det": ("a", "the"),
        "location_adj": ("old", "public", "quiet", "small"),
        "location": ("archive", "arena", "earth", "garden", "harbor", "path", "port", "station", "window"),
    }
    transitions = [
        (START, "subject_det", SUBJ_DET),
        (SUBJ_DET, "subject_adj", SUBJ_ADJ),
        (SUBJ_DET, "subject", SUBJ_NOUN),
        (SUBJ_ADJ, "subject", SUBJ_NOUN),
        (SUBJ_NOUN, "adverb", ADV),
        (SUBJ_NOUN, "verb", VERB),
        (ADV, "verb", VERB),
        (VERB, "object_det", OBJ_DET),
        (OBJ_DET, "object_adj", OBJ_ADJ),
        (OBJ_DET, "object", OBJ_NOUN),
        (OBJ_ADJ, "object", OBJ_NOUN),
        # A transitive clause may end at the object or continue into a PP.
        (OBJ_NOUN, "prep", PREP),
        (OBJ_NOUN, "_end", END),
        (PREP, "location_det", LOC_DET),
        (LOC_DET, "location_adj", LOC_ADJ),
        (LOC_DET, "location", LOC_NOUN),
        (LOC_ADJ, "location", LOC_NOUN),
        (LOC_NOUN, "_end", END),
    ]
    for source, role, target in transitions:
        if role == "_end":
            continue
        for word in role_words[role]:
            _word_edges(edges, next_node, source, target, word, role)
    # The END transition is epsilon at the grammar level.  The product does
    # not permit epsilon character edges; instead, object/location final words
    # are compiled twice, once to END and once to the PP continuation.
    # Recompile the final lexical choices directly to END.
    for word in role_words["object"]:
        _word_edges(edges, next_node, OBJ_ADJ, END, word, "object")
        _word_edges(edges, next_node, OBJ_DET, END, word, "object")
    for word in role_words["location"]:
        _word_edges(edges, next_node, LOC_ADJ, END, word, "location")
        _word_edges(edges, next_node, LOC_DET, END, word, "location")
    parser_transitions = [
        (a, b, c) for a, b, c in transitions if b != "_end"
    ] + [
        (OBJ_DET, "object", END), (OBJ_ADJ, "object", END),
        (LOC_DET, "location", END), (LOC_ADJ, "location", END),
    ]
    return Grammar(START, END, tuple(edges), role_words,
                   tuple(parser_transitions))


def replay(grammar: Grammar, path: tuple[Edge, ...]) -> dict:
    cursor = grammar.start
    words: list[str] = []
    roles: list[str] = []
    tape: list[str] = []
    for edge in path:
        if edge.source != cursor:
            return {"ok": False, "reason": "disconnected_path"}
        cursor = edge.target
        tape.append(edge.char)
        if edge.completed_word is not None:
            words.append(edge.completed_word)
            roles.append(edge.role or "")
    text = " ".join(words)
    return {"ok": cursor == grammar.end, "words": words, "roles": roles,
            "tape": "".join(tape), "text": text,
            "exact": bool(tape) and tape == tape[::-1], "letters": len(tape)}


def independent_parse(grammar: Grammar, words: list[str]) -> dict:
    """Reparse word boundaries and role transitions without the product trace."""
    by_role = {role: set(forms) for role, forms in grammar.role_words.items()}
    # Dynamic programming over the word-boundary role graph.
    outgoing: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for source, role, target in grammar.role_transitions:
        outgoing[source].append((role, target))
    states: dict[int, tuple[str, ...]] = {grammar.start: ()}
    for word in words:
        nxt: dict[int, tuple[str, ...]] = {}
        for state, roles in states.items():
            for role, target in outgoing[state]:
                if word in by_role[role] and target not in nxt:
                    nxt[target] = roles + (role,)
        states = nxt
    # Object and location nouns may finish at END through the omitted epsilon.
    # Accept only paths whose final lexical role is a noun and whose sequence
    # has one subject, one transitive verb, and one object/location.
    accepted = []
    for state, roles in states.items():
        if state not in (8, 12):
            continue
        if "subject" in roles and "verb" in roles and ("object" in roles or "location" in roles):
            accepted.append(roles)
    return {"ok": bool(accepted), "role_paths": [list(x) for x in accepted]}


def _reachable(edges: tuple[Edge, ...], end: int) -> dict[int, set[int]]:
    outgoing: dict[int, list[Edge]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source].append(edge)
    cache: dict[int, set[int]] = {}

    def visit(node: int) -> set[int]:
        if node in cache:
            return cache[node]
        seen = {node}
        for edge in outgoing[node]:
            seen.update(visit(edge.target))
        cache[node] = seen
        return seen

    visit(end)
    for node in list(outgoing):
        visit(node)
    return cache


def exact_product(grammar: Grammar, max_states: int = 300_000) -> dict:
    outgoing: dict[int, list[Edge]] = defaultdict(list)
    incoming: dict[int, list[Edge]] = defaultdict(list)
    for edge in grammar.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)
    reachable = _reachable(grammar.edges, grammar.end)
    stack = [(grammar.start, grammar.end, (), ())]
    records: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    states = 0
    while stack and states < max_states:
        left, right, prefix, suffix = stack.pop()
        states += 1
        middle: list[tuple[Edge, ...]] = []
        if left == right:
            middle.append(())
        middle.extend((edge,) for edge in outgoing[left] if edge.target == right)
        for center in middle:
            path = prefix + center + suffix
            rec = replay(grammar, path)
            if not (rec["ok"] and rec["exact"]):
                continue
            key = tuple(rec["words"])
            if key in seen:
                continue
            seen.add(key)
            rec["center_characters"] = len(center)
            rec["independent_parse"] = independent_parse(grammar, rec["words"])
            records.append(rec)
        for first in outgoing[left]:
            for last in incoming[right]:
                if first.char != last.char:
                    continue
                if last.source not in reachable.get(first.target, set()):
                    continue
                stack.append((first.target, last.source,
                              prefix + (first,), (last,) + suffix))
    return {"states": states, "truncated": bool(stack), "records": records}


def anti_shortcut(words: list[str]) -> dict:
    tapes = [normalize(w) for w in words]
    word_order_mirror = tapes == [w[::-1] for w in reversed(tapes)]
    self_palindromic_spans = [w for w in tapes if len(w) > 2 and w == w[::-1]]
    return {"word_order_mirror": word_order_mirror,
            "self_palindromic_spans": self_palindromic_spans,
            "repeated_words": len(words) != len(set(words))}


def independent_audit(text: str) -> dict:
    tape = normalize(text)
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    return {"algorithm": "independent_two_pointer", "letters": len(tape),
            "exact": bool(tape) and not mismatches, "mismatches": mismatches,
            "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest()}


def withheld_control() -> dict:
    # This control validates the product kernel only.  It is never emitted as
    # a generated candidate and is not used to seed the fresh grammar run.
    words = ("an", "aide", "rips", "nine", "memos", "some", "men", "inspire", "diana")
    edges: list[Edge] = []
    nxt = [100]
    state = 0
    for index, word in enumerate(words):
        target = index + 1
        _word_edges(edges, nxt, state, target, word, f"control_{index}")
        state = target
    g = Grammar(0, len(words), tuple(edges), {}, ())
    result = exact_product(g, max_states=100_000)
    return {"control_id": "withheld_existing_38_letter_seed", "exact_paths": len(result["records"]),
            "not_a_generated_candidate": True}


def run(max_states: int = 300_000) -> dict:
    grammar = build_clause_grammar()
    product = exact_product(grammar, max_states=max_states)
    exact_records = []
    admitted = []
    for rec in product["records"]:
        text = rec["text"].capitalize() + "."
        audit = independent_audit(text)
        shortcut = anti_shortcut(rec["words"])
        parse = rec["independent_parse"]
        eligible = (MIN_LETTERS <= audit["letters"] <= MAX_LETTERS and
                    audit["exact"] and parse["ok"] and
                    not shortcut["word_order_mirror"] and
                    not shortcut["self_palindromic_spans"] and
                    not shortcut["repeated_words"])
        row = {"rendered": text, "words": rec["words"], "roles": rec["roles"],
               "letters": audit["letters"], "independent_audit": audit,
               "independent_parse": parse, "anti_shortcut": shortcut,
               "provenance": {"author_authored_lexicon": True, "catalogue_seed": False},
               "reader_status": "human-unreviewed; programmatic checks do not certify readability",
               "mechanically_admitted": eligible}
        exact_records.append(row)
        if eligible:
            admitted.append(row)
    result = {
        "status": "completed_no_admitted_closure" if not admitted else "completed_admitted_closure",
        "method": "exact_by_construction_optional_slot_clause_product",
        "grammar": {"states": 14, "optional_modifier_edges": True,
                     "independent_word_boundaries": True, "complete_clause_only": True},
        "search": {"max_states": max_states, "states": product["states"],
                   "truncated": product["truncated"], "min_letters": MIN_LETTERS,
                   "max_letters": MAX_LETTERS, "rlaif_per_candidate": False},
        "withheld_control": withheld_control(),
        "exact_candidates": exact_records, "mechanically_admitted": admitted,
        "reader_facing_next_test": (
            "If a closure is admitted, render it beside a shuffled-word control and run a randomized blinded reader test; "
            "otherwise expand the grammar with a second independent clause attachment and rerun the same product."),
        "next_repair": {"operator": "add_independent_clause_attachment",
                        "target": "increase lexical and syntactic reach without relaxing exact matching",
                        "reason": "the optional-slot clause product is finite and may exhaust before a 40-letter human-readable closure"},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "lexicon": "fresh authored ordinary-word inventory",
                       "catalogue_lookup": "none", "seed_used_in_output": False},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
