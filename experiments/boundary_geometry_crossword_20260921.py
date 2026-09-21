"""Choose irreducible word-boundary geometry, then solve lexical overlaps.

No complete sentence or palindrome seed supplies search letters. Word lengths
and grammatical roles select six fixed overlap graphs before lexical search.
Every mirrored letter equation is then a constraint between word variables.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
import re

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("name", "adverb", "request", "determiner", "person", "to", "action", "determiner", "adjective", "artifact")
# Complete contemporary-English request/complement construction. Each bank is
# authored independently of the palindrome equation; no benchmark words are
# inserted as a phrase or retained as a candidate scaffold.
BANK = {
    "name": "Amy Ben Dan Eva Ian Kim Leo Max Sam Zoe Emma Iris Jane June Leah Lily Lucy Nora Ruby Vera Alice Clara Grace Julia Laura Maria Susan Alicia Gloria Helena Louise Monica Nicole Olivia Pamela Regina Serena Teresa".lower().split(),
    "adverb": "boldly calmly gently kindly loudly merely neatly rarely safely slowly warmly briefly clearly eagerly happily quickly quietly usually politely carefully patiently privately".split(),
    "request": "asks tells urges orders invites".split(),
    "determiner": ["a", "an", "the"],
    "person": "aide clerk artist editor intern reader worker writer painter scholar reporter".split(),
    "to": ["to"],
    "action": "copy edit file read check label print study repair return revise".split(),
    "adjective": "new old blue fair pale wide brief clear final large short small formal sealed secret careful private complete detailed weathered".split(),
    "artifact": "map file list memo note page chart draft letter record report account message outline summary document".split(),
}
FUNCTION = frozenset({"a", "an", "the", "to"})


def normalize(text):
    return "".join(c.lower() for c in text if "A" <= c <= "Z" or "a" <= c <= "z")


def audit(text):
    tape = normalize(text)
    lo, hi = 0, len(tape) - 1
    mismatches = []
    while lo < hi:
        if tape[lo] != tape[hi]:
            mismatches.append([lo, hi, tape[lo], tape[hi]])
        lo += 1
        hi -= 1
    return {"letters": len(tape), "normalized": tape,
            "pointer_exact": bool(tape) and not mismatches,
            "mismatches": mismatches,
            "sha256_forward": sha256(tape.encode()).hexdigest(),
            "sha256_reverse": sha256(tape[::-1].encode()).hexdigest()}


@dataclass(frozen=True)
class Geometry:
    widths: tuple[int, ...]

    @property
    def length(self):
        return sum(self.widths)

    @property
    def boundaries(self):
        total = 0
        result = []
        for width in self.widths[:-1]:
            total += width
            result.append(total)
        return tuple(result)

    @property
    def reflected_boundary_pairs(self):
        boundaries = set(self.boundaries)
        return tuple((b, self.length-b) for b in sorted(boundaries)
                     if b < self.length-b and self.length-b in boundaries)

    @property
    def owners(self):
        return tuple((slot, offset) for slot, width in enumerate(self.widths)
                     for offset in range(width))

    @property
    def equations(self):
        owners = self.owners
        return tuple((*owners[i], *owners[-1-i]) for i in range(self.length // 2))

    @property
    def connected(self):
        reached = {0}
        while True:
            expanded = reached | {other for left, _, right, _ in self.equations
                                  for slot, other in ((left, right), (right, left))
                                  if slot in reached}
            if expanded == reached:
                return len(reached) == len(self.widths)
            reached = expanded

    def as_dict(self):
        return {"widths": self.widths, "letters": self.length,
                "boundaries": self.boundaries,
                "reflected_boundary_pairs": self.reflected_boundary_pairs,
                "connected": self.connected,
                "equations": self.equations}


def frozen_geometries():
    """Length-only selection; letters never influence which graph is chosen."""
    options = [sorted({len(word) for word in BANK[role]}) for role in ROLES]
    rng = random.Random(20260921)
    graphs = []
    for target in (41, 43, 47):
        selected = set()
        for _ in range(20_000):
            widths = tuple(rng.choice(lengths) for lengths in options)
            if sum(widths) != target or widths in selected:
                continue
            graph = Geometry(widths)
            if graph.reflected_boundary_pairs or not graph.connected:
                continue
            selected.add(widths)
            graphs.append(graph)
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise RuntimeError(f"could_not_freeze_two_geometries_at_{target}")
    return tuple(graphs)


def initial_domains(graph):
    return tuple(tuple(word for word in BANK[role] if len(word) == width
                       and (len(word) == 1 or word != word[::-1]))
                 for role, width in zip(ROLES, graph.widths))


def pair_valid(i, a, j, b, equations):
    for left, lc, right, rc in equations:
        if left == i and right == j and a[lc] != b[rc]:
            return False
        if left == j and right == i and b[lc] != a[rc]:
            return False
    if a == b and a not in FUNCTION:
        return False
    # Both determiner phrases preserve their immediate vowel/consonant rule.
    for det, head in ((3, 4), (7, 8)):
        if (i, j) == (det, head):
            if a == "a" and b[0] in "aeiou" or a == "an" and b[0] not in "aeiou":
                return False
        if (j, i) == (det, head):
            if b == "a" and a[0] in "aeiou" or b == "an" and a[0] not in "aeiou":
                return False
    return True


def propagate(graph, domains, *, equations=True):
    """AC-3 style fixed point, with an explicit first empty-domain witness."""
    active = graph.equations if equations else ()
    domains = [list(words) for words in domains]
    revisions = 0
    for i, words in enumerate(domains):
        domains[i] = [word for word in words if all(
            word[lc] == word[rc] for left, lc, right, rc in active if left == right == i)]
        if not domains[i]:
            return None, {"slot": i, "role": ROLES[i], "cause": "empty_initial_or_center_word_domain",
                          "removed_words": words, "revisions": revisions,
                          "overlap_equations": [row for row in active if row[0] == row[2] == i]}
    changed = True
    while changed:
        changed = False
        for i in range(len(domains)):
            for j in range(len(domains)):
                if i == j:
                    continue
                supported = [a for a in domains[i] if any(pair_valid(i, a, j, b, active) for b in domains[j])]
                revisions += 1
                if len(supported) != len(domains[i]):
                    changed = True
                    removed = [a for a in domains[i] if a not in supported]
                    domains[i] = supported
                    if not supported:
                        return None, {"slot": i, "role": ROLES[i], "unsupported_against_slot": j,
                                      "other_role": ROLES[j], "removed_words": removed,
                                      "opposite_domain": domains[j], "revisions": revisions,
                                      "overlap_equations": [row for row in active if {row[0], row[2]} == {i, j}]}
    return tuple(tuple(words) for words in domains), {"revisions": revisions}


def render(words):
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def structural_audit(words):
    content = [w for w in words if w not in FUNCTION]
    nested = []
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            if start == 0 and end == len(words):
                continue
            tape = "".join(words[start:end])
            if tape == tape[::-1]:
                nested.append([start, end])
    return {"distinct_content": len(content) == len(set(content)),
            "no_palindromic_content_word": all(w != w[::-1] for w in content),
            "no_proper_palindromic_multiword_span": not nested,
            "nested_spans": nested,
            "not_word_reflection": tuple(words) != tuple(w[::-1] for w in reversed(words))}


def solve(graph, *, state_limit=5000, equations=True, solution_limit=8):
    nodes = 0
    solutions = []
    obstruction = None

    def visit(domains, trace):
        nonlocal nodes, obstruction
        if nodes >= state_limit or len(solutions) >= solution_limit:
            return
        nodes += 1
        supported, why = propagate(graph, domains, equations=equations)
        if supported is None:
            if obstruction is None:
                obstruction = why
            return
        if all(len(words) == 1 for words in supported):
            words = tuple(domain[0] for domain in supported)
            text = render(words)
            report = audit(text)
            if equations:
                assert report["pointer_exact"]
            solutions.append({"rendered": text, "words": words, "audit": report,
                              "structural_audit": structural_audit(words), "decisions": trace})
            return
        chosen = min((i for i, words in enumerate(supported) if len(words) > 1),
                     key=lambda i: (len(supported[i]), i))
        for word in supported[chosen]:
            child = list(supported)
            child[chosen] = (word,)
            visit(tuple(child), trace + [{"slot": chosen, "word": word}])

    visit(initial_domains(graph), [])
    return {"states": nodes, "state_limit": state_limit,
            "budget_exhausted": nodes >= state_limit, "solutions": solutions,
            "first_obstruction": obstruction}


def run():
    rows = []
    for graph in frozen_geometries():
        row = {"geometry": graph.as_dict(), "initial_domain_sizes": list(map(len, initial_domains(graph))),
               "construction": solve(graph),
               "grammar_control": solve(graph, equations=False, solution_limit=1)}
        for control in row["grammar_control"]["solutions"]:
            shuffled = list(control["words"])
            random.Random(20260921 + len(rows)).shuffle(shuffled)
            control["shuffled_control"] = {"rendered": render(shuffled),
                                           "audit": audit(render(shuffled))}
        rows.append(row)
    return {"experiment_id": "boundary-geometry-crossword-20260921", "host": platform.node(),
            "method": "freeze non-reflected word boundaries, then solve connected lexical overlaps by domain propagation",
            "roles": ROLES, "bank": BANK, "cases": rows,
            "summary": {"geometries": len(rows), "exact_outputs": sum(len(r["construction"]["solutions"]) for r in rows),
                        "construction_states": sum(r["construction"]["states"] for r in rows),
                        "root_domain_contradictions": sum(r["construction"]["states"] == 1 and not r["construction"]["solutions"] for r in rows),
                        "grammatical_controls": sum(len(r["grammar_control"]["solutions"]) for r in rows)},
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "bank_sha256": sha256(json.dumps(BANK, sort_keys=True).encode()).hexdigest(),
            "target": "an independently exact intact English output at 41, 43, or 47 letters",
            "reader_gate": "No automated readability certificate. Any exact survivor needs catalogue screening and randomized blinded intact-prose/shuffled controls.",
            "construction_provenance": "fresh typed lexical banks; six geometry graphs fixed before inspecting letters; no palindrome seed or finished reversal"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))
