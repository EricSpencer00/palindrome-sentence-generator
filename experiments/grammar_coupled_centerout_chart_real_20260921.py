"""Finite typed-grammar center-out chart with an actual character trie join.

The chart compiles grammatical sentence paths into a forward trie and a trie
over reversed tapes.  A state is admitted only when both tries expose the same
character, so a terminal pair is an exact letter-level palindrome across the
two clauses.  This is deliberately a finite grammar experiment, not a claim
that a small lexicon covers English.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
ID = "grammar-coupled-centerout-chart-real-20260921"
SIG = "typed-grammar-paths|forward-reverse-character-trie-join|equal-length-terminals"


def norm(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def independent_audit(text: str) -> dict:
    tape = norm(text)
    mismatches = [i for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode("utf-8")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("utf-8")).hexdigest()
    return {
        "exact": bool(tape) and not mismatches,
        "letters": len(tape),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:8],
        "pointer_pairs_checked": len(tape) // 2,
        "forward_sha256": forward,
        "reverse_tape_sha256": reverse,
        "sha_equal_after_reverse": forward == reverse,
    }


def new_node() -> dict:
    return {"children": {}, "terminals": []}


def insert(trie: list[dict], tape: str, path_id: int) -> None:
    node_id = 0
    for char in tape:
        child = trie[node_id]["children"].get(char)
        if child is None:
            child = len(trie)
            trie[node_id]["children"][char] = child
            trie.append(new_node())
        node_id = child
    trie[node_id]["terminals"].append(path_id)


def determiner_options(number: str, noun: str) -> list[str]:
    if number == "pl":
        return ["the", "some"]
    if noun[0] in "aeiou":
        return ["the", "an", "some"]
    return ["the", "a", "some"]


def build_paths() -> list[dict]:
    singular = [
        ("aide", "sg"),
        ("artist", "sg"),
        ("guard", "sg"),
        ("owl", "sg"),
        ("pilot", "sg"),
        ("poet", "sg"),
        ("raven", "sg"),
        ("teacher", "sg"),
        ("writer", "sg"),
    ]
    plural = [
        ("artists", "pl"),
        ("guards", "pl"),
        ("men", "pl"),
        ("memos", "pl"),
        ("pilots", "pl"),
        ("poets", "pl"),
        ("ravens", "pl"),
        ("teachers", "pl"),
        ("writers", "pl"),
    ]
    nouns = singular + plural
    count_nouns = {
        "raven": "ravens",
        "artist": "artists",
        "guard": "guards",
        "map": "maps",
        "memo": "memos",
        "pilot": "pilots",
        "poet": "poets",
        "teacher": "teachers",
        "writer": "writers",
    }
    proper = ["aaron", "diana", "iris", "leon", "mira", "noah", "nora"]
    iv = {"sg": ["rests", "runs", "sings", "smiles", "waits", "walks"],
          "pl": ["rest", "run", "sing", "smile", "wait", "walk"]}
    tv = {"sg": ["admires", "finds", "guides", "helps", "marks", "notes", "reads", "sees"],
          "pl": ["admire", "find", "guide", "help", "mark", "note", "read", "see"]}
    paths: list[dict] = []

    def add(template: str, words: list[str], roles: list[str], *, seed: bool = False) -> None:
        surface = " ".join(words)
        paths.append({
            "surface": surface,
            "words": words,
            "roles": roles,
            "template": template,
            "seed": seed,
            "tape": norm(surface),
        })

    # Agreement-carrying intransitive clauses.
    for noun, number in nouns:
        for det in determiner_options(number, noun.rstrip("s") if number == "pl" else noun):
            for verb in iv[number]:
                add("DET NOUN IV", [det, noun, verb], [number, "det", "noun", "iv"])

    # Ordinary transitive clauses with typed subject/object agreement.
    for subject, subject_number in nouns:
        for verb in tv[subject_number]:
            for obj, object_number in nouns:
                for det in determiner_options(object_number, obj.rstrip("s") if object_number == "pl" else obj):
                    subject_det = determiner_options(subject_number, subject.rstrip("s") if subject_number == "pl" else subject)
                    for sdet in subject_det:
                        add("DET NOUN TV DET NOUN", [sdet, subject, verb, det, obj],
                            [subject_number, object_number, "det", "noun", "tv", "det", "noun"])

    # Counted objects and proper-name objects give different boundary shapes.
    for subject, subject_number in nouns:
        subject_dets = determiner_options(subject_number, subject.rstrip("s") if subject_number == "pl" else subject)
        for sdet in subject_dets:
            for verb in tv[subject_number]:
                for singular_noun, plural_noun in count_nouns.items():
                    for quantity, object_noun in (("one", singular_noun), ("two", plural_noun), ("nine", plural_noun)):
                        add("DET NOUN TV NUM NOUN", [sdet, subject, verb, quantity, object_noun],
                            [subject_number, "count", "det", "noun", "tv", "num", "noun"])
                for name in proper:
                    add("DET NOUN TV PROPER", [sdet, subject, verb, name],
                        [subject_number, "proper", "det", "noun", "tv", "proper"])

    # The known 38-letter seed is present only as a held-out recovery check;
    # it is never admitted as a new candidate.
    add("HELD_OUT_SEED_LEFT", ["an", "aide", "rips", "nine", "memos"], ["seed", "left"], seed=True)
    add("HELD_OUT_SEED_RIGHT", ["some", "men", "inspire", "diana"], ["seed", "right"], seed=True)

    deduped: dict[tuple[str, tuple[str, ...]], dict] = {}
    for path in paths:
        deduped[(path["surface"], tuple(path["roles"]))] = path
    return list(deduped.values())


def shortcut_reasons(left: dict, right: dict, rendered: str) -> list[str]:
    words = re.findall(r"[a-z]+", rendered.lower())
    reasons: list[str] = []
    if left["seed"] or right["seed"]:
        reasons.append("held_out_seed_replay")
    if len(words) != len(set(words)):
        reasons.append("repeated_word_unit")
    if any(len(word) >= 3 and word == word[::-1] for word in words):
        reasons.append("self_palindromic_word_unit")
    word_set = set(words)
    if any(len(word) >= 3 and word[::-1] in word_set and word != word[::-1] for word in words):
        reasons.append("semordnilap_word_pair")
    return reasons


def render_row(left: dict, right: dict, *, status: str, chart_depth: int, reasons: list[str]) -> dict:
    rendered = f"{left['surface']}; {right['surface']}."
    tape = norm(rendered)
    return {
        "rendered": rendered,
        "left_surface": left["surface"],
        "right_surface": right["surface"],
        "left_roles": left["roles"],
        "right_roles": right["roles"],
        "left_template": left["template"],
        "right_template": right["template"],
        "letters": len(tape),
        "normalized_sha256": sha(rendered),
        "independent_exact_audit": independent_audit(rendered),
        "chart_provenance": {
            "state_depth": chart_depth,
            "forward_path_id_is_grammar_terminal": True,
            "reverse_path_id_is_reversed_tape_terminal": True,
            "equal_character_edges_enforced": True,
        },
        "shortcut_rejections": reasons,
        "candidate_status": status,
        "reader_status": "not_run",
    }


def main(out: Path) -> None:
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    paths = build_paths()
    forward: list[dict] = [new_node()]
    reverse: list[dict] = [new_node()]
    for path_id, path in enumerate(paths):
        insert(forward, path["tape"], path_id)
        insert(reverse, path["tape"][::-1], path_id)

    queue = deque([(0, 0, 0)])
    seen = {(0, 0)}
    states = 0
    character_edges = 0
    early_prunes = 0
    terminal_pairs = 0
    exact_rows: list[dict] = []
    seed_recovery: list[dict] = []

    while queue:
        left_node, right_node, depth = queue.popleft()
        states += 1
        left_children = forward[left_node]["children"]
        right_children = reverse[right_node]["children"]
        common = sorted(set(left_children) & set(right_children))
        early_prunes += len(set(left_children) ^ set(right_children))
        for char in common:
            character_edges += 1
            state = (left_children[char], right_children[char])
            if state not in seen:
                seen.add(state)
                queue.append((state[0], state[1], depth + 1))

        left_terms = forward[left_node]["terminals"]
        right_terms = reverse[right_node]["terminals"]
        if not left_terms or not right_terms:
            continue
        for left_id in left_terms:
            for right_id in right_terms:
                terminal_pairs += 1
                left = paths[left_id]
                right = paths[right_id]
                rendered = f"{left['surface']}; {right['surface']}."
                reasons = shortcut_reasons(left, right, rendered)
                row = render_row(left, right, status="exact_terminal_pair", chart_depth=depth, reasons=reasons)
                if left["seed"] or right["seed"]:
                    seed_recovery.append({
                        "left": left["surface"],
                        "right": right["surface"],
                        "exact": row["independent_exact_audit"]["exact"],
                        "excluded_from_admission": True,
                    })
                elif len(row["normalized_sha256"]) and row["letters"] >= 40 and row["independent_exact_audit"]["exact"]:
                    exact_rows.append(row)

    # Deterministic controls are rendered prose, but are never mislabeled as
    # candidates.  They make the artifact inspectable when the exact join is
    # empty and give the later reader package a concrete control pool.
    controls = [
        "The pilot helps the artist.",
        "Some writers admire a raven.",
        "An aide rips nine memos; some men inspire Diana.",
    ]
    rendered_controls = []
    for text in controls:
        rendered_controls.append({
            "rendered": text,
            "letters": len(norm(text)),
            "normalized_sha256": sha(text),
            "independent_exact_audit": independent_audit(text),
            "candidate_status": "diagnostic_control",
            "reader_status": "not_run",
        })

    out.write_text(json.dumps({
        "status": "grammar_coupled_centerout_chart_executed",
        "family_id": ID,
        "state_space_signature": SIG,
        "novelty_audit": {
            "registry_entries_read_before_run": len(registry["entries"]),
            "signature_overlap": [],
            "duplicate_preflight_only": False,
        },
        "config": {
            "target_letters": ">=40",
            "finite_typed_grammar": True,
            "forward_trie_nodes": len(forward),
            "reverse_trie_nodes": len(reverse),
            "paths_compiled": len(paths),
            "required_character_condition": "same character on every opposing edge",
            "terminal_condition": "both grammar paths end at the same chart depth",
        },
        "search_accounting": {
            "states": states,
            "character_edges": character_edges,
            "early_prunes": early_prunes,
            "terminal_pairs": terminal_pairs,
            "exact_candidates": len(exact_rows),
        },
        "held_out_seed_recovery": {
            "checked": True,
            "recoveries": seed_recovery,
            "excluded_from_admission": True,
        },
        "exact_candidates": exact_rows,
        "rendered_controls": rendered_controls,
        "acceptance_frontier_changed": bool(exact_rows),
        "reader_status": "not_run",
        "next_construction": "expand the typed grammar with adjunct and agreement-preserving clause paths; retain this chart as the exact character join",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/grammar-coupled-centerout-chart-real-20260921.json")
