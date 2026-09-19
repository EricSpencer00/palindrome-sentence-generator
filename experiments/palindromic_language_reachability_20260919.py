"""Repair: exact bounded palindromic paths in ONE factored prose automaton.

This is not a new linguistic family. Unlike recent completed-clause products,
states are (matched letter depth, forward NFA node, backward NFA node).
Word/slot boundaries need not coincide, and the center can be inside a word.
One predecessor per state is sufficient for existence, not quality ranking.
No candidate scoring model, reverse realization, or catalogue is used.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ID = "palindromic-language-reachability-20260919"


def letters(text):
    return "".join(c for c in text.lower() if "a" <= c <= "z")


def audit(text):
    tape = letters(text)
    i, j = 0, len(tape) - 1
    exact = bool(tape)
    while i < j:
        if tape[i] != tape[j]:
            exact = False
        i += 1
        j -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    backward = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return dict(letters=len(tape), two_pointer_exact=exact,
                sha256_forward=forward, sha256_reverse=backward,
                sha_equal=forward == backward)


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    char: str
    surface: str


class Language:
    def __init__(self):
        self.nodes = 2
        self.start, self.end = 0, 1
        self.out = defaultdict(list)
        self.inc = defaultdict(list)
        self.edges = []

    def node(self):
        result = self.nodes
        self.nodes += 1
        return result

    def word(self, src, dst, text, space):
        # Surface fragments preserve the original, independently specified
        # spaces. Characters are labels, not reconstructed reversed words.
        fragments, pending = [], " " if space else ""
        for char in text.lower():
            if "a" <= char <= "z":
                fragments.append((char, pending + char))
                pending = ""
            else:
                pending += char
        if not fragments:
            raise ValueError("Empty lexical productions are not supported")
        for i, (char, fragment) in enumerate(fragments):
            nxt = dst if i == len(fragments) - 1 else self.node()
            edge = Edge(src, nxt, char, fragment)
            self.edges.append(edge)
            self.out[src].append(edge)
            self.inc[nxt].append(edge)
            src = nxt

    def template(self, slots):
        cursor = self.start
        for i, options in enumerate(slots):
            nxt = self.end if i == len(slots) - 1 else self.node()
            for text in options:
                self.word(cursor, nxt, text, space=i > 0)
            cursor = nxt


def solve(language, max_letters=160):
    root = (0, language.start, language.end)
    predecessor = {root: None}
    frontier = [root]
    closures, layers, dead_ends = [], [], []
    transitions = 0
    for depth in range(max_letters // 2 + 1):
        if not frontier:
            break
        layers.append(len(frontier))
        nxt = []
        for state in frontier:
            _, left, right = state
            if left == right:
                closures.append((state, None))
            if 2 * depth + 1 <= max_letters:
                closures.extend((state, e) for e in language.out[left]
                                if e.dst == right)
            left_by_char, right_by_char = defaultdict(list), defaultdict(list)
            for e in language.out[left]:
                left_by_char[e.char].append(e)
            for e in language.inc[right]:
                right_by_char[e.char].append(e)
            common = left_by_char.keys() & right_by_char.keys()
            if not common and len(dead_ends) < 12:
                dead_ends.append(dict(depth=depth, left_node=left, right_node=right,
                                     forward_next=sorted(left_by_char),
                                     backward_next=sorted(right_by_char)))
            if depth == max_letters // 2:
                continue
            for char in sorted(common):
                for le in left_by_char[char]:
                    for re in right_by_char[char]:
                        transitions += 1
                        key = (depth + 1, le.dst, re.src)
                        if key not in predecessor:
                            predecessor[key] = (state, le, re)
                            nxt.append(key)
        frontier = nxt
    rows, tapes = [], set()
    for state, middle in closures:
        left, right = [], []
        while predecessor[state] is not None:
            state, le, re = predecessor[state]
            left.append(le)
            right.append(re)
        path = list(reversed(left)) + ([middle] if middle else []) + right
        if not path:
            continue
        # Reversing a list of predecessor edges only restores forward traversal.
        assert path[0].src == language.start and path[-1].dst == language.end
        assert all(a.dst == b.src for a, b in zip(path, path[1:]))
        text = "".join(e.surface for e in path).strip().capitalize() + "."
        tape = letters(text)
        if tape in tapes:
            continue
        tapes.add(tape)
        check = audit(text)
        assert check["two_pointer_exact"] and check["sha_equal"]
        rows.append(dict(rendered=text, audit=check,
                         center_inside_lexical_path=middle is not None,
                         reader_status="unreviewed; exactness is not readability"))
    return dict(nfa_nodes=language.nodes, nfa_edges=len(language.edges),
                paired_states=len(predecessor), paired_transitions=transitions,
                layer_state_counts=layers, dead_end_certificates=dead_ends,
                max_letters=max_letters, representative_exact_candidates=rows,
                represented_closure_states=len(closures),
                longest_exact=max((r["audit"]["letters"] for r in rows), default=0),
                exhaustive_existence_within_bound=True,
                caveat="One witness per state loses alternative surfaces/quality; no quality optimum claim.")


def _render_witness(left_edges, middle, right_edges):
    """Render a packed witness whose right half is stored outer-to-inner."""
    path = list(left_edges) + ([middle] if middle else []) + list(reversed(right_edges))
    if not path:
        return None
    text = "".join(e.surface for e in path).strip().capitalize() + "."
    return text, path


def _witness_rank(witness):
    """Stable, deliberately weak ordering for bounded witness retention.

    This is only a diversity-preserving tie breaker.  It is not a readability
    certificate and is never used by the admission gate.  Prefer witnesses
    with more lexical boundaries and fewer repeated surface words so that a
    packed state does not discard the most human-looking alternatives first.
    """
    left_edges, right_edges = witness
    surfaces = [e.surface.strip() for e in left_edges + right_edges if e.surface.strip()]
    words = [w.lower() for s in surfaces for w in s.split() if w.isalpha()]
    repeated = len(words) - len(set(words))
    boundaries = sum(s.count(" ") for s in surfaces)
    return (-boundaries, repeated, " ".join(surfaces))


def solve_packed(language, max_letters=160, witnesses_per_state=32):
    """Find exact paths while retaining bounded alternatives per paired state.

    ``solve`` answers existence but keeps one arbitrary predecessor.  That is
    unsafe for a readable-language search: two derivations can reach the same
    character state while differing completely in lexical surface.  This
    variant keeps up to ``witnesses_per_state`` distinct edge witnesses per
    state, deduplicates edge identities, and emits all distinct exact tapes
    represented by the retained witnesses.  It remains a bounded diagnostic,
    not an exhaustive quality optimizer.
    """
    if witnesses_per_state < 1:
        raise ValueError("witnesses_per_state must be positive")
    root = (0, language.start, language.end)
    witnesses = {root: [((), ())]}
    visited_states = {root}
    frontier = [root]
    layers, closures, dead_ends = [], [], []
    transitions = 0
    dropped = 0
    for depth in range(max_letters // 2 + 1):
        if not frontier:
            break
        layers.append(len(frontier))
        next_witnesses = defaultdict(list)
        for state in frontier:
            _, left, right = state
            state_witnesses = witnesses[state]
            if left == right:
                closures.extend((state, None, w) for w in state_witnesses)
            if 2 * depth + 1 <= max_letters:
                for edge in language.out[left]:
                    if edge.dst == right:
                        closures.extend((state, edge, w) for w in state_witnesses)
            left_by_char, right_by_char = defaultdict(list), defaultdict(list)
            for edge in language.out[left]:
                left_by_char[edge.char].append(edge)
            for edge in language.inc[right]:
                right_by_char[edge.char].append(edge)
            common = left_by_char.keys() & right_by_char.keys()
            if not common and len(dead_ends) < 12:
                dead_ends.append(dict(depth=depth, left_node=left, right_node=right,
                                      forward_next=sorted(left_by_char),
                                      backward_next=sorted(right_by_char)))
            if depth == max_letters // 2:
                continue
            for char in sorted(common):
                for left_edge in left_by_char[char]:
                    for right_edge in right_by_char[char]:
                        transitions += len(state_witnesses)
                        key = (depth + 1, left_edge.dst, right_edge.src)
                        bucket = next_witnesses[key]
                        for left_path, right_path in state_witnesses:
                            witness = (left_path + (left_edge,), right_path + (right_edge,))
                            if witness in bucket:
                                continue
                            bucket.append(witness)
                        if len(bucket) > witnesses_per_state:
                            bucket.sort(key=_witness_rank)
                            del bucket[witnesses_per_state:]
                            dropped += 1
        frontier = sorted(next_witnesses)
        visited_states.update(frontier)
        witnesses = {state: next_witnesses[state] for state in frontier}

    rows, tapes = [], set()
    for state, middle, packed_witness in closures:
        left_path, right_path = packed_witness
        rendered = _render_witness(left_path, middle, right_path)
        if rendered is None:
            continue
        text, path = rendered
        assert path[0].src == language.start and path[-1].dst == language.end
        assert all(a.dst == b.src for a, b in zip(path, path[1:]))
        tape = letters(text)
        if tape in tapes:
            continue
        tapes.add(tape)
        check = audit(text)
        assert check["two_pointer_exact"] and check["sha_equal"]
        rows.append(dict(rendered=text, audit=check,
                         center_inside_lexical_path=middle is not None,
                         reader_status="unreviewed; exactness is not readability"))
    return dict(nfa_nodes=language.nodes, nfa_edges=len(language.edges),
                paired_states=len(visited_states), paired_transitions=transitions,
                layer_state_counts=layers, dead_end_certificates=dead_ends,
                max_letters=max_letters, representative_exact_candidates=rows,
                represented_closure_states=len(closures),
                longest_exact=max((r["audit"]["letters"] for r in rows), default=0),
                witnesses_per_state=witnesses_per_state,
                dropped_witness_buckets=dropped,
                exhaustive_existence_within_bound=False,
                caveat="Bounded packed witnesses preserve alternatives but do not certify readability or a quality optimum.")


def compile_templates(templates):
    language = Language()
    for slots in templates:
        language.template(slots)
    return language


def oracle_checks():
    cases = [
        (("a", "ab"), ("ba", "b")),
        (("ab", "ba", "a"), ("a", "b"), ("ba", "ab")),
        (("a b a", "ab ba", "ab ab", "b ab"),),
    ]
    checks = []
    for slots in cases:
        expected = {len(letters(" ".join(words))) for words in itertools.product(*slots)
                    if audit(" ".join(words))["two_pointer_exact"]}
        result = solve(compile_templates([slots]), max_letters=20)
        actual = {r["audit"]["letters"] for r in result["representative_exact_candidates"]}
        assert actual == expected, (actual, expected)
        checks.append(dict(expected_lengths=sorted(expected), actual_lengths=sorted(actual)))
    return checks


def run():
    # Exactly the 81 independently grammatical controls in the earlier PCFG
    # lane, represented as one sentence language, not paired finished clauses.
    base = [(("the baker", "a quiet sailor", "the kind nurse"),
             ("records", "carries", "opens"),
             ("the map", "a letter", "the parcel"),
             ("before dawn", "near the harbor", "beside the northern harbor"))]
    # Concrete root-obligation repair: add grammatical initial pronouns and
    # ordinary final adjuncts with compatible edge letters (i...i, we...ew,
    # the...ht). This is lexical support repair, explicitly not a new family.
    repair = base + [(("we", "you", "i"), ("see", "carry", "record"),
                      ("the map", "a letter", "the parcel"),
                      ("at night", "in the dew", "in miami")),
                     (("the baker", "the sailor", "the nurse"),
                      ("sees", "carries", "records"),
                      ("the map", "a letter", "the parcel"), ("at night",))]
    registry_path = ROOT / "docs/experiment-novelty-registry.json"
    registry = json.loads(registry_path.read_text())
    registry_ids = {r["id"] for r in registry["entries"]}
    overlaps = [x for x in ("terminal-aware-grammar-intersection-20260916",
                            "earley-finite-state-grammar-intersection-20260916",
                            "char-product-automaton-20260918") if x in registry_ids]
    results = {}
    for label, templates in (("base", base), ("endpoint_repair", repair)):
        result = solve(compile_templates(templates))
        result["factorized_sentence_count"] = sum(
            math.prod(len(slot) for slot in slots) for slots in templates)
        result["rendered_controls"] = [dict(rendered=(text := " ".join(slot[index] for slot in slots).capitalize() + "."),
                                            audit=audit(text), provenance="grammar path; noncandidate control")
                                        for slots in templates for index in (0, -1)]
        results[label] = result
    packed_results = {}
    for label, templates in (("base", base), ("endpoint_repair", repair)):
        result = solve_packed(compile_templates(templates), witnesses_per_state=64)
        result["factorized_sentence_count"] = sum(
            math.prod(len(slot) for slot in slots) for slots in templates)
        result["rendered_controls"] = [dict(rendered=(text := " ".join(slot[index] for slot in slots).capitalize() + "."),
                                            audit=audit(text), provenance="grammar path; noncandidate control")
                                       for slots in templates for index in (0, -1)]
        packed_results[label] = result
    source = Path(__file__)
    payload = dict(experiment_id=ID, results=results, packed_results=packed_results,
        oracle_checks=oracle_checks(),
        novelty=dict(disposition="implementation repair, NOT a new linguistic family",
                     prior_families=overlaps, registry_entries=len(registry_ids),
                     registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                     difference="A single sentence NFA is intersected with palindrome constraints using packed node pairs; no equal-half clause requirement or completed-clause Cartesian enumeration."),
        provenance=dict(source=str(source.relative_to(ROOT)),
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        catalogue_imported=False, per_candidate_rlaif=False,
                        completed_prose_reversed_for_realization=False),
        strict_gate=dict(admitted=0, readable_over_38=0, human_readability_test="not performed"),
        architecture_reset=dict(
            recent_lane_findings=[
                "pcfg_fsa_intersection_20260919 materializes 81 sentences and compares all 6561 clause pairs; its name does not describe an incremental PCFG intersection.",
                "morpheme_scene_composition_20260919 enumerates full scene products and adds ed/ing to already inflected verbs; role metadata is not an agreement constraint.",
                "character_paired_grammar_20260919 audits three fixed strings; online_trace records mismatches rather than constraining generation."],
            recommendation="Compile a single feature-unified sentence grammar into a shared character automaton; use paired-node co-reachability as a hard lookahead oracle for semantic derivations, with packed alternatives retained for later one-time ranking.",
            scaling="With N character nodes, depth L, maximum edge degree d: at most O(L*N^2) paired states; transitions join matching characters, avoiding complete sentence pairs. Rich recursive syntax requires bounded-stack compilation and may grow sharply.",
            next_repair="Acquire independently grammatical lexical alternatives conditioned on the emitted dead-end character sets and grammar roles. Require a longer surviving frontier before extending the grammar; expanding middle slots cannot fix outer-edge unsatisfiability.",
            limitation="The mechanism is scalable over a factored finite grammar, but these tiny banks do not establish a plausible path to long readable prose. No claimed breakthrough."))
    return payload


if __name__ == "__main__":
    result = run()
    output = ROOT / "runs" / (ID + ".json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("nfa_nodes", "paired_states", "layer_state_counts", "longest_exact")}
                      for k, v in result["results"].items()}, indent=2))
