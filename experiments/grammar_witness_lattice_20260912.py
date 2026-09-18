"""Intersect complete semantic-frame realizations with exact palindrome closure.

This experiment does not wrap a core in reflected phrases. A finite grammar
realizes one connected causal event in active/passive voice, with its reason
before/after the matrix clause. Multiword alternatives change both word count
and letter length. The compiler preserves every whole-sentence path; a paired
graph search cancels letters across arbitrary phrase and word boundaries.

Every rendered diagnostic is a complete path, including rejected branches.
An independent surface parser must recover a whole derivation before its
construction witness is valid. Neither that witness nor exactness certifies
readability. No catalogue text supplies construction material.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS = 100
MAX_LETTERS = 180
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[,.;?]")


def letters(text: str) -> str:
    return "".join(c for c in text.lower() if "a" <= c <= "z")


@dataclass(frozen=True)
class Expr:
    kind: str
    label: str
    text: str = ""
    children: tuple["Expr", ...] = ()


def lit(role: str, text: str) -> Expr:
    return Expr("literal", role, text)


def seq(label: str, *children: Expr) -> Expr:
    return Expr("sequence", label, children=children)


def alt(label: str, *children: Expr) -> Expr:
    return Expr("choice", label, children=children)


def menu(role: str, options: Iterable[str]) -> Expr:
    return alt(role, *(lit(role, option) for option in options))


@dataclass(frozen=True)
class Frame:
    identifier: str
    intent: str
    source: str
    agent: tuple[str, ...]
    active: tuple[str, ...]
    passive: tuple[str, ...]
    patient: tuple[str, ...]
    destination: tuple[str, ...]
    reason_subject: tuple[str, ...]
    reason_verb: tuple[str, ...]
    reason_object: tuple[str, ...]
    reason_context: tuple[str, ...]

    def grammar(self) -> Expr:
        """Keep agreement, argument selection, and attachment inside each route.

        The active/passive branch changes several dependencies together. It
        cannot produce an active subject with a stranded passive auxiliary.
        Every plural NP and verbal realization was authored for this frame.
        """
        subject = menu("matrix_agent", self.agent)
        patient = menu("matrix_patient", self.patient)
        destination = menu("matrix_destination", self.destination)
        active = seq("active_clause", subject, menu("matrix_finite", self.active),
                     patient, destination)
        passive_subject = menu("matrix_agent", ("us" if value == "we" else value for value in self.agent))
        passive = seq("passive_clause", patient, menu("matrix_finite", self.passive),
                      destination, lit("agent_marker", "by"), passive_subject)
        main = alt("matrix_voice", active, passive)
        reason = seq("reason_clause", menu("reason_subject", self.reason_subject),
                     menu("reason_finite", self.reason_verb),
                     menu("reason_object", self.reason_object),
                     menu("reason_context", self.reason_context))
        marker = menu("causal_marker", ("because", "since", "as"))
        return seq("sentence", alt("causal_attachment",
                   seq("matrix_first", main, marker, reason),
                   seq("reason_first", marker, reason, lit("clause_comma", ","), main)),
                   lit("terminal", "."))


# These complete source sentences and conservative realization choices were
# authored for this experiment. They are not mined from palindrome examples.
# The causal link binds the clauses to the same event; there are no detachable
# palindrome cores, names, catalogue frames, or reflected constituent menus.
FRAMES = (
    Frame("storm-relief", "Volunteers supply blankets because families face cold weather.",
          "Regional volunteers deliver warm blankets to tired evacuees because displaced families need protection from bitter weather.",
          ("regional volunteers", "local relief volunteers", "trained volunteers", "we"),
          ("deliver", "bring", "help deliver"),
          ("are delivered", "are being brought", "have been delivered"),
          ("warm blankets", "dry wool blankets", "thick clean blankets"),
          ("to tired evacuees", "to stranded evacuees", "to newly arrived evacuees"),
          ("displaced families", "homeless families", "those families"),
          ("need", "require", "desperately need"),
          ("protection", "urgent protection", "shelter"),
          ("from bitter weather", "from cold winds", "during winter storms")),
    Frame("hospital-supplies", "Nurses supply dressings because doctors need them for ward care.",
          "Experienced nurses carry clean dressings to injured workers because exhausted doctors need extra supplies near their ward.",
          ("experienced nurses", "skilled nurses", "our trained nurses", "we"),
          ("carry", "bring", "help transport"),
          ("are carried", "are being brought", "have been transported"),
          ("clean dressings", "fresh sterile dressings", "sealed medical dressings"),
          ("to injured workers", "to recovering workers", "to badly injured workers"),
          ("exhausted doctors", "busy doctors", "those doctors"),
          ("need", "require", "urgently need"),
          ("extra supplies", "additional supplies", "fresh medical supplies"),
          ("near their ward", "inside their ward", "throughout their ward")),
    Frame("museum-repairs", "Conservators restore portraits because curators need display pieces.",
          "Skilled conservators repair damaged portraits inside quiet workshops because museum curators need finished paintings for public exhibitions.",
          ("skilled conservators", "our expert conservators", "trained conservators", "we"),
          ("repair", "restore", "help restore"),
          ("are repaired", "are being restored", "have been repaired"),
          ("damaged portraits", "old damaged portraits", "faded oil portraits"),
          ("inside quiet workshops", "in nearby workshops", "within private workshops"),
          ("museum curators", "local museum curators", "those curators"),
          ("need", "require", "urgently need"),
          ("finished paintings", "restored paintings", "newly restored paintings"),
          ("for public exhibitions", "for forthcoming exhibitions", "for open exhibitions")),
    Frame("bakery-service", "Bakers prepare pastries because guests want desserts after meals.",
          "Stressed bakers prepare fresh pastries inside crowded kitchens because hungry guests request carefully prepared desserts after dinner.",
          ("stressed bakers", "experienced bakers", "our trained bakers", "we"),
          ("prepare", "bake", "help prepare"),
          ("are prepared", "are being baked", "have been prepared"),
          ("fresh pastries", "small fruit pastries", "delicate cream pastries"),
          ("inside crowded kitchens", "in busy kitchens", "within warm kitchens"),
          ("hungry guests", "newly arrived guests", "those guests"),
          ("request", "want", "eagerly request"),
          ("carefully prepared desserts", "freshly baked desserts", "small sweet desserts"),
          ("after dinner", "following dinner", "after their meals")),
)


@dataclass(frozen=True)
class Edge:
    identifier: int
    source: int
    target: int
    label: str
    text: str


class Lattice:
    """An acyclic grammar graph; epsilon edges preserve joint branch choices."""

    def __init__(self, expression: Expr):
        self.expression = expression
        self.edges: list[Edge] = []
        self.outgoing: dict[int, list[Edge]] = defaultdict(list)
        self.incoming: dict[int, list[Edge]] = defaultdict(list)
        self.nodes = 0
        self.start, self.end = self._node(), self._node()
        self._compile(expression, self.start, self.end)

    def _node(self) -> int:
        node = self.nodes
        self.nodes += 1
        return node

    def _edge(self, source: int, target: int, label: str, text: str = "") -> None:
        edge = Edge(len(self.edges), source, target, label, text)
        self.edges.append(edge)
        self.outgoing[source].append(edge)
        self.incoming[target].append(edge)

    def _compile(self, expr: Expr, source: int, target: int) -> None:
        if expr.kind == "literal":
            self._edge(source, target, expr.label, expr.text)
        elif expr.kind == "choice":
            for child in expr.children:
                entry = self._node()
                self._edge(source, entry, f"choose:{expr.label}:{child.label}")
                self._compile(child, entry, target)
        elif expr.kind == "sequence":
            if not expr.children:
                self._edge(source, target, expr.label)
            for index, child in enumerate(expr.children):
                stop = target if index == len(expr.children) - 1 else self._node()
                self._compile(child, source, stop)
                source = stop
        else:
            raise ValueError(f"unknown grammar node {expr.kind}")

    @lru_cache(maxsize=None)
    def path_bounds(self, start: int, end: int) -> tuple[int, int, int] | None:
        """Return min/max letters and number of derivations between vertices."""
        if start == end:
            return 0, 0, 1
        paths = []
        for edge in self.outgoing[start]:
            rest = self.path_bounds(edge.target, end)
            if rest is not None:
                width = len(letters(edge.text))
                paths.append((rest[0] + width, rest[1] + width, rest[2]))
        if not paths:
            return None
        return min(x[0] for x in paths), max(x[1] for x in paths), sum(x[2] for x in paths)

    def completion(self, start: int, end: int, *, epsilon_only: bool = False) -> tuple[int, ...] | None:
        if start == end:
            return ()
        for edge in self.outgoing[start]:
            if epsilon_only and letters(edge.text):
                continue
            if self.path_bounds(edge.target, end) is not None:
                rest = self.completion(edge.target, end, epsilon_only=epsilon_only)
                if rest is not None:
                    return (edge.identifier,) + rest
        return None

    @lru_cache(maxsize=None)
    def emissions(self, node: int, side: int) -> tuple[tuple[tuple[int, ...], Edge], ...]:
        answer = []
        for edge in (self.outgoing[node] if side == 1 else self.incoming[node]):
            if letters(edge.text):
                answer.append(((edge.identifier,), edge))
                continue
            adjacent = edge.target if side == 1 else edge.source
            for path, emission in self.emissions(adjacent, side):
                segment = ((edge.identifier,) + path if side == 1 else path + (edge.identifier,))
                answer.append((segment, emission))
        return tuple(answer)

    def render(self, path: tuple[int, ...]) -> str:
        text = " ".join(self.edges[index].text for index in path if self.edges[index].text)
        text = re.sub(r"\s+([,.;?])", r"\1", text)
        return text[:1].upper() + text[1:]

    def parse(self, rendered: str) -> tuple[int, ...] | None:
        """Independently parse the complete surface, without a search path/debt."""
        tokens = tuple(TOKEN.findall(rendered.lower()))
        # Reject unparsed symbols rather than silently normalizing them away.
        if re.sub(r"[a-z\s,.;?']", "", rendered.lower()):
            return None

        @lru_cache(maxsize=None)
        def visit(node: int, offset: int) -> tuple[int, ...] | None:
            if node == self.end:
                return () if offset == len(tokens) else None
            for edge in self.outgoing[node]:
                units = tuple(TOKEN.findall(edge.text.lower()))
                if tokens[offset:offset + len(units)] == units:
                    rest = visit(edge.target, offset + len(units))
                    if rest is not None:
                        return (edge.identifier,) + rest
            return None

        result = visit(self.start, 0)
        # Canonical punctuation and full token coverage are part of the witness.
        return result if result is not None and self.render(result) == rendered else None


def independent_exact_audit(text: str) -> dict:
    """Rebuild the ASCII tape and compare paired positions, not search debt."""
    if any(c.isalpha() and not c.isascii() for c in text):
        return {"exact": False, "letters": 0, "mismatches": [], "unsupported_letters": True}
    tape = "".join(re.findall("[A-Za-z]", text)).lower()
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2)
                  if tape[i] != tape[len(tape) - i - 1]]
    boundaries, position = [], 0
    for word in re.findall("[A-Za-z]+", text):
        position += len(word)
        boundaries.append(position)
    interior = set(boundaries[:-1])
    return {"exact": bool(tape) and not mismatches, "letters": len(tape),
            "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest(),
            "shifted_word_boundaries": sorted(interior - {len(tape) - x for x in interior}),
            "unsupported_letters": False}


def sentence_witness(lattice: Lattice, rendered: str, frame: Frame | None) -> dict:
    parsed = lattice.parse(rendered)
    roles: dict[str, list[str]] = defaultdict(list)
    if parsed is not None:
        for index in parsed:
            edge = lattice.edges[index]
            if edge.text:
                roles[edge.label].append(edge.text)
    required = {"matrix_agent", "matrix_finite", "matrix_patient", "matrix_destination",
                "reason_subject", "reason_finite", "reason_object", "reason_context", "causal_marker"}
    grammar_matches_frame = frame is not None and lattice.expression == frame.grammar()
    intact = grammar_matches_frame and parsed is not None and required.issubset(roles)
    return {"status": "complete_finite_grammar_derivation" if intact else "no_intact_sentence_witness",
            "intact": intact, "independent_surface_parse": parsed is not None,
            "grammar_matches_semantic_frame": grammar_matches_frame,
            "frame_id": frame.identifier if frame else None,
            "intent": frame.intent if frame else None, "roles": dict(roles),
            "parsed_path": list(parsed) if parsed is not None else None,
            "scope": "Construction syntax and semantic-role witness only; naturalness and readability are untested."}


def audit_record(lattice: Lattice, path: tuple[int, ...], frame: Frame | None,
                 reason: str | None, *, kind: str) -> dict:
    text = lattice.render(path)
    exact = independent_exact_audit(text)
    witness = sentence_witness(lattice, text, frame)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    rejections = [key for key, value in central.items() if not value]
    if not witness["intact"]:
        rejections.append("no_intact_grammatical_sentence_witness")
    if reason is not None and reason not in rejections:
        rejections.append(reason)
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact,
            "current_central_admission": central, "central_gate_passed": all(central.values()),
            "sentence_witness": witness, "intact_grammatical_sentence_witness": witness["intact"],
            "derivation_path": list(path), "hard_rejection": reason,
            "rejection_codes": rejections, "mechanically_admitted": not rejections,
            "reader_status": "unreviewed; no blinded human readability evidence"}


def solve(lattice: Lattice, *, frame: Frame | None = None, min_letters: int = MIN_LETTERS,
          max_letters: int = MAX_LETTERS, max_states: int = 20000,
          record_rejections: bool = True) -> dict:
    """Exactly intersect the bounded grammar language without fixing a tape.

    Debt is an unmatched letter suffix read from the outside inward. A longer
    phrase on one side can cancel several phrases on the other. Graph
    reachability prevents mixing incompatible voice/attachment branches. The
    unconsumed middle may close inside a lexical item, with no centre boundary.
    """
    if min_letters < 0 or max_letters < min_letters or max_states < 1:
        raise ValueError("invalid length or state bounds")
    stats = Counter(states=0, emissions_tried=0, exact_closures=0,
                    incompatible_grammar_branches=0, cross_phrase_cancellations=0)
    rejections: list[dict] = []
    solutions: list[tuple[int, ...]] = []
    traces: dict[tuple[int, ...], list[dict]] = {}
    exhausted = True

    def reject(reason: str, left: tuple[int, ...], right: tuple[int, ...],
               lo: int, hi: int, detail: dict) -> None:
        stats[reason] += 1
        if not record_rejections:
            return
        middle = lattice.completion(lo, hi)
        if middle is None:
            raise AssertionError("rejection must have a complete grammatical continuation")
        row = audit_record(lattice, left + middle + right, frame, reason,
                           kind="complete_sentence_representative_of_rejected_branch")
        row["constraint_failure"] = detail
        row["unexpanded_middle_derivations"] = lattice.path_bounds(lo, hi)[2]
        rejections.append(row)

    def visit(lo: int, hi: int, left: tuple[int, ...], right: tuple[int, ...],
              debt: str, owner: int, used: frozenset[str], length: int,
              trace: list[dict]) -> None:
        nonlocal exhausted
        if stats["states"] >= max_states:
            exhausted = False
            return
        stats["states"] += 1
        bounds = lattice.path_bounds(lo, hi)
        if bounds is None:
            stats["incompatible_grammar_branches"] += 1
            return
        if length + bounds[0] > max_letters or length + bounds[1] < min_letters:
            reject("length_unreachable", left, right, lo, hi,
                   {"selected_letters": length, "remaining_min_max": bounds[:2]})
            return
        centre = lattice.completion(lo, hi, epsilon_only=True)
        if centre is not None:
            if length and debt == debt[::-1] and min_letters <= length <= max_letters:
                path = left + centre + right
                solutions.append(path)
                traces[path] = trace
                stats["exact_closures"] += 1
            else:
                reject("central_debt_not_palindromic", left, right, lo, hi, {"debt": debt})
            # An epsilon middle need not be the only grammar alternative.
        side = -owner if debt else 1
        for segment, edge in lattice.emissions(lo if side == 1 else hi, side):
            if not exhausted:
                break
            next_lo, next_hi = (edge.target, hi) if side == 1 else (lo, edge.source)
            if lattice.path_bounds(next_lo, next_hi) is None:
                stats["incompatible_grammar_branches"] += 1
                continue
            stats["emissions_tried"] += 1
            next_left, next_right = (left + segment, right) if side == 1 else (left, segment + right)
            tokens = tuple(re.findall(r"[a-z]+(?:'[a-z]+)?", edge.text.lower()))
            incoming = letters(edge.text)[::side]
            shared = min(len(debt), len(incoming))
            if len(tokens) != len(set(tokens)) or used.intersection(tokens):
                reason = "repeated_word_in_derivation"
            elif any(letters(word) == letters(word)[::-1] for word in tokens):
                reason = "self_palindromic_word_in_derivation"
            elif debt[:shared] != incoming[:shared]:
                reason = "reflected_prefix_contradiction"
            else:
                reason = None
            detail = {"side": "left" if side == 1 else "right", "phrase": edge.text,
                      "role": edge.label, "debt_before": debt, "incoming": incoming,
                      "compared_letters": shared}
            if reason:
                reject(reason, next_left, next_right, next_lo, next_hi, detail)
                continue
            new_debt, new_owner = ((debt[shared:], owner) if len(debt) > len(incoming)
                                   else (incoming[shared:], side))
            if debt and len(debt) != len(incoming):
                stats["cross_phrase_cancellations"] += 1
            detail.update(debt_after=new_debt, selected_letters=length + len(incoming))
            visit(next_lo, next_hi, next_left, next_right, new_debt, new_owner,
                  used.union(tokens), length + len(incoming), trace + [detail])

    visit(lattice.start, lattice.end, (), (), "", 1, frozenset(), 0, [])
    unique = {lattice.render(path): path for path in solutions}
    closures = []
    for rendered, path in sorted(unique.items()):
        row = audit_record(lattice, path, frame, None, kind="exact_complete_derivation")
        if not row["independent_exact_audit"]["exact"]:
            raise AssertionError("independent audit contradicted exact search")
        row["cancellation_trace"] = traces[path]
        closures.append(row)
    return {"exhausted_frozen_grammar": exhausted, "stats": dict(stats),
            "grammar_derivations": lattice.path_bounds(lattice.start, lattice.end)[2],
            "solutions": sorted(unique), "exact_closures": closures,
            "hard_rejections": rejections}


def run(*, max_states: int = 20000) -> dict:
    runs = []
    for frame in FRAMES:
        grammar = frame.grammar()
        lattice = Lattice(grammar)
        source_path = lattice.parse(frame.source)
        if source_path is None:
            raise AssertionError(f"source sentence has no complete derivation: {frame.identifier}")
        result = solve(lattice, frame=frame, max_states=max_states)
        runs.append({"frame": asdict(frame), "grammar_sha256": digest(asdict(grammar)),
                     "grammar_nodes": lattice.nodes, "grammar_edges": len(lattice.edges),
                     "source_control": audit_record(lattice, source_path, frame, None,
                                                    kind="authored_intact_nonpalindromic_control"),
                     **result})
    exact = [row for result in runs for row in result["exact_closures"]]
    return {"status": "complete_bounded_semantic_grammar_lattice_intersection",
            "config": {"max_states_per_frame": max_states, "min_letters": MIN_LETTERS,
                       "max_letters": MAX_LETTERS, "exactness_enforced_during_search": True,
                       "complete_grammar_path_required_during_search": True,
                       "fixed_tape": False, "fixed_word_count": False, "catalogue_used_for_generation": False},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "frames_sha256": digest([asdict(frame) for frame in FRAMES]),
                           "admission_sha256": sha256((ROOT / "llm_palindrome/admission.py").read_bytes()).hexdigest(),
                           "exclusion_catalogue_sha256": sha256((ROOT / "data/known_palindromes.json").read_bytes()).hexdigest(),
                           "material": "Four task-authored causal event frames and complete source sentences; no borrowed palindrome, proper-name substitution, reflected phrase inventory, or language score."},
            "runs": runs, "exact_closures": exact,
            "mechanically_admitted": [row for row in exact if row["mechanically_admitted"]],
            "readable_survivors": [],
            "scope": "Exhaustion applies only to these finite semantic-frame realization languages. A whole grammar derivation is a construction witness, never a human readability certificate. No global originality or readability claim is made."}


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=20000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "frames": len(result["runs"]),
                      "states": sum(row["stats"]["states"] for row in result["runs"]),
                      "hard_rejections": sum(len(row["hard_rejections"]) for row in result["runs"]),
                      "exact_closures": len(result["exact_closures"]),
                      "mechanically_admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
