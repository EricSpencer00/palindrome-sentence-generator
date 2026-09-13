"""Candidate-first assisted construction pilot for exact English palindromes.

This is deliberately an *assistant*, not a language-model wrapper and not a
finite sentence-template generator.  A human or a model may submit bounded,
coordinated edits to the left and right textual regions.  The pilot preserves
the raw proposal, checks its character consequences, retains a compact chart
of every dictionary word-boundary/prefix analysis, and only exposes a surface
when exactness and the shared mechanical gate both clear.

No client, model invocation, corpus mining, or readability certification is
implemented here.  A source label of ``model`` is provenance supplied by the
caller; it never causes a request to a model.  Syntax and lexical diagnostics
help an author decide what to repair, but they never promote a candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.lexicon import load_lexicon
from llm_palindrome.safe_vocab import is_allowed


DATE = "20260913"
MIN_CANDIDATE_LETTERS = 100
MAX_CANDIDATE_LETTERS = 240
MAX_PROPOSAL_LETTERS = 48
DEFAULT_STATE_BUDGET = 100_000
WORD_RE = re.compile(r"[a-z]+")

# These short closed-class forms are intentionally installed even if a corpus
# tags them inconsistently.  They are ordinary English, not a six-character
# lexical proxy.  The much larger content inventory below is selected from the
# shipped lexicon independently of any proposal, fringe, or reverse match.
ORDINARY_FUNCTION_WORDS = frozenset(
    "a an the this that these those my your our their i you he she it we they "
    "me him her us them who which whose and or but because while if when as "
    "of to in on at by for from with without near after before is are was were "
    "be been being do does did have has had can could will would may might should "
    "not no yes".split()
)


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _text_digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _safe_normalize(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("proposal text must be a string")
    tape = normalize_letters(value)
    if not tape:
        return ""
    if any(ch not in "abcdefghijklmnopqrstuvwxyz" for ch in tape):
        raise ValueError("proposal uses unsupported letters")
    return tape


@lru_cache(maxsize=1)
def lexical_evidence() -> dict[str, Any]:
    """Freeze broad, independently sourced L1 evidence before proposals exist."""
    base = sorted(load_lexicon(str(ROOT / "data" / "lexicon.txt")))
    # All ordinary safe dictionary forms are retained.  The narrow short-word
    # policy removes dictionary labels such as ``b`` and ``st`` while retaining
    # ordinary function words such as ``a``, ``an``, ``in``, and ``to``.  This
    # is deliberately not a six-letter content filter.
    safe = tuple(word for word in base if word.isascii() and word.isalpha() and is_allowed(word)
                 and (len(word) >= 3 or word in ORDINARY_FUNCTION_WORDS))
    tags: dict[str, set[str]] = defaultdict(set)
    evidence: dict[str, Any] = {
        "lexicon_sha256": _text_digest("\n".join(base)),
        "selection_timing": "before proposal ingestion, fringe construction, or reverse-character comparison",
        "content_policy": "safe shipped dictionary forms of length >=3 plus explicit ordinary function words; no minimum-six-letter filter",
        "wordfreq": "not used for candidate selection",
    }
    try:
        from nltk.corpus import brown
        for word, tag in brown.tagged_words(tagset="universal"):
            word = word.casefold()
            if word.isascii() and word.isalpha():
                tags[word].add(tag)
        evidence["brown_universal_pos"] = "available"
    except Exception as error:  # Diagnostics must disclose unavailable data.
        evidence["brown_universal_pos"] = "unavailable"
        evidence["brown_error"] = type(error).__name__
    content = frozenset(safe)
    inventory = frozenset(content | ORDINARY_FUNCTION_WORDS)
    evidence["safe_dictionary_count"] = len(content)
    evidence["ordinary_function_word_count"] = len(ORDINARY_FUNCTION_WORDS)
    evidence["inventory_sha256"] = _text_digest("\n".join(sorted(inventory)))
    evidence["brown_pos_counts"] = {
        tag: sum(1 for word in inventory if tag in tags.get(word, set()))
        for tag in ("NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON", "CONJ")
    }
    return {"words": inventory, "tags": {word: frozenset(value) for word, value in tags.items()}, "evidence": evidence}


class Trie:
    """A word trie used only to retain lexical analyses, never to certify prose."""

    def __init__(self, words: Iterable[str]):
        self.children: list[dict[str, int]] = [{}]
        self.terminal: list[str | None] = [None]
        for word in sorted(set(words)):
            cursor = 0
            for char in word:
                child = self.children[cursor].get(char)
                if child is None:
                    child = len(self.children)
                    self.children[cursor][char] = child
                    self.children.append({})
                    self.terminal.append(None)
                cursor = child
            self.terminal[cursor] = word

    def words_from(self, tape: str, start: int) -> Iterator[tuple[int, str]]:
        cursor = 0
        for end in range(start, len(tape)):
            child = self.children[cursor].get(tape[end])
            if child is None:
                return
            cursor = child
            word = self.terminal[cursor]
            if word is not None:
                yield end + 1, word

    def prefix_at_end(self, tape: str, start: int) -> str | None:
        cursor = 0
        for char in tape[start:]:
            child = self.children[cursor].get(char)
            if child is None:
                return None
            cursor = child
        return tape[start:]


@lru_cache(maxsize=1)
def word_trie() -> Trie:
    return Trie(lexical_evidence()["words"])


@dataclass(frozen=True)
class SyntaxObligation:
    """A permissive, non-authoritative grammar-state diagnostic."""

    phase: str = "subject_start"
    clauses: int = 0


def _categories(word: str) -> frozenset[str]:
    if word in ORDINARY_FUNCTION_WORDS:
        fixed = {
            "a": {"DET"}, "an": {"DET"}, "the": {"DET"}, "this": {"DET"}, "that": {"DET", "REL"},
            "these": {"DET"}, "those": {"DET"}, "my": {"DET"}, "your": {"DET"}, "our": {"DET"}, "their": {"DET"},
            "who": {"REL", "PRON"}, "which": {"REL", "PRON"}, "whose": {"REL", "DET"},
            "and": {"CONJ"}, "or": {"CONJ"}, "but": {"CONJ"}, "because": {"SUB"}, "while": {"SUB"},
            "if": {"SUB"}, "when": {"SUB"}, "as": {"SUB"},
            "of": {"ADP"}, "to": {"ADP"}, "in": {"ADP"}, "on": {"ADP"}, "at": {"ADP"}, "by": {"ADP"},
            "for": {"ADP"}, "from": {"ADP"}, "with": {"ADP"}, "without": {"ADP"}, "near": {"ADP"},
            "after": {"ADP"}, "before": {"ADP"},
        }
        if word in fixed:
            return frozenset(fixed[word])
    tags = set(lexical_evidence()["tags"].get(word, ()))
    mapped = {tag for tag in tags if tag in {"NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON", "CONJ"}}
    return frozenset(mapped or {"LEX"})


def syntax_step(state: SyntaxObligation, word: str) -> frozenset[SyntaxObligation]:
    """Keep possible obligations; this is evidence for repair, never a filter."""
    categories = _categories(word)
    out: set[SyntaxObligation] = set()
    if state.phase == "subject_start":
        if categories & {"DET", "PRON"}:
            out.add(SyntaxObligation("subject_head", state.clauses))
        if categories & {"NOUN"}:
            out.add(SyntaxObligation("predicate", state.clauses))
    elif state.phase == "subject_head":
        if "ADJ" in categories:
            out.add(state)
        if categories & {"NOUN", "PRON"}:
            out.add(SyntaxObligation("predicate", state.clauses))
    elif state.phase == "predicate":
        if "VERB" in categories:
            out.add(SyntaxObligation("object_start", state.clauses))
        if "REL" in categories:
            out.add(state)
    elif state.phase == "object_start":
        if categories & {"DET", "PRON"}:
            out.add(SyntaxObligation("object_head", state.clauses))
        if "ADJ" in categories:
            out.add(SyntaxObligation("object_head", state.clauses))
        if "NOUN" in categories:
            out.add(SyntaxObligation("tail", state.clauses))
    elif state.phase == "object_head":
        if "ADJ" in categories:
            out.add(state)
        if categories & {"NOUN", "PRON"}:
            out.add(SyntaxObligation("tail", state.clauses))
    elif state.phase == "tail":
        if "ADP" in categories:
            out.add(SyntaxObligation("object_start", state.clauses))
        if "CONJ" in categories:
            out.add(SyntaxObligation("predicate", state.clauses))
        if "SUB" in categories and state.clauses < 2:
            out.add(SyntaxObligation("subject_start", state.clauses + 1))
    return frozenset(out)


@dataclass(frozen=True)
class LexicalAnalysis:
    tape: str
    complete_segmentations: int
    boundary_positions: tuple[int, ...]
    open_prefixes: tuple[tuple[int, str], ...]
    syntax_obligations: tuple[SyntaxObligation, ...]


def analyze_tape(tape: str) -> LexicalAnalysis:
    """Compactly preserve every word-boundary and unfinished-prefix analysis.

    ``complete_segmentations`` is a dynamic-programming count, not a sampled
    list.  The chart never chooses a preferred tokenization.  Syntax states
    are a second, deliberately permissive diagnostic layer over the lexical
    chart, so lexical alternatives are never discarded when syntax is weak.
    """
    trie = word_trie()
    reachable = [0] * (len(tape) + 1)
    reachable[0] = 1
    boundaries: set[int] = {0}
    syntax: dict[int, set[SyntaxObligation]] = defaultdict(set)
    syntax[0].add(SyntaxObligation())
    for start in range(len(tape)):
        if not reachable[start]:
            continue
        for end, word in trie.words_from(tape, start):
            reachable[end] += reachable[start]
            boundaries.add(end)
            for obligation in syntax.get(start, set()):
                syntax[end].update(syntax_step(obligation, word))
    open_prefixes = []
    for start in sorted(boundaries):
        prefix = trie.prefix_at_end(tape, start)
        if prefix:
            open_prefixes.append((start, prefix))
    return LexicalAnalysis(
        tape=tape,
        complete_segmentations=reachable[-1],
        boundary_positions=tuple(sorted(boundaries)),
        open_prefixes=tuple(open_prefixes),
        syntax_obligations=tuple(sorted(syntax.get(len(tape), set()), key=repr)),
    )


def enumerate_segmentations(tape: str) -> tuple[tuple[str, ...], ...]:
    """Materialize every lexical rendering for a closure; never sample one."""
    trie = word_trie()

    @lru_cache(maxsize=None)
    def visit(start: int) -> tuple[tuple[str, ...], ...]:
        if start == len(tape):
            return ((),)
        rows = []
        for end, word in trie.words_from(tape, start):
            rows.extend((word,) + tail for tail in visit(end))
        return tuple(rows)

    return visit(0)


@dataclass(frozen=True)
class Region:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ConstructionState:
    state_id: str
    parent_id: str | None
    left_tape: str
    right_tape: str
    left_analysis: LexicalAnalysis
    right_analysis: LexicalAnalysis
    debt_side: str | None
    symmetric_debt: str
    proposal_history: tuple[str, ...] = ()
    edit_history: tuple[dict[str, Any], ...] = ()

    @property
    def paired_tape(self) -> str:
        return self.right_tape[::-1]


def symmetric_debt(left_tape: str, right_tape: str) -> tuple[str | None, str]:
    """Return the still-unpaired inward characters, or reject a true conflict."""
    paired_right = right_tape[::-1]
    shared = min(len(left_tape), len(paired_right))
    if left_tape[:shared] != paired_right[:shared]:
        raise ValueError("character_conflict_between_independently_proposed_regions")
    if len(left_tape) > shared:
        return "left", left_tape[shared:]
    if len(paired_right) > shared:
        return "right", paired_right[shared:]
    return None, ""


def state_from_tapes(*, parent_id: str | None, left_tape: str, right_tape: str,
                     proposal_history: tuple[str, ...] = (), edit_history: tuple[dict[str, Any], ...] = ()) -> ConstructionState:
    debt_side, debt = symmetric_debt(left_tape, right_tape)
    identity = _digest({"parent": parent_id, "left": left_tape, "right": right_tape,
                        "history": proposal_history, "edits": edit_history})
    return ConstructionState(
        state_id=identity,
        parent_id=parent_id,
        left_tape=left_tape,
        right_tape=right_tape,
        left_analysis=analyze_tape(left_tape),
        right_analysis=analyze_tape(right_tape),
        debt_side=debt_side,
        symmetric_debt=debt,
        proposal_history=proposal_history,
        edit_history=edit_history,
    )


def root_state() -> ConstructionState:
    return state_from_tapes(parent_id=None, left_tape="", right_tape="")


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    parent_state_id: str
    source: dict[str, str]
    operation: str
    left_text: str = ""
    right_text: str = ""
    center_text: str = ""
    left_region: Region | None = None
    right_region: Region | None = None
    notes: str = ""

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Proposal":
        def region(key: str) -> Region | None:
            value = raw.get(key)
            if value is None:
                return None
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            return Region(int(value["start"]), int(value["end"]), str(value["text"]))
        source = raw.get("source", {})
        if not isinstance(source, dict) or not isinstance(source.get("kind"), str):
            raise ValueError("source.kind is required provenance")
        return cls(
            proposal_id=str(raw["proposal_id"]), parent_state_id=str(raw["parent_state_id"]), source={str(k): str(v) for k, v in source.items()},
            operation=str(raw["operation"]), left_text=str(raw.get("left_text", "")), right_text=str(raw.get("right_text", "")),
            center_text=str(raw.get("center_text", "")), left_region=region("left_region"), right_region=region("right_region"),
            notes=str(raw.get("notes", "")),
        )


def _bounded(tape: str, field_name: str) -> None:
    if len(tape) > MAX_PROPOSAL_LETTERS:
        raise ValueError(f"{field_name}_exceeds_{MAX_PROPOSAL_LETTERS}_letter_bound")


def apply_proposal(state: ConstructionState, proposal: Proposal) -> tuple[ConstructionState | None, dict[str, Any]]:
    """Replay a proposal exactly; a rejected proposal never mutates its parent."""
    event = {"proposal_id": proposal.proposal_id, "parent_state_id": state.state_id, "source": proposal.source,
             "operation": proposal.operation, "input": asdict(proposal), "accepted": False}
    try:
        if proposal.parent_state_id != state.state_id:
            raise ValueError("parent_state_mismatch")
        if proposal.operation == "continue":
            left = _safe_normalize(proposal.left_text)
            right = _safe_normalize(proposal.right_text)
            _bounded(left, "left_continuation")
            _bounded(right, "right_continuation")
            if not left or not right:
                raise ValueError("coordinated_continuation_requires_nonempty_both_sides")
            # Right text is supplied in its natural reading direction and is
            # prepended at the open middle.  It is never synthesized by
            # reversing the left proposal.
            next_left = state.left_tape + left
            next_right = right + state.right_tape
            edit = {"kind": "continue", "left_added": left, "right_added": right,
                    "both_sides_supplied": True}
        elif proposal.operation == "reopen":
            if proposal.left_region is None or proposal.right_region is None:
                raise ValueError("coordinated_reopen_requires_both_regions")
            lreg, rreg = proposal.left_region, proposal.right_region
            if not (0 <= lreg.start <= lreg.end <= len(state.left_tape)):
                raise ValueError("left_region_out_of_bounds")
            if not (0 <= rreg.start <= rreg.end <= len(state.right_tape)):
                raise ValueError("right_region_out_of_bounds")
            left = _safe_normalize(lreg.text)
            right = _safe_normalize(rreg.text)
            _bounded(left, "left_replacement")
            _bounded(right, "right_replacement")
            next_left = state.left_tape[:lreg.start] + left + state.left_tape[lreg.end:]
            next_right = state.right_tape[:rreg.start] + right + state.right_tape[rreg.end:]
            edit = {"kind": "reopen", "left_region": asdict(lreg), "right_region": asdict(rreg),
                    "normalized_left": left, "normalized_right": right, "both_sides_supplied": True}
        else:
            raise ValueError("unsupported_operation")
        child = state_from_tapes(parent_id=state.state_id, left_tape=next_left, right_tape=next_right,
                                 proposal_history=state.proposal_history + (proposal.proposal_id,),
                                 edit_history=state.edit_history + (edit,))
        event.update({"accepted": True, "child_state_id": child.state_id,
                      "left_letters": len(child.left_tape), "right_letters": len(child.right_tape),
                      "debt": {"side": child.debt_side, "letters": child.symmetric_debt},
                      "analysis": state_diagnostic(child)})
        return child, event
    except (TypeError, ValueError) as error:
        event["rejection"] = str(error)
        return None, event


def state_diagnostic(state: ConstructionState) -> dict[str, Any]:
    """Surface diagnostics.  None of these fields decide candidate admission."""
    return {
        "left": {"letters": len(state.left_tape), "complete_segmentations": state.left_analysis.complete_segmentations,
                 "boundary_positions": state.left_analysis.boundary_positions, "open_prefixes": state.left_analysis.open_prefixes,
                 "syntax_obligations": [asdict(row) for row in state.left_analysis.syntax_obligations]},
        "right": {"letters": len(state.right_tape), "complete_segmentations": state.right_analysis.complete_segmentations,
                  "boundary_positions": state.right_analysis.boundary_positions, "open_prefixes": state.right_analysis.open_prefixes,
                  "syntax_obligations": [asdict(row) for row in state.right_analysis.syntax_obligations]},
        "symmetric_debt": {"side": state.debt_side, "letters": state.symmetric_debt},
    }


def close_proposal(state: ConstructionState, proposal: Proposal) -> dict[str, Any]:
    """Audit a final proposed middle; this never repairs or invents text."""
    event = {"proposal_id": proposal.proposal_id, "parent_state_id": state.state_id, "source": proposal.source,
             "operation": proposal.operation, "input": asdict(proposal), "accepted": False, "closures": []}
    try:
        if proposal.parent_state_id != state.state_id:
            raise ValueError("parent_state_mismatch")
        if proposal.operation != "finalize":
            raise ValueError("finalization_requires_finalize_operation")
        center = _safe_normalize(proposal.center_text)
        _bounded(center, "center")
        # The full tape check deliberately permits the midpoint to land within
        # a word analysis.  It does not force a token-boundary center.
        tape = state.left_tape + center + state.right_tape
        event["tape_sha256"] = _text_digest(tape)
        event["letters"] = len(tape)
        if tape != tape[::-1]:
            raise ValueError("finalized_tape_is_not_an_exact_letter_palindrome")
        final_analysis = analyze_tape(tape)
        segmentations = enumerate_segmentations(tape)
        if len(segmentations) != final_analysis.complete_segmentations:
            raise AssertionError("lexical_chart_and_full_enumeration_disagree")
        event["accepted"] = True
        event["all_viable_lexical_analyses"] = len(segmentations)
        event["final_analysis"] = {"boundary_positions": final_analysis.boundary_positions,
                                   "open_prefixes": final_analysis.open_prefixes,
                                   "syntax_obligations": [asdict(row) for row in final_analysis.syntax_obligations]}
        for words in segmentations:
            rendered = " ".join(words).capitalize() + "."
            checks = mechanical_admission_checks(rendered, min_letters=MIN_CANDIDATE_LETTERS,
                                                 max_letters=MAX_CANDIDATE_LETTERS)
            # This is diagnostic evidence about a possible analysis only.  It
            # neither asserts fluency nor replaces blinded human assessment.
            syntax = SyntaxObligation("tail") in final_analysis.syntax_obligations
            closure = {"rendered": rendered, "render_sha256": _text_digest(rendered), "words": words,
                       "letters": len(tape), "exact_letter_palindrome": True,
                       "mechanical_checks": checks, "mechanically_eligible": all(checks.values()),
                       "tentative_syntax_tail_diagnostic": syntax,
                       "provenance": {"proposal_chain": state.proposal_history + (proposal.proposal_id,),
                                      "source": proposal.source, "external_provenance": "unverified"},
                       "human_study": "not_run"}
            event["closures"].append(closure)
        return event
    except (AssertionError, TypeError, ValueError) as error:
        event["rejection"] = str(error)
        return event


class FairProposalQueue:
    """Round-robin queue over parent regions; no branch owns the frontier."""

    def __init__(self, proposals: Iterable[Proposal]):
        self.by_parent: dict[str, deque[Proposal]] = defaultdict(deque)
        self.parents: deque[str] = deque()
        for proposal in proposals:
            if proposal.parent_state_id not in self.by_parent:
                self.parents.append(proposal.parent_state_id)
            self.by_parent[proposal.parent_state_id].append(proposal)

    def pop_ready(self, states: dict[str, ConstructionState]) -> Proposal | None:
        for _ in range(len(self.parents)):
            parent = self.parents.popleft()
            proposals = self.by_parent[parent]
            if not proposals:
                continue
            if parent not in states:
                self.parents.append(parent)
                continue
            proposal = proposals.popleft()
            if proposals:
                self.parents.append(parent)
            return proposal
        return None

    def unresolved(self) -> Iterator[Proposal]:
        for parent in self.parents:
            yield from self.by_parent[parent]


def candidate_records(finalization_events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only 100+ exact surfaces clearing every central mechanical gate."""
    records = []
    for event in finalization_events:
        for closure in event.get("closures", []):
            if closure["letters"] >= MIN_CANDIDATE_LETTERS and closure["exact_letter_palindrome"] and closure["mechanically_eligible"]:
                records.append(closure)
    return records


def run_pilot(proposals: Iterable[Proposal], *, state_budget: int = DEFAULT_STATE_BUDGET) -> dict[str, Any]:
    """Run a deterministic, fair proposal ledger without calling a model."""
    if state_budget < 1:
        raise ValueError("state_budget must be positive")
    ordinary, finals = [], []
    for proposal in proposals:
        (finals if proposal.operation == "finalize" else ordinary).append(proposal)
    root = root_state()
    states = {root.state_id: root}
    queue = FairProposalQueue(ordinary)
    events: list[dict[str, Any]] = []
    while len(states) < state_budget:
        proposal = queue.pop_ready(states)
        if proposal is None:
            break
        child, event = apply_proposal(states[proposal.parent_state_id], proposal)
        events.append(event)
        if child is not None:
            states[child.state_id] = child
    for proposal in queue.unresolved():
        events.append({"proposal_id": proposal.proposal_id, "parent_state_id": proposal.parent_state_id,
                       "source": proposal.source, "operation": proposal.operation, "accepted": False,
                       "rejection": "unresolved_parent_or_state_budget", "input": asdict(proposal)})
    final_events = []
    # Finalizations use their stated parent state and are also logged, even
    # when a prior edit failed.  They create no child and cannot monopolize
    # the edit queue.
    for proposal in finals:
        parent = states.get(proposal.parent_state_id)
        if parent is None:
            final_events.append({"proposal_id": proposal.proposal_id, "parent_state_id": proposal.parent_state_id,
                                 "source": proposal.source, "operation": proposal.operation, "accepted": False,
                                 "rejection": "unresolved_parent_or_state_budget", "input": asdict(proposal), "closures": []})
        else:
            final_events.append(close_proposal(parent, proposal))
    candidates = candidate_records(final_events)
    return {
        "status": "assisted_candidate_construction_pilot_complete",
        "config": {"date": DATE, "state_budget": state_budget, "max_proposal_letters": MAX_PROPOSAL_LETTERS,
                   "candidate_letter_range": [MIN_CANDIDATE_LETTERS, MAX_CANDIDATE_LETTERS],
                   "scheduling": "round-robin across parent regions; all queued proposals replayed or logged",
                   "model_calls": "none", "full_sentence_reflection": False, "word_order_mirror_construction": False,
                   "central_mechanical_admission": True, "readability_or_semantic_scores": "diagnostic only"},
        "lexical_evidence": lexical_evidence()["evidence"], "root_state_id": root.state_id,
        "states_created": len(states), "state_budget_exhausted": len(states) >= state_budget,
        "proposal_events": events, "finalization_events": final_events,
        "eligible_100_plus_candidates": candidates,
        "human_reader_study": {"triggered": False, "reason": "no mechanically eligible 100+ exact candidate" if not candidates else "candidate exists; study package has not been run"},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "lexicon_path": "data/lexicon.txt"},
    }


def fixture_proposals(root_id: str) -> tuple[Proposal, ...]:
    """Deterministic mechanics fixture, intentionally far below candidate size."""
    return (
        Proposal("fixture-continue", root_id, {"kind": "fixture", "label": "queue-and-debt"}, "continue", "ab", "ba"),
        Proposal("fixture-reopen", "__child_of_fixture_continue__", {"kind": "fixture", "label": "global-reopen"}, "reopen",
                 left_region=Region(0, 1, "c"), right_region=Region(1, 2, "c")),
    )


def _load_proposals(path: Path | None) -> list[Proposal]:
    if path is None:
        # Resolve the fixture's child ID through a preliminary deterministic
        # replay so it remains a real parent-linked proposal, not a magic ID.
        root = root_state()
        first = Proposal("fixture-continue", root.state_id, {"kind": "fixture", "label": "queue-and-debt"}, "continue", "ab", "ba")
        child, _ = apply_proposal(root, first)
        assert child is not None
        return [first, Proposal("fixture-reopen", child.state_id, {"kind": "fixture", "label": "global-reopen"}, "reopen",
                                left_region=Region(0, 1, "c"), right_region=Region(1, 2, "c")),
                Proposal("fixture-finalize", child.state_id, {"kind": "fixture", "label": "too-short-control"}, "finalize", center_text="")]
    raw = json.loads(path.read_text())
    rows = raw["proposals"] if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("proposal file must be a JSON list or an object with a proposals list")
    return [Proposal.from_json(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, help="JSON proposal ledger; no model is called")
    parser.add_argument("--state-budget", type=int, default=DEFAULT_STATE_BUDGET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_pilot(_load_proposals(args.proposals), state_budget=args.state_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "states_created": result["states_created"],
                      "proposal_events": len(result["proposal_events"]), "finalizations": len(result["finalization_events"]),
                      "eligible_100_plus_candidates": len(result["eligible_100_plus_candidates"]),
                      "human_reader_study_triggered": result["human_reader_study"]["triggered"]}, sort_keys=True))


if __name__ == "__main__":
    main()
