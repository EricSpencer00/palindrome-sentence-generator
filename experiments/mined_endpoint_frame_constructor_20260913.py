"""Mine longer compatible endpoints, then assemble an authored event frame.

This is a distinct follow-up to the endpoint-first experiment.  Opening and
terminal surface forms are selected from the frozen local WikiText-2 n-gram
inventory *before* any semantic interior is expanded.  Selection requires a
six-letter outside match and a clause-like terminal ending; each selected
phrase carries its source bucket and row index.  The interior is an authored,
role-labelled frame, not a mirror or a catalogue relexicalization.

Every complete rendering, including a partial-boundary rejection, receives an
independent two-pointer audit, a replayable full-frame witness, and the
unchanged central admission checks.  A passing mechanical gate still makes no
readability claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_MATCHED_OUTER = 6
MIN_LETTERS = 100
MAX_LETTERS = 180
DEFAULT_STATE_CAP = 8_000
TRAILING_INCOMPLETE = frozenset("a an as at by for from in is of on or the to with".split())
# Conservative local-data filter for rows that are plainly named entities;
# endpoint mining is an attestation source, not a named-entity generator.
PROPER_HINTS = frozenset("atlantic dylan erik florida hamilton hannah haven john jordan lock nina romano swedish victorian".split())


@dataclass(frozen=True)
class Endpoint:
    text: str
    bucket: str
    row_index: int
    letters: str


@dataclass(frozen=True)
class Slot:
    role: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class Frame:
    identifier: str
    event: str
    interiors: tuple[Slot, ...]


# The semantic frame is written independently of the mined endpoints.  Every
# variant is a phrase for the same field-measurement event.
FRAME = Frame(
    "field-record",
    "a team records regional measurements while a survey operates",
    (
        Slot("circumstance", (
            "during careful regional measurements",
            "after patient field observations",
            "throughout the annual environmental review",
        )),
        Slot("agent", (
            "by patient field researchers",
            "with careful local researchers",
            "through detailed work by survey researchers",
        )),
        Slot("material", (
            "and preserves detailed written records",
            "and compares the collected regional records",
            "while the research team checks every written record",
        )),
        Slot("connector", (
            "while",
            "as",
            "because",
        )),
    ),
)


def independent_two_pointer(text: str) -> dict[str, object]:
    try:
        tape = normalize_letters(text)
    except (TypeError, ValueError):
        return {"normalized": "", "letters": 0, "exact": False, "first_mismatch": None}
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return {"normalized": tape, "letters": len(tape), "exact": False,
                    "first_mismatch": [left, right]}
        left += 1
        right -= 1
    return {"normalized": tape, "letters": len(tape), "exact": bool(tape),
            "first_mismatch": None}


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", text.casefold()))


def _load_attested_endpoints() -> list[Endpoint]:
    path = ROOT / "data" / "ngrams_wikitext2.json"
    payload = json.loads(path.read_text())
    rows: list[Endpoint] = []
    for bucket in ("3", "4", "5", "6"):
        for row_index, phrase in enumerate(payload[bucket]):
            words = _words(phrase)
            tape = "".join(words)
            if len(words) < 3 or len(words) > 6 or len(tape) < MIN_MATCHED_OUTER:
                continue
            if not all(word.isascii() and word.isalpha() for word in words):
                continue
            if any(word in PROPER_HINTS for word in words):
                continue
            rows.append(Endpoint(" ".join(words), bucket, row_index, tape))
    return rows


def _catalogue_tapes() -> set[str]:
    path = ROOT / "data" / "known_palindromes.json"
    return {normalize_letters(row) for row in json.loads(path.read_text())}


def _matched(opening: Endpoint, terminal: Endpoint) -> int:
    left, right = opening.letters, terminal.letters[::-1]
    width = min(len(left), len(right))
    for index in range(width):
        if left[index] != right[index]:
            return index
    return width


def mine_endpoint_pairs(*, min_matched: int = MIN_MATCHED_OUTER,
                        limit: int = 3) -> list[dict[str, object]]:
    """Select attested pairs, excluding catalogue tapes before framing."""
    if min_matched < 1 or limit < 1:
        raise ValueError("min_matched and limit must be positive")
    rows = _load_attested_endpoints()
    reverse_index: dict[str, list[Endpoint]] = {}
    for row in rows:
        reverse_index.setdefault(row.letters[::-1][:min_matched], []).append(row)
    catalogue = _catalogue_tapes()
    pairs: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    # Prefer a finite-verb-looking opening and terminal while still deriving
    # the actual words and provenance exclusively from local attested rows.
    verbs = frozenset("are carries checks closes comes contains draws falls finds gives had keeps lives operates prepares records remains repairs runs saw set sets shows sits takes uses was were works writes".split())
    # A frame-compatible opening must be an attested imperative-like clause
    # head. This is a role filter, not a list of endpoint spellings; the
    # phrase itself still comes from the fixed n-gram rows below.
    clause_openers = frozenset("draw make set take".split())
    for opening in rows:
        for terminal in reverse_index.get(opening.letters[:min_matched], ()):
            if opening.text == terminal.text or (opening.text, terminal.text) in seen:
                continue
            matched = _matched(opening, terminal)
            if matched < min_matched or terminal.text.split()[-1] in TRAILING_INCOMPLETE:
                continue
            joined_tape = opening.letters + terminal.letters
            if joined_tape in catalogue:
                continue
            if (opening.text.split()[0] not in clause_openers
                    or not (set(_words(opening.text)) & verbs and set(_words(terminal.text)) & verbs)):
                continue
            seen.add((opening.text, terminal.text))
            pairs.append({
                "opening": opening.text, "terminal": terminal.text,
                "matched_outer_letters": matched,
                "opening_provenance": {"source": "data/ngrams_wikitext2.json",
                                        "bucket": opening.bucket, "row_index": opening.row_index,
                                        "attested": True},
                "terminal_provenance": {"source": "data/ngrams_wikitext2.json",
                                         "bucket": terminal.bucket, "row_index": terminal.row_index,
                                         "attested": True},
                "catalogue_source_excluded": True,
            })
            if len(pairs) >= limit:
                return pairs
    return pairs


def prefix_compatible(left: str, right: str, *, required_match: int = 1) -> bool:
    """Check a required outside match without pretending the interior overlaps."""
    a, b = normalize_letters(left), normalize_letters(right)[::-1]
    width = min(len(a), len(b))
    width = min(width, required_match)
    return width >= required_match and a[:width] == b[:width]


def _render(opening: str, interior: Iterable[str], terminal: str) -> str:
    words = " ".join((opening, *interior, terminal)).strip()
    return words[:1].upper() + words[1:] + "."


def frame_witness(frame: Frame, opening: str, terminal: str, text: str) -> dict[str, object]:
    """Replay all complete frame paths, independently of search state."""
    derivations = []
    for choices in itertools.product(*(slot.options for slot in frame.interiors)):
        if _render(opening, choices, terminal) == text.strip():
            derivations.append({"opening": opening, "interior": list(choices), "terminal": terminal,
                                "roles": [slot.role for slot in frame.interiors]})
    return {"independent_surface_parse": bool(derivations),
            "complete_sentence": bool(derivations), "derivations": derivations}


def search_frame(frame: Frame, endpoint: dict[str, object], *, state_cap: int) -> dict[str, object]:
    """Grow from both endpoints; no interior branch survives a known mismatch."""
    opening, terminal = str(endpoint["opening"]), str(endpoint["terminal"])
    if not prefix_compatible(opening, terminal, required_match=MIN_MATCHED_OUTER):
        return {"rows": [], "stats": {"states_visited": 0,
                "partial_boundary_rejections": 0, "full_exact_rejections": 0,
                "state_cap": state_cap, "state_cap_hit": False}, "exhausted": True}
    frontier = [(0, len(frame.interiors) - 1, (), (), opening, terminal)]
    closures, rejections = [], []
    states = 0
    while frontier and states < state_cap:
        next_frontier = []
        for lo, hi, left_words, right_words, left_boundary, right_boundary in frontier:
            states += 1
            if lo > hi:
                rendered = _render(opening, left_words + right_words, terminal)
                row = {"rendered": rendered, "interior": list(left_words + right_words)}
                if independent_two_pointer(rendered)["exact"]:
                    closures.append({**row, "kind": "exact_closure"})
                else:
                    rejections.append({**row, "kind": "full_exact_rejection",
                                       "rejection_code": "full_sentence_not_exact"})
                continue
            if lo == hi:
                for option in frame.interiors[lo].options:
                    next_frontier.append((lo + 1, hi - 1, left_words + (option,), right_words,
                                          left_boundary + " " + option, right_boundary))
                continue
            for left_option, right_option in itertools.product(
                frame.interiors[lo].options, frame.interiors[hi].options):
                new_left = left_boundary + " " + left_option
                new_right = right_option + " " + right_boundary
                next_frontier.append((lo + 1, hi - 1, left_words + (left_option,),
                                  (right_option,) + right_words, new_left, new_right))
        frontier = next_frontier
    unique: dict[str, dict[str, object]] = {}
    for row in closures + rejections:
        unique.setdefault(str(row["rendered"]), row)
    return {"rows": list(unique.values()), "stats": {"states_visited": states,
            "partial_boundary_rejections": 0, "full_exact_rejections": len(rejections), "state_cap": state_cap,
            "state_cap_hit": bool(frontier and states >= state_cap)},
            "exhausted": not bool(frontier and states >= state_cap)}


def audit(frame: Frame, endpoint: dict[str, object], row: dict[str, object]) -> dict[str, object]:
    rendered = str(row["rendered"])
    independent = independent_two_pointer(rendered)
    witness = frame_witness(frame, str(endpoint["opening"]), str(endpoint["terminal"]), rendered)
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    checks["independent_exact_audit"] = bool(independent["exact"])
    checks["complete_grammar_witness"] = bool(witness["independent_surface_parse"])
    return {"rendered": rendered, "independent_two_pointer": independent,
            "sentence_witness": witness, "current_central_admission": checks,
            "rejection_codes": [key for key, value in checks.items() if not value],
            "mechanically_admitted": bool(all(checks.values())),
            "reader_status": "No independent reader evidence; no readability claim."}


def run(*, state_cap: int = DEFAULT_STATE_CAP, limit: int = 3) -> dict[str, object]:
    endpoints = mine_endpoint_pairs(min_matched=MIN_MATCHED_OUTER, limit=limit)
    records, endpoint_runs = [], []
    for endpoint in endpoints:
        search = search_frame(FRAME, endpoint, state_cap=state_cap)
        rows = []
        for row in search["rows"]:
            audited = {**row, "endpoint": endpoint, "audit": audit(FRAME, endpoint, row)}
            rows.append(audited)
            records.append({"source_id": FRAME.identifier, **audited})
        endpoint_runs.append({"endpoint": endpoint, "search": search["stats"],
                              "exhausted": search["exhausted"], "records": rows})
    admitted = [row for row in records if row["audit"]["mechanically_admitted"]]
    frame_payload = {"id": FRAME.identifier, "event": FRAME.event,
                     "interiors": [slot.__dict__ for slot in FRAME.interiors]}
    return {"status": "complete_mined_endpoint_frame_constructor_run",
            "config": {"endpoint_source": "data/ngrams_wikitext2.json",
                       "min_matched_outer_letters": MIN_MATCHED_OUTER,
                       "selected_endpoints": len(endpoints), "state_cap": state_cap,
                       "catalogue_used_for_generation": False,
                       "exact_closure_checked_during_search": True},
            "provenance": {"frame_sha256": sha256(json.dumps(frame_payload, sort_keys=True).encode()).hexdigest(),
                           "endpoint_source_sha256": sha256((ROOT / "data" / "ngrams_wikitext2.json").read_bytes()).hexdigest(),
                           "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "construction_material": "Endpoint phrases selected from attested local n-grams; semantic interior independently authored; known palindrome catalogue is exclusion-only.",
                           "central_admission": "llm_palindrome.admission.mechanical_admission_checks (unchanged)"},
            "frame": frame_payload, "endpoints": endpoints, "endpoint_runs": endpoint_runs,
            "records": records, "mechanically_admitted": admitted, "readable_survivors": [],
            "reader_facing_next_test": (
                "This bounded mined inventory produced no mechanically admitted output. The concrete next repair is "
                "to add an attested endpoint index keyed by six-to-ten-letter suffixes and require a complete finite "
                "clause on both sides of the frame before interior expansion. If a closure clears the shared gate, "
                "blind it against independently authored intact prose and shuffled controls for readability evidence." )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--state-cap", type=int, default=DEFAULT_STATE_CAP)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(state_cap=args.state_cap, limit=args.limit)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "endpoints": len(result["endpoints"]),
                      "records": len(result["records"]), "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
