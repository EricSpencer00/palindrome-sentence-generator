"""Constrained semantic authoring with a Python-owned character ledger.

The proposal interface contains typed role alternatives and optional semantic
attachments, never a requested palindrome or a supplied full tape.  Python
assembles each complete surface from role choices, screens known material
before construction, and performs the only exactness decision with an
outside-in character ledger.  A mismatch is a rejected construction and can
never be returned as a candidate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 30, 160
WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class ModelResponse:
    """A semantic proposal, not a generated sentence or palindrome tape."""
    proposal_id: str
    frame: str
    role_options: tuple[tuple[str, tuple[str, ...]], ...]
    verb_object_types: tuple[tuple[str, str], ...]
    verb_subject_type: str
    verbatim_text: str | None = None

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(role for role, _ in self.role_options)

    @property
    def choices(self) -> tuple[tuple[str, ...], ...]:
        return tuple(words for _, words in self.role_options)


ROLE_LEXICON = {
    "subject_det": frozenset({"a", "an", "the"}),
    "subject_adj": frozenset({"agile", "careful", "patient", "skilled"}),
    "subject": frozenset({"artist", "editor", "teacher", "writer"}),
    "verb": frozenset({"builds", "drafts", "paints", "repairs"}),
    "object_det": frozenset({"a", "an", "the"}),
    "object_adj": frozenset({"brief", "clean", "detailed", "solid"}),
    "object": frozenset({"canvas", "letter", "model", "report"}),
    "prep": frozenset({"in", "at"}),
    "place_det": frozenset({"a", "an", "the"}),
    "place_adj": frozenset({"quiet", "public", "remote", "small"}),
    "place": frozenset({"garden", "office", "studio", "workshop"}),
}
INDEPENDENT_SEMANTICS = {
    "artist": "person", "editor": "person", "teacher": "person", "writer": "person",
    "canvas": "artifact", "letter": "artifact", "model": "artifact", "report": "artifact",
    "garden": "place", "office": "place", "studio": "place", "workshop": "place",
}


def authored_proposals() -> tuple[ModelResponse, ...]:
    # These are ordinary semantic alternatives: Python owns the surface
    # assembly and no alternative contains a pre-authored full sentence.
    common = (
        ("subject_det", ("a", "an", "the")),
        ("subject_adj", tuple(sorted(ROLE_LEXICON["subject_adj"]))),
        ("subject", tuple(sorted(ROLE_LEXICON["subject"]))),
        ("verb", tuple(sorted(ROLE_LEXICON["verb"]))),
        ("object_det", ("a", "an", "the")),
        ("object_adj", tuple(sorted(ROLE_LEXICON["object_adj"]))),
        ("object", tuple(sorted(ROLE_LEXICON["object"]))),
        ("prep", ("in", "at")),
        ("place_det", ("a", "an", "the")),
        ("place_adj", tuple(sorted(ROLE_LEXICON["place_adj"]))),
        ("place", tuple(sorted(ROLE_LEXICON["place"]))),
    )
    return (
        ModelResponse("repair_locative", "svo_with_locative_attachment", common,
                      tuple((verb, "artifact") for verb in sorted(ROLE_LEXICON["verb"])), "person"),
        ModelResponse("paint_locative", "svo_with_locative_attachment", common,
                      (("paints", "artifact"),), "person"),
    )


def _catalogue() -> frozenset[str]:
    return frozenset(json.loads((ROOT / "data" / "known_palindromes.json").read_text()))


def preconstruction_gate(response: ModelResponse, catalogue: frozenset[str] | None = None) -> dict:
    """Reject known material before any role product or ledger is built."""
    catalogue = catalogue if catalogue is not None else _catalogue()
    submitted = normalize_letters(response.verbatim_text or "")
    hits = sorted(trigger for trigger in catalogue if trigger and trigger in submitted)
    role_hits = []
    for role, options in response.role_options:
        for option in options:
            normalized = normalize_letters(option)
            if normalized in catalogue:
                role_hits.append({"role": role, "option": option, "normalized": normalized})
    reasons = []
    if hits: reasons.append("known_catalogue_material_in_model_response")
    if role_hits: reasons.append("known_catalogue_material_in_role_option")
    return {"accepted": not reasons, "construction_started": False, "reasons": reasons,
            "catalogue_hits": hits, "role_hits": role_hits}


def assemble_surface(response: ModelResponse, words: tuple[str, ...]) -> str:
    if len(words) != len(response.roles):
        raise ValueError("role/word arity mismatch")
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def outside_in_ledger(text: str) -> dict:
    """The only exactness authority: compare actual rendered letters outside-in."""
    tape = normalize_letters(text)
    events = []
    for pair_index in range(len(tape) // 2):
        right_index = len(tape) - 1 - pair_index
        event = {"pair": pair_index + 1, "left_index": pair_index, "right_index": right_index,
                 "left": tape[pair_index], "right": tape[right_index],
                 "equal": tape[pair_index] == tape[right_index]}
        events.append(event)
        if not event["equal"]:
            return {"exact": False, "letters": len(tape), "events": events,
                    "first_mismatch": event, "normalized_sha256": sha256(tape.encode()).hexdigest()}
    return {"exact": bool(tape), "letters": len(tape), "events": events,
            "first_mismatch": None, "normalized_sha256": sha256(tape.encode()).hexdigest()}


def independent_parse(response: ModelResponse, text: str) -> dict:
    """Parse the completed surface from a separate role inventory."""
    tokens = tuple(WORD_RE.findall(text.lower()))
    if len(tokens) != 11:
        return {"ok": False, "reason": "wrong_role_arity", "tokens": list(tokens)}
    roles = ("subject_det", "subject_adj", "subject", "verb", "object_det", "object_adj", "object", "prep", "place_det", "place_adj", "place")
    for role, token in zip(roles, tokens):
        if token not in ROLE_LEXICON.get(role, frozenset()):
            return {"ok": False, "reason": f"unknown_{role}", "tokens": list(tokens)}
    subject, verb, obj, place = tokens[2], tokens[3], tokens[6], tokens[10]
    object_type = INDEPENDENT_SEMANTICS.get(obj)
    valency = object_type == "artifact" and dict(response.verb_object_types).get(verb) == object_type
    agreement = tokens[0] in {"a", "an", "the"} and tokens[4] in {"a", "an", "the"} and tokens[8] in {"a", "an", "the"}
    modifier_phonology = all(det == "the" or det == ("an" if adj[0] in "aeiou" else "a")
                             for det, adj in ((tokens[0], tokens[1]), (tokens[4], tokens[5]), (tokens[8], tokens[9])))
    semantic_subject = INDEPENDENT_SEMANTICS.get(subject) == response.verb_subject_type
    semantic_place = INDEPENDENT_SEMANTICS.get(place) == "place"
    return {"ok": agreement and modifier_phonology and valency and semantic_subject and semantic_place,
            "agreement_ok": agreement, "modifier_phonology_ok": modifier_phonology,
            "valency_ok": valency, "semantic_subject_ok": semantic_subject,
            "semantic_attachment_ok": semantic_place, "tokens": list(tokens), "frame": response.frame}


def construct_candidate(response: ModelResponse, words: tuple[str, ...], catalogue: frozenset[str] | None = None) -> dict:
    screened = preconstruction_gate(response, catalogue)
    if not screened["accepted"]:
        return {"record_kind": "preconstruction_rejection", "proposal_id": response.proposal_id,
                "rendered": None, "preconstruction_gate": screened,
                "construction_started": False, "reader_status": "not a candidate"}
    text = assemble_surface(response, words)
    ledger = outside_in_ledger(text)
    parsed = independent_parse(response, text)
    gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in gate.items() if not value]
    if not ledger["exact"]: codes.append("outside_in_ledger_mismatch")
    if not parsed["ok"]: codes.append("independent_parse_failed")
    return {"record_kind": "assembled_surface_rejection_or_candidate", "proposal_id": response.proposal_id,
            "rendered": text, "construction_started": True, "outside_in_ledger": ledger,
            "independent_parse": parsed, "central_admission": gate,
            "mechanically_admitted": not codes, "rejection_codes": codes,
            "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(max_combinations: int = 20_000) -> dict:
    if max_combinations < 1: raise ValueError("max_combinations must be positive")
    proposals = authored_proposals(); catalogue = _catalogue()
    screens = [{"proposal_id": proposal.proposal_id, **preconstruction_gate(proposal, catalogue)}
               for proposal in proposals]
    records = []; tested = 0
    for proposal, screen in zip(proposals, screens):
        screened = screen
        if not screened["accepted"]: continue
        for words in itertools.product(*proposal.choices):
            if tested >= max_combinations: break
            tested += 1
            row = construct_candidate(proposal, tuple(words), catalogue)
            if row["outside_in_ledger"]["exact"] or len(records) < 100:
                records.append(row)
        if tested >= max_combinations: break
    admitted = [row for row in records if row.get("mechanically_admitted")]
    return {"status": "constrained_character_ledger_authoring",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                       "max_combinations": max_combinations, "python_owns_surface_assembly": True,
                       "python_owns_outside_in_exact_ledger": True, "preconstruction_catalogue_gate": True,
                       "typed_semantic_alternatives_only": True, "full_tape_reflection": False,
                       "independent_complete_reparse": True, "corpus_generation": False,
                       "human_readability_required_after_admission": True},
            "proposal_count": len(proposals), "all_proposals_screened": True,
            "proposal_screening": screens, "combinations_tested": tested,
            "combination_cap_reached": tested >= max_combinations,
            "truncated_by_combination_cap": tested >= max_combinations,
            "records": records, "exact_candidates": [r for r in records if r["outside_in_ledger"]["exact"]],
            "admitted_candidates": admitted,
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "material": "typed semantic alternatives assembled into surfaces by Python; no catalogue text used",
                           "known_catalogue_sha256": sha256("\n".join(sorted(catalogue)).encode()).hexdigest()},
            "next_construction_operator": "Ask for a new semantically licensed role/attachment alternative, screen it before construction, and rerun the Python ledger; never patch a mismatching token or reflect a tape.",
            "reader_facing_next_test": "Only an admitted exact surface can enter blinded intact-prose versus shuffled-control reading; programmatic checks do not certify readability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", required=True, type=Path); parser.add_argument("--max-combinations", type=int, default=20_000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True); result = run(args.max_combinations); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "tested": result["combinations_tested"], "exact": len(result["exact_candidates"]), "admitted": len(result["admitted_candidates"])}, indent=2))


if __name__ == "__main__": main()
