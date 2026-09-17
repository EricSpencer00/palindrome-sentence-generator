"""Closed-form grammar-family probe for clause growth.

The state is a list of typed, complete clauses.  ``extend`` appends a fresh
clause and recomputes the character obligations from the whole rendered
state; it never copies a finished tape or emits its reverse.  This is a
deliberately small *construction family*, not a search sweep: one base and
one extension are retained as evidence.  The probe currently demonstrates
that ordinary clause growth changes the state but does not close the exact
character equation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-closed-form-grammar-family-20260917.json"
EXPERIMENT_ID = "luna-closed-form-grammar-family-20260917"
SIGNATURE = (
    "typed-complete-clause-monoid|append-state-operator|live-character-obligations|"
    "fresh-nonmirror-prose|independent-pointer-sha"
)
SEED_BENCHMARK = "An aide rips nine memos; some men inspire Diana."


@dataclass(frozen=True)
class Clause:
    key: str
    text: str
    sense: str


@dataclass(frozen=True)
class GrammarState:
    clauses: tuple[Clause, ...]

    def render(self) -> str:
        return " ".join(clause.text for clause in self.clauses)

    def extend(self, clause: Clause) -> "GrammarState":
        """The sole growth operator: append one complete semantic clause."""
        if clause.key in {item.key for item in self.clauses}:
            raise ValueError("clause keys must be fresh")
        if not re.fullmatch(r"[A-Z][^.!?]*[.!?]", clause.text):
            raise ValueError("an extension must be a complete capitalized clause")
        return GrammarState(self.clauses + (clause,))


BASE = GrammarState(
    (
        Clause("dawn-catalog", "At dawn, the archivist catalogs river maps beside the old observatory.", "catalogue a place-bound collection"),
        Clause("noon-courier", "Before noon, a patient courier delivers sealed parcels to the village clinic.", "deliver medical parcels"),
    )
)
EXTENSION = Clause(
    "evening-garden",
    "By evening, the groundskeeper labels young seedlings near the rain barrel.",
    "label living specimens for later care",
)


def tape(text: str) -> str:
    return "".join(character.lower() for character in text if "a" <= character.lower() <= "z")


def pointer_audit(text: str) -> dict[str, object]:
    value = tape(text)
    mismatches = []
    left, right = 0, len(value) - 1
    while left < right:
        if value[left] != value[right]:
            mismatches.append({"left": left, "right": right, "left_char": value[left], "right_char": value[right]})
        left += 1
        right -= 1
    return {"algorithm": "independent_two_pointer", "letters": len(value), "exact": bool(value) and not mismatches, "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None}


def hash_audit(text: str) -> dict[str, object]:
    value = tape(text)
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "forward": forward, "reverse": reverse, "exact": bool(value) and forward == reverse}


def obligation_ledger(state: GrammarState) -> list[dict[str, object]]:
    """Record outer-pair debt after every complete clause is appended."""
    value = tape(state.render())
    rows = []
    emitted = 0
    for clause in state.clauses:
        emitted += len(tape(clause.text))
        pairs = [(i, len(value) - 1 - i) for i in range(len(value) // 2)]
        resolved = sum(value[i] == value[j] for i, j in pairs[: min(emitted, len(pairs))])
        first = next((i for i, j in pairs if value[i] != value[j]), None)
        rows.append({"clause": clause.key, "emitted_letters": emitted, "outer_pairs": len(pairs), "resolved_prefix_pairs": resolved, "first_unresolved_offset": first, "remaining_pair_debt": len(pairs) - resolved})
    return rows


def anti_shortcut(state: GrammarState) -> dict[str, object]:
    words = [word.lower() for word in re.findall(r"[A-Za-z]+", state.render())]
    unique_content = [word for word in words if word not in {"a", "the", "at", "before", "by", "to", "near", "the"}]
    return {
        "seed_copied_or_wrapped": SEED_BENCHMARK.lower() in state.render().lower(),
        "finished_tape_reversal": False,
        "word_order_mirror": words == list(reversed(words)),
        "repeated_nonfunction_word": len(unique_content) != len(set(unique_content)),
        "repeated_clause_unit": len({clause.key for clause in state.clauses}) != len(state.clauses),
        "posthoc_character_edit": False,
    }


def novelty_preflight() -> dict[str, object]:
    registry = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    entries = registry.get("entries", []) + registry.get("excluded", [])
    collisions = [item.get("id") for item in entries if item.get("id") != EXPERIMENT_ID and item.get("signature") == SIGNATURE]
    if collisions:
        raise RuntimeError(f"duplicate family signature rejected: {collisions}")
    return {"status": "passed", "performed_before_rendering": True, "registry_entries_read": len(entries), "exact_signature_collision": False, "duplicate_sweep_rejected": True, "single_base_plus_one_extension": True}


def evidence(state: GrammarState, label: str) -> dict[str, object]:
    rendered = state.render()
    pointer = pointer_audit(rendered)
    return {"label": label, "rendered": rendered, "clauses": [{"key": c.key, "sense": c.sense, "complete": bool(re.fullmatch(r"[A-Z][^.!?]*[.!?]", c.text))} for c in state.clauses], "letters": pointer["letters"], "pointer_audit": pointer, "hash_audit": hash_audit(rendered), "obligation_ledger": obligation_ledger(state), "anti_shortcut": anti_shortcut(state)}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    extended = BASE.extend(EXTENSION)
    base_row, extended_row = evidence(BASE, "base"), evidence(extended, "one_fresh_extension")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_exact_closure",
        "reader_eligible": False,
        "family": {"nonterminal": "STATE", "production": "STATE -> STATE COMPLETE_CLAUSE", "operator": "extend(state, fresh_complete_clause)", "arbitrary_size": True, "base_clause_count": len(BASE.clauses), "extension_clause_count": len(extended.clauses)},
        "proof_obligation_ledger": {"closure_equation": "normalized_letters(text)[i] == normalized_letters(text)[N-1-i]", "base": base_row["obligation_ledger"], "extension": extended_row["obligation_ledger"], "all_clauses_complete": all(row["complete"] for row in extended_row["clauses"]), "closed_form_proved": False},
        "novelty_preflight": preflight,
        "candidates": {"base": base_row, "one_extension": extended_row},
        "provenance": {"source": "fresh hand-authored observatory, clinic, and garden scenes", "seed_role": "38-letter benchmark metadata only", "seed_used_in_output": False, "selection": "typed complete clauses selected before rendering", "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "audits": ["independent two-pointer", "forward/reverse SHA-256", "live clause obligation ledger", "novelty preflight"]},
        "next_test": {"reader_facing": "Blindly rate the base and one-extension texts for ordinary meaning, then compare against shuffled-clause and intact-prose controls.", "operator": "replace the first unresolved boundary with one fresh typed clause, recompute the entire ledger, and retain only a complete non-mirrored clause", "state_change": "append or replace a semantic clause; never mutate characters after rendering"},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run()["proof_obligation_ledger"], sort_keys=True))
