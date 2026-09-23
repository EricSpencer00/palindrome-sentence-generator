"""Search a four-event scene chain through a live bilateral residual grammar.

The left side is a typed SVO chain: the next subject is always the previous
object.  The right side is generated backwards, one clause at a time, from a
character residual.  Its next backwards clause is constrained by the subject
of the clause already discovered, so no complete reciprocal-chain catalogue
exists before the residual search.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-672-discourse-linked-reverse-chain-20260922.json"
SNAPSHOT = ROOT / "runs" / "incumbent-672-global-novelty-snapshot-20260922.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_CUT, RIGHT_CUT = 48, 520
CHAIN_LENGTH = 4

RELATION_RE = re.compile(r"\b([A-Z][a-z]+)\s+(sees|stops|spots)\s+([A-Z][a-z]+)\b")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    mismatch = next(
        ({"offset": i, "left": tape[i], "right": tape[-1 - i]}
         for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "normalized_letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def raw_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            seen += 1
            if seen == count:
                return index + 1
    raise ValueError(count)


@dataclass(frozen=True)
class Entity:
    surface: str
    kind: str = "animate"


@dataclass(frozen=True)
class Predicate:
    surface: str
    relation_family: str
    valency: str = "transitive"


@dataclass(frozen=True)
class Clause:
    subject: Entity
    predicate: Predicate
    object: Entity

    @property
    def surface(self) -> str:
        return f"{self.subject.surface} {self.predicate.surface} {self.object.surface}."

    @property
    def tape(self) -> str:
        return normalize(self.surface)

    @property
    def relation(self) -> str:
        return f"{self.subject.surface.casefold()} {self.predicate.surface} {self.object.surface.casefold()}"

    @property
    def frame(self) -> str:
        return f"{self.subject.surface.casefold()}|{self.predicate.surface}|{self.object.surface.casefold()}"


ENTITIES = tuple(Entity(name) for name in (
    "Nora", "Leon", "Aron", "Noel", "Mara", "Aram", "Nadia", "Aidan", "Liam", "Ira",
))
# Both five-letter predicates preserve the 52-letter target side while still
# allowing the search to select a varied relation family.  ``sees`` remains a
# typed control and is not excluded from the grammar.
PREDICATES = (
    Predicate("stops", "control"),
    Predicate("spots", "observation"),
    Predicate("sees", "observation"),
)


def iter_clauses(subject: Entity | None = None, object_: Entity | None = None) -> Iterator[Clause]:
    """Generate one typed clause frontier; no chain paths are precomputed."""
    subjects = (subject,) if subject is not None else ENTITIES
    objects = (object_,) if object_ is not None else ENTITIES
    for subj in subjects:
        for predicate in PREDICATES:
            for obj in objects:
                yield Clause(subj, predicate, obj)


class ReverseResidualGrammar:
    """Character residual decoder for the right chain, grown tail-first."""

    def __init__(self) -> None:
        self.states_examined = 0

    def consume_clause(
        self,
        left_tape: str,
        object_constraint: Entity | None,
    ) -> list[tuple[Clause, dict[str, object]]]:
        frontier = list(iter_clauses(object_=object_constraint))
        trace: list[dict[str, object]] = []
        for offset, emitted in enumerate(left_tape):
            before = len(frontier)
            choices = sorted({candidate.tape[::-1][offset] for candidate in frontier if offset < len(candidate.tape)})
            frontier = [candidate for candidate in frontier
                        if offset < len(candidate.tape) and candidate.tape[::-1][offset] == emitted]
            self.states_examined += before
            trace.append({
                "offset": offset,
                "emitted": emitted,
                "expected_choices": choices,
                "frontier_before": before,
                "frontier_after": len(frontier),
                "matched": bool(frontier),
            })
            if not frontier:
                return []
        terminals = [candidate for candidate in frontier if len(candidate.tape) == len(left_tape)]
        return [(candidate, {
            "left_tape": left_tape,
            "trace": trace,
            "terminal_count": len(terminals),
            "object_constraint": object_constraint.surface if object_constraint else None,
        }) for candidate in terminals]


def relation_index_from_snapshot() -> tuple[dict[str, int], dict[str, object]]:
    payload = json.loads(SNAPSHOT.read_text())
    return dict(payload["relation_counts"]), payload


def freeze_global_snapshot() -> dict[str, object]:
    """Scan every tracked historical run, including failed-attempt strings."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "runs", "*.json"], cwd=ROOT, text=True
    ).splitlines()
    # The snapshot itself and this run do not exist at scan time; keeping the
    # exclusion explicit makes the frozen scope reproducible on regeneration.
    paths = [ROOT / rel for rel in tracked if Path(rel).name not in {SNAPSHOT.name, OUT.name}]
    relation_counts: dict[str, int] = {}
    manifest: list[dict[str, object]] = []
    total_bytes = 0
    for path in paths:
        data = path.read_bytes()
        total_bytes += len(data)
        manifest.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        text = data.decode("utf-8", errors="replace")
        for subject, predicate, object_ in RELATION_RE.findall(text):
            relation = f"{subject.casefold()} {predicate} {object_.casefold()}"
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = {
        "snapshot_id": "incumbent-672-global-novelty-snapshot-20260922",
        "snapshot_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "tracked runs/**/*.json; full JSON text, including rendered strings, rejected rows, and failed attempt_rendered fields",
        "excluded_from_scope": [str(SNAPSHOT.relative_to(ROOT)), str(OUT.relative_to(ROOT))],
        "file_count": len(paths),
        "manifest_entry_count": len(manifest),
        "total_bytes": total_bytes,
        "manifest_sha256": manifest_hash,
        "relation_pattern": RELATION_RE.pattern,
        "relation_counts": dict(sorted(relation_counts.items())),
        "descendants_scanned_for_global_novelty": True,
        "descendants_counted_as_568_parent_evidence": False,
        "note": "All historical outputs are scanned for relation reuse; descendants are not treated as valid lineage evidence for the 568 parent.",
    }
    SNAPSHOT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def audit_project(text: str) -> dict[str, object]:
    from llm_palindrome.validator import is_palindrome
    checked = independent_audit(text)
    checked["project_validator_exact"] = bool(is_palindrome(text))
    return checked


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent_row = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent_row["rendered"])
    base_audit = independent_audit(base)
    assert base_audit["normalized_letters"] == 568 and base_audit["sha256_forward"] == PARENT_SHA256
    assert base_audit["two_pointer_exact"]
    assert SNAPSHOT.exists(), "freeze the global novelty snapshot before searching"
    prior_relations, snapshot = relation_index_from_snapshot()
    left_raw = raw_after_letters(base, LEFT_CUT) + 1
    right_raw = raw_after_letters(base, RIGHT_CUT) + 1
    assert (left_raw, right_raw) == (62, 722)
    assert base[left_raw - 1] == "." and base[right_raw - 1] == "."

    reverse_grammar = ReverseResidualGrammar()
    attempts: list[dict[str, object]] = []
    accepted: tuple[list[Clause], list[Clause], list[dict[str, object]]] | None = None
    # Search left chain depth-first.  Each new clause is generated only after
    # the preceding object is known; its right counterpart is decoded online.
    def search(left_chain: list[Clause], right_reverse: list[Clause], traces: list[dict[str, object]]) -> None:
        nonlocal accepted
        if accepted is not None:
            return
        if len(left_chain) == CHAIN_LENGTH:
            right_chain = list(reversed(right_reverse))
            if all(right_chain[i].object.surface == right_chain[i + 1].subject.surface for i in range(CHAIN_LENGTH - 1)):
                relations = [clause.relation for clause in left_chain + right_chain]
                unique = len(set(relations)) == len(relations)
                novel = all(prior_relations.get(relation, 0) == 0 for relation in relations)
                predicates = {clause.predicate.surface for clause in left_chain}
                complete = all(clause.surface.endswith(".") and len(clause.tape) > 0 for clause in left_chain + right_chain)
                if unique and novel and len(predicates) >= 2 and complete:
                    accepted = (left_chain[:], right_chain, traces[:])
                else:
                    attempts.append({"status": "rejected_chain_gate", "left_chain": [c.frame for c in left_chain], "right_chain": [c.frame for c in right_chain], "unique_relations": unique, "novel_relations": novel, "predicate_families": sorted(predicates), "complete_clauses": complete})
            return
        subject = left_chain[-1].object if left_chain else None
        for left_clause in iter_clauses(subject=subject):
            if any(left_clause.surface == prior.surface for prior in left_chain):
                continue
            if left_clause.subject.surface == left_clause.object.surface:
                continue
            if left_clause.tape == left_clause.tape[::-1]:
                continue
            # Decode the counterpart from this clause's emitted characters;
            # object constraint joins it to the already discovered right tail.
            object_constraint = right_reverse[-1].subject if right_reverse else None
            decoded = reverse_grammar.consume_clause(left_clause.tape, object_constraint)
            if not decoded:
                attempts.append({"status": "rejected_residual", "depth": len(left_chain), "left_state": left_clause.frame, "left_tape": left_clause.tape, "object_constraint": object_constraint.surface if object_constraint else None, "cursor": {"left": LEFT_CUT + sum(len(c.tape) for c in left_chain), "right_reverse": RIGHT_CUT - 1 - sum(len(c.tape) for c in left_chain)}, "residual_trace": []})
                continue
            for right_clause, trace in decoded:
                if right_clause.subject.surface == right_clause.object.surface:
                    continue
                if right_clause.tape == right_clause.tape[::-1] or right_clause.surface in [c.surface for c in right_reverse]:
                    continue
                next_trace = traces + [{"depth": len(left_chain), "left": left_clause.frame, "right_discovered": right_clause.frame, "residual": trace}]
                search(left_chain + [left_clause], right_reverse + [right_clause], next_trace)
                if accepted is not None:
                    return

    search([], [], [])
    if accepted is None:
        raise AssertionError("no novel connected chain found; inspect attempts and residual cursor evidence")
    left_chain, right_chain, traces = accepted
    left_block = " ".join(clause.surface for clause in left_chain)
    right_block = " ".join(clause.surface for clause in right_chain)
    assert normalize(left_block) == normalize(right_block)[::-1]
    rendered = base[:left_raw] + " " + left_block + base[left_raw:right_raw] + " " + right_block + base[right_raw:]
    result = audit_project(rendered)
    assert result["normalized_letters"] > 648 and result["two_pointer_exact"] and result["sha_equal"] and result["project_validator_exact"]
    assert base[:left_raw] == rendered[:left_raw]
    assert base[right_raw:] == rendered[-len(base[right_raw:]):]
    inserted_relations = [clause.relation for clause in left_chain + right_chain]
    row = {
        "id": f"discourse-linked-reverse-chain-{result['normalized_letters']}",
        "working_status": "exact_candidate_novel_connected_four_event_chain",
        "rendered": rendered,
        "audit": result,
        "independent_audit": result,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": result["normalized_letters"] - 568,
        "seam": {"normalized_cuts": [LEFT_CUT, RIGHT_CUT], "raw_boundaries": [left_raw, right_raw], "complete_sentence_boundaries": True, "inserted_left_letters": len(normalize(left_block)), "inserted_right_letters": len(normalize(right_block))},
        "source_composition": {
            "parent_outside_seam_byte_for_byte": True,
            "prefix_sha256": hashlib.sha256(base[:left_raw].encode()).hexdigest(),
            "middle_sha256": hashlib.sha256(base[left_raw:right_raw].encode()).hexdigest(),
            "suffix_sha256": hashlib.sha256(base[right_raw:].encode()).hexdigest(),
            "left_inserted_surface": left_block,
            "right_inserted_surface": right_block,
        },
        "left_chain": [{"surface": c.surface, "frame": c.frame, "relation": c.relation} for c in left_chain],
        "right_chain_rendered_order": [{"surface": c.surface, "frame": c.frame, "relation": c.relation} for c in right_chain],
        "online_search": {"states_examined": reverse_grammar.states_examined, "attempt_count": len(attempts), "accepted_depth": CHAIN_LENGTH, "residual_traces": traces, "attempts": attempts[:256], "right_chain_discovered_tail_first": True, "left_subject_follows_previous_object": True, "right_subject_follows_previous_object": True},
        "novelty": {"snapshot_id": snapshot["snapshot_id"], "snapshot_commit": snapshot["snapshot_commit"], "manifest_sha256": snapshot["manifest_sha256"], "relations_checked": inserted_relations, "all_inserted_relations_absent": all(prior_relations.get(r, 0) == 0 for r in inserted_relations), "rejected_prior_edges": sorted(r for r in inserted_relations if prior_relations.get(r, 0)), "failed_attempts_scanned": True},
        "provenance": {"generator": "typed left chain + online reverse-character residual/right chain grammar", "candidate_discovered_online": True, "preauthored_pair_catalogue": False, "repeated_clauses": False, "self_palindromic_clauses": False, "punctuation_changes_letters": False, "human_certified": False, "reader_status": "pending human review; exactness and generator audits are not reader certification"},
        "global_gate": {"rendered_full_tape_exact": True, "new_relations_unique": True, "connected_four_event_scene": True, "human_certified": False, "status": "exact novel candidate; pending human review"},
    }
    return {
        "experiment_id": "incumbent-672-discourse-linked-reverse-chain-20260922",
        "method": "online typed left four-event chain with reverse-character residual/right chain grammar",
        "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        "seam": {"normalized_cuts": [LEFT_CUT, RIGHT_CUT], "raw_boundaries": [left_raw, right_raw], "fresh_complete_boundary": True},
        "novelty_snapshot": {"artifact": str(SNAPSHOT.relative_to(ROOT)), "snapshot_id": snapshot["snapshot_id"], "snapshot_commit": snapshot["snapshot_commit"], "file_count": snapshot["file_count"], "total_bytes": snapshot["total_bytes"], "manifest_sha256": snapshot["manifest_sha256"], "scope": snapshot["scope"], "descendants_scanned_for_global_novelty": snapshot["descendants_scanned_for_global_novelty"], "descendants_counted_as_568_parent_evidence": snapshot["descendants_counted_as_568_parent_evidence"]},
        "config": {"chain_length": CHAIN_LENGTH, "online_right_residual": True, "complete_boundary_seam": True, "reject_prior_edges": True, "reject_repeated_clauses": True, "reject_self_palindromic_clauses": True, "max_attempt_records": 256},
        "stats": {"independently_exact_children": 1, "longest_letters": result["normalized_letters"], "states_examined": reverse_grammar.states_examined, "attempted_paths": len(attempts), "accepted_paths": 1},
        "rows": [row],
    }


def main() -> None:
    if not SNAPSHOT.exists():
        freeze_global_snapshot()
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    row = payload["rows"][0]
    print(json.dumps({"artifact": str(OUT.relative_to(ROOT)), "letters": row["independent_audit"]["normalized_letters"], "sha256": row["independent_audit"]["sha256_forward"], "states_examined": payload["stats"]["states_examined"]}, sort_keys=True))


if __name__ == "__main__":
    main()
