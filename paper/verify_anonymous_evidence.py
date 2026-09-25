"""Verify the selected text and two constructive replays without Git or a model.

Run this file inside the extracted anonymous evidence archive with Python 3.10+.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from pathlib import PurePosixPath

from check_seam_invariant import exhaustive_algebra_audit
from replay_clause_search import replay


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
SENTENCE_RE = re.compile(r"[^.!?]*[.!?](?:[\"”’']*)|[^.!?]+$")


def letters(text: str) -> str:
    return re.sub("[^A-Za-z]", "", text).lower()


def raw_exact(text: str) -> bool:
    left, right = 0, len(text) - 1
    found = False
    while left <= right:
        while left <= right and not ("A" <= text[left] <= "Z" or "a" <= text[left] <= "z"):
            left += 1
        while left <= right and not ("A" <= text[right] <= "Z" or "a" <= text[right] <= "z"):
            right -= 1
        if left <= right:
            found = True
            if text[left].lower() != text[right].lower():
                return False
            left += 1
            right -= 1
    return found


def outside_in_report(text: str) -> dict[str, object]:
    tape = letters(text)
    left, right = 0, len(tape) - 1
    while left < right and tape[left] == tape[right]:
        left += 1
        right -= 1
    exact = left >= right
    return {
        "letters": len(tape),
        "exact": exact,
        "first_mismatch": None if exact else {
            "left_offset": left,
            "right_offset": right,
            "left": tape[left],
            "right": tape[right],
        },
    }


def structural_metrics(text: str) -> dict[str, object]:
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    trigrams = Counter(tuple(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2)))
    trigram_total = sum(trigrams.values())
    trigram_excess = sum(count - 1 for count in trigrams.values() if count > 1)
    sentences = [part.strip() for part in SENTENCE_RE.findall(text) if part.strip()]
    sentence_keys = [" ".join(sentence.split()).casefold() for sentence in sentences]
    sentence_counts = Counter(sentence_keys)
    return {
        "word_count": len(tokens),
        "unique_lowercase_word_count": len(set(tokens)),
        "repeated_trigram_excess_occurrences": trigram_excess,
        "trigram_occurrences": trigram_total,
        "repeated_trigram_rate": trigram_excess / trigram_total if trigram_total else 0.0,
        "sentence_count": len(sentences),
        "duplicate_sentence_count": sum(count - 1 for count in sentence_counts.values() if count > 1),
    }


def clause(frame: str, entities: dict[str, str], predicates: set[str]) -> tuple[str, str, str]:
    subject, predicate, obj = frame.split("|")
    if subject not in entities or obj not in entities or predicate not in predicates:
        raise AssertionError(f"Unknown clause frame: {frame}")
    return subject, predicate, obj


def verify_comparison(directory: Path, parent: str, saved_672: dict[str, object],
                      prior_relations: dict[str, int]) -> dict[str, object]:
    audit = json.loads((directory / "comparison-audit.json").read_text())
    with gzip.open(directory / "comparison-candidates.json.gz", "rt", encoding="utf-8") as stream:
        comparison = json.load(stream)
    arms = [comparison["arms"][name]["stats"] for name in ("online", "offline")]
    expected_digest = audit["candidate_set_digest"]
    assert comparison["candidate_sets_match"]
    assert arms[0]["candidate_set_digest"] == arms[1]["candidate_set_digest"] == expected_digest
    assert audit["independent_candidate_rows_checked"] == 38498
    assert audit["all_candidates_replayed_from_parent"]
    assert audit["all_candidates_pass_two_independent_exactness_checks"]
    assert audit["all_candidate_provenance_and_novelty_gates_recomputed"]
    assert audit["all_structural_metrics_recomputed"]

    grammar = comparison["grammar"]
    entities = {item.casefold(): item for item in grammar["entities"]}
    predicates = set(grammar["predicates"])
    base = letters(parent)
    assert len(base) == comparison["parent"]["letters"] == 568
    assert hashlib.sha256(base.encode("ascii")).hexdigest() == comparison["parent"]["sha256"]
    assert comparison["shared_protocol"]["seam_normalized_cuts"] == [48, 520]

    keys = set()
    key_digest = hashlib.sha256()
    lengths = []
    target_matches = []
    candidates = comparison["accepted_candidates"]
    for row in candidates:
        provenance = row["provenance"]
        assert provenance["parent_sha256"] == comparison["parent"]["sha256"]
        assert provenance["parent_id"] == comparison["parent"]["id"]
        assert provenance["seam_normalized_cuts"] == [48, 520]
        assert provenance["method_arms"] == ["online_character_residual", "offline_reverse_pair_index"]
        left_frames = provenance["left_chain"]
        right_reverse = provenance["right_reverse_pairing_order"]
        right_frames = provenance["right_chain_rendered_order"]
        assert len(left_frames) == len(right_reverse) == len(right_frames) == 4
        assert right_reverse == list(reversed(right_frames))

        left = [clause(frame, entities, predicates) for frame in left_frames]
        right = [clause(frame, entities, predicates) for frame in right_frames]
        assert all(left[i][2] == left[i + 1][0] for i in range(3))
        assert all(right[i][2] == right[i + 1][0] for i in range(3))
        all_frames = left_frames + right_frames
        assert len(set(all_frames)) == 8
        assert len({frame.split("|")[1] for frame in left_frames}) >= 2
        relation_strings = [frame.replace("|", " ") for frame in all_frames]
        assert all(prior_relations.get(relation, 0) == 0 for relation in relation_strings)

        left_text = " ".join(f"{entities[s]} {p} {entities[o]}." for s, p, o in left)
        right_text = " ".join(f"{entities[s]} {p} {entities[o]}." for s, p, o in right)
        left_tape, right_tape = letters(left_text), letters(right_text)
        assert left_tape == right_tape[::-1]
        expected_tape = base[:48] + left_tape + base[48:520] + right_tape + base[520:]
        rendered = row["rendered"]
        tape = letters(rendered)
        assert tape == expected_tape and tape == tape[::-1] and raw_exact(rendered)
        digest = hashlib.sha256(tape.encode("ascii")).hexdigest()
        saved_audit = row["audit"]
        assert saved_audit["outside_in_exact"] and saved_audit["normalizer_reverse_equal"]
        assert saved_audit["letters"] == len(tape) and saved_audit["sha256_forward"] == digest

        key = "||".join(left_frames + ["--"] + right_reverse)
        assert key not in keys
        keys.add(key)
        if len(keys) > 1:
            key_digest.update(b"\n")
        key_digest.update(key.encode("utf-8"))
        lengths.append(len(tape))
        if rendered == saved_672["surface"]:
            target_matches.append((row, digest))

    count = len(candidates)
    assert count == arms[0]["accepted_candidates"] == arms[1]["accepted_candidates"] == 38498
    assert key_digest.hexdigest() == expected_digest
    assert len(target_matches) == 1
    match, target_digest = target_matches[0]
    assert target_digest == saved_672["normalized_sha256"]
    assert match["provenance"]["parent_id"] == comparison["parent"]["id"]
    assert match["provenance"]["seam_normalized_cuts"] == [48, 520]
    summary = audit["length_summary"]
    assert [min(lengths), statistics.median(lengths), max(lengths)] == [
        summary["min"], summary["median"], summary["max"]
    ]
    return {
        "operator_equivalence_candidates": count,
        "candidate_set_digest": key_digest.hexdigest(),
        "exact_672_membership_count": len(target_matches),
        "all_candidates_replayed_and_exact": True,
    }


def verify_readability_calibration(directory: Path,
                                  selected: dict[str, dict[str, object]]) -> dict[str, object]:
    report = json.loads((directory / "readability-calibration.json").read_text())
    assert report["status"] == "programmatic_diagnostic_not_human_readability_result"
    assert report["method"]["training_file_count"] == 401
    assert report["method"]["heldout_file_count"] == 99
    assert report["method"]["shuffles_per_item"] == 32
    assert report["method"]["brown_tokenized_sentence_stream_sha256"]
    assert report["method"]["candidate_sentence_segmentation"]
    assert report["method"]["control_sentence_segmentation"] == "NLTK Brown sentence boundaries retained"
    assert len(report["candidates"]) == len(report["matched_heldout_prose_controls"]) == 11
    assert len(report["length_stratified_heldout_controls"]) == 108

    candidates = {row["id"]: row for row in report["candidates"]}
    controls = {row["matched_candidate_id"]: row
                for row in report["matched_heldout_prose_controls"]}
    assert set(candidates) == set(selected)
    for item_id, row in candidates.items():
        audit = selected[item_id]
        assert row["surface"] == audit["surface"]
        assert row["letters"] == audit["letters"]
        assert row["exactness"]["normalized_sha256"] == audit["normalized_sha256"]
        assert all(row["exactness"]["checks"].values())
        assert row["scorer_tokens"] == controls[item_id]["scorer_tokens"]
        assert "surface" not in controls[item_id]

    long_ids = [item_id for item_id, row in candidates.items()
                if 498 <= row["letters"] <= 752]
    assert len(long_ids) == 9
    output_scores = [candidates[item_id]["brown_order_gain_vs_own_shuffle"]
                     for item_id in long_ids]
    control_scores = [controls[item_id]["brown_order_gain_vs_own_shuffle"]
                      for item_id in long_ids]
    assert all(control > output
               for output, control in zip(output_scores, control_scores))
    assert round(statistics.fmean(output_scores), 3) == -0.109
    assert round(statistics.fmean(control_scores), 3) == 1.522

    curve = report["length_curve"]
    assert [row["target_scorer_tokens"] for row in curve] == [
        16, 32, 64, 128, 192, 256, 512, 1024, 2048
    ]
    assert all(row["control_count"] == row["positive_order_gain_controls"] == 12
               for row in curve)
    assert all(row["brown_order_gain_vs_own_shuffle"] > 0
               for row in report["length_stratified_heldout_controls"])
    return {
        "exact_candidates_checked": len(candidates),
        "long_output_control_pairs": len(long_ids),
        "all_long_output_controls_higher": True,
        "additional_length_controls": 108,
        "human_readability_evidence": "not supplied; metric is a local word-order diagnostic",
    }


def verify_lineage_tape_replay(selected: dict[str, dict[str, object]]) -> dict[str, object]:
    lineage = json.loads((Path(__file__).resolve().parent / "selected-results.json").read_text())[
        "lineage_tape_replay"
    ]
    assert "not candidate-generation procedures" in lineage["representation"]
    stages = lineage["stages"]
    assert [(stage["parent_id"], stage["child_id"]) for stage in stages] == [
        ("568-pinned", "640-event-chain"),
        ("640-event-chain", "686-shell-cycle"),
        ("686-shell-cycle", "736-mixed-cycle"),
        ("736-mixed-cycle", "752-center-path"),
    ]
    for stage in stages:
        parent = letters(selected[stage["parent_id"]]["surface"])
        child = letters(selected[stage["child_id"]]["surface"])
        assert len(parent) == stage["parent_letters"]
        assert len(child) == stage["child_letters"]
        assert hashlib.sha256(parent.encode("ascii")).hexdigest() == stage["parent_sha256"]
        assert hashlib.sha256(child.encode("ascii")).hexdigest() == stage["child_sha256"]
        replayed = parent
        edits = stage["edits"]
        assert edits
        for edit in sorted(edits, key=lambda row: row["parent_span_half_open"][0], reverse=True):
            start, end = edit["parent_span_half_open"]
            assert replayed[start:end] == edit["old"]
            replayed = replayed[:start] + edit["new"] + replayed[end:]
        assert replayed == child
    return {"stages_replayed": len(stages), "scope": "saved normalized tapes only"}


def verify(directory: Path) -> dict[str, object]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or
                "\\" in name or not relative.parts):
            raise ValueError("unsafe path in archive manifest")
        target = directory.joinpath(*relative.parts)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"File digest mismatch: {name}")
    data = json.loads((directory / "selected-results.json").read_text())
    rows = {row["id"]: row for row in data["results"]}
    if len(rows) != 11:
        raise AssertionError("Expected eleven selected examples")
    for row in rows.values():
        tape = letters(row["surface"])
        assert tape and tape == tape[::-1] and raw_exact(row["surface"])
        assert len(tape) == row["letters"]
        assert hashlib.sha256(tape.encode("ascii")).hexdigest() == row["normalized_sha256"]
        assert structural_metrics(row["surface"]) == row["metrics"]

    calibration = verify_readability_calibration(directory, rows)
    lineage_replay = verify_lineage_tape_replay(rows)

    lineage_ids = ["568-pinned", "640-event-chain", "686-shell-cycle", "736-mixed-cycle", "752-center-path"]
    lineage = [rows[row_id] for row_id in lineage_ids]
    lineage_lengths = [row["letters"] for row in lineage]
    assert lineage_lengths == [568, 640, 686, 736, 752]
    assert [lineage_lengths[i] - lineage_lengths[i - 1] for i in range(1, len(lineage_lengths))] == [72, 46, 50, 16]

    parent = rows["568-pinned"]["surface"]
    fixture = json.loads((directory / "seam-fixture.json").read_text())
    left, right = fixture["raw_cursors"]
    skipped = fixture["right_punctuation_skip"]
    assert not letters(parent[right:right + skipped])
    child = (parent[:left] + fixture["left_insert"] + parent[left:right]
             + fixture["right_insert"] + parent[right + skipped:])
    assert child == rows["630-god-dog"]["surface"] and raw_exact(child)

    one_sided = {
        "left_only": parent[:left] + fixture["left_insert"] + parent[left:],
        "right_only": (parent[:left] + parent[left:right]
                       + fixture["right_insert"] + parent[right + skipped:]),
    }
    expected_ablations = fixture["one_sided_ablation_expectations"]
    for name, surface in one_sided.items():
        report = outside_in_report(surface)
        assert not raw_exact(surface)
        assert report == expected_ablations[name]

    relations = json.loads((directory / "relation-index.json").read_text())
    searched = replay(parent, relations)
    assert searched["rendered"] == rows["672-reverse-chain"]["surface"]
    assert searched["states_examined"] == 9273
    assert searched["rejected_attempts"] == 35
    assert searched["accepted_paths"] == 1
    algebra = exhaustive_algebra_audit()
    assert algebra["all_checks_passed"]
    comparison = verify_comparison(directory, parent, rows["672-reverse-chain"], relations)
    return {
        "selected_exact_examples": len(rows),
        "lineage_lengths": lineage_lengths,
        "lineage_metrics_recomputed": True,
        "lineage_tape_replay": lineage_replay,
        "seam_replay_letters": len(letters(child)),
        "one_sided_seam_ablations": {
            name: outside_in_report(surface) for name, surface in one_sided.items()
        },
        "clause_search_letters": searched["letters"],
        "clause_search_frontier_examinations": searched["states_examined"],
        "clause_search_rejected_attempts": searched["rejected_attempts"],
        "algebra_checks": algebra["checks_performed"],
        "matched_operator_check": comparison,
        "readability_calibration": calibration,
        "human_readability_evidence": "not supplied; no human-study result is claimed",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
