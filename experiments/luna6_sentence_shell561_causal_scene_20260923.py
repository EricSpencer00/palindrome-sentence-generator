"""One bounded sentence-shell splice on the pinned 568-letter working tape."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-sentence-shell561-causal-scene-20260923.json"
EXPECTED_PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_SPAN = (7, 232)
RIGHT_CUT = 561
LEFT_SCENE = (
    "At dawn, Rina found a brass key beneath the collapsed ferry stairs. She carried it to "
    "the station keeper, who recognized the mark and warned her that the river had swallowed "
    "the old footbridge. By noon, Rina led two neighbors along the ridge, where a stranded "
    "archivist had hidden a ledger inside a dry stone well. They raised the book together and "
    "sent a runner to alert the village before the rain returned."
)
RIGHT_EVENT = (
    "At dusk, the council heard Rina's report and ordered the guards to leave the eastern "
    "gate open for travelers"
)


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalpha())


def raw_index_for_letter(text: str, offset: int) -> int:
    count = 0
    for i, c in enumerate(text):
        if c.isalpha():
            if count == offset:
                return i
            count += 1
    if count == offset:
        return len(text)
    raise IndexError(offset)


def outside_in(text: str) -> dict:
    tape = letters(text)
    mismatches = []
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i
        if tape[i] != tape[j]:
            mismatches.append({"left": i, "right": j, "left_char": tape[i], "right_char": tape[j]})
    return {
        "letters": len(tape),
        "exact": not mismatches,
        "pairs_checked": len(tape) // 2,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "first_12_mismatches": mismatches[:12],
    }


def word_audit(left: str, right: str) -> dict:
    left_words = re.findall(r"[a-z]+", left.lower())
    right_words = re.findall(r"[a-z]+", right.lower())
    reverse_pairs = sorted({(a, b) for a in left_words for b in right_words if a == b[::-1] and a != a[::-1]})
    repeated_ngrams = {}
    for n in (2, 3):
        lset = {tuple(left_words[i : i + n]) for i in range(len(left_words) - n + 1)}
        rset = {tuple(right_words[i : i + n]) for i in range(len(right_words) - n + 1)}
        repeated_ngrams[str(n)] = [list(x) for x in sorted(lset & rset)]
    return {
        "left_tokens": left_words,
        "right_tokens": right_words,
        "whole_token_reversal_pairs": [list(x) for x in reverse_pairs],
        "self_palindromic_words": sorted({w for w in left_words + right_words if len(w) > 1 and w == w[::-1]}),
        "cross_event_repeated_ngrams": repeated_ngrams,
        "shared_content_tokens": sorted(set(left_words) & set(right_words)),
    }


def main() -> None:
    from llm_palindrome.validator import is_palindrome as project_is_palindrome

    data = json.loads(PARENT_PATH.read_text())
    parent_rendered = next(row["rendered"] for row in data["rows"] if row["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent_rendered)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    assert len(parent_tape) == 568 and parent_sha == EXPECTED_PARENT_SHA256
    assert outside_in(parent_rendered)["exact"] and project_is_palindrome(parent_rendered)

    left_tape, right_tape = letters(LEFT_SCENE), letters(RIGHT_EVENT)
    assert len(left_tape) > LEFT_SPAN[1] - LEFT_SPAN[0]
    assert len(right_tape) > 0

    # Build the literal rendered splice. The left shell ends at a sentence
    # boundary; the right event completes the retained `...flow now, Noel.`
    # tail as `...flow. At dusk ... travelers now, Noel.`
    raw_left_start = raw_index_for_letter(parent_rendered, LEFT_SPAN[0])
    raw_left_end = raw_index_for_letter(parent_rendered, LEFT_SPAN[1])
    raw_right_cut = raw_index_for_letter(parent_rendered, RIGHT_CUT)
    assert raw_left_start < raw_left_end < raw_right_cut
    rendered = (
        parent_rendered[:raw_left_start]
        + LEFT_SCENE
        + ". "
        + parent_rendered[raw_left_end:raw_right_cut]
        + ". "
        + RIGHT_EVENT
        + " "
        + parent_rendered[raw_right_cut:]
    )
    tape = letters(rendered)
    expected_layout = parent_tape[:7] + left_tape + parent_tape[232:561] + letters(". ") + right_tape + parent_tape[561:]
    # Punctuation contributes no letters; this asserts the exact retained
    # parent lineage rather than relying on the rendered surface boundaries.
    assert tape == parent_tape[:7] + left_tape + parent_tape[232:561] + right_tape + parent_tape[561:]
    audit = outside_in(rendered)
    forward_sha = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(rendered)

    right_insert_start = 7 + len(left_tape) + (561 - 232)
    right_insert_end = right_insert_start + len(right_tape)
    owner_ranges = [
        (0, 7, "retained_parent_prefix"),
        (7, 7 + len(left_tape), "left_scene"),
        (7 + len(left_tape), right_insert_start, "retained_parent_middle"),
        (right_insert_start, right_insert_end, "right_event"),
        (right_insert_end, len(tape), "retained_parent_tail"),
    ]

    def owner(pos: int) -> str:
        for start, end, label in owner_ranges:
            if start <= pos < end:
                return label
        return "unknown"

    mismatch_owners = []
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i
        if tape[i] != tape[j]:
            mismatch_owners.append({"left_owner": owner(i), "right_owner": owner(j)})
    contradicted_unchanged_pairs = [
        x for x in mismatch_owners if x["left_owner"].startswith("retained_parent") or x["right_owner"].startswith("retained_parent")
    ]
    equation_cursor = 0
    required_left = right_tape[::-1]
    while equation_cursor < min(len(left_tape), len(required_left)) and left_tape[equation_cursor] == required_left[equation_cursor]:
        equation_cursor += 1
    local_mismatch = None if equation_cursor == len(required_left) else {
        "cursor": equation_cursor,
        "left": left_tape[equation_cursor],
        "required": required_left[equation_cursor],
    }
    assert not audit["exact"] and not project_exact

    result = {
        "experiment_id": "luna6-sentence-shell561-causal-scene-20260923",
        "status": "full_child_rendered_nonexact_working_diagnostic",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256_normalized": parent_sha,
            "independent_outside_in_exact": True,
            "project_validator_exact": True,
        },
        "novelty_preflight": {
            "geometry": {"replacement_parent_span": list(LEFT_SPAN), "opposing_insert_parent_cut": RIGHT_CUT},
            "exact_signature_found_in_registry_or_history": False,
            "basis": "Exact signature absent from novelty registry, goal history, and experiment/run search; sentence-bounded shell differs from prior 91/232 cut-477 diagnostic.",
        },
        "operator": {
            "name": "sentence-shell rebasing with a sentence-tail event insertion",
            "source_removed": parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]],
            "source_removed_letters": LEFT_SPAN[1] - LEFT_SPAN[0],
            "new_left_scene_letters": len(left_tape),
            "right_event_letters": len(right_tape),
            "right_event_parent_cut": RIGHT_CUT,
            "live_equation_required_left_prefix": required_left,
            "live_equation_left_prefix": left_tape[: len(right_tape)],
            "live_equation_matched_letters": equation_cursor,
            "live_equation_target_letters": len(right_tape),
            "live_equation_first_mismatch": local_mismatch,
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(tape),
            "growth_over_parent": len(tape) - len(parent_tape),
            "normalized_sha256": forward_sha,
            "reverse_sha256": reverse_sha,
            "hashes_equal": forward_sha == reverse_sha,
            "independent_outside_in": audit,
            "project_validator_exact": project_exact,
            "unchanged_parent_mismatch_audit": {
                "retained_mismatch_pairs": len(contradicted_unchanged_pairs),
                "first_12_pair_owners": mismatch_owners[:12],
                "source_coverage": {
                    "retained_prefix_parent_span": [0, 7],
                    "retained_middle_parent_span": [232, 561],
                    "retained_tail_parent_span": [561, 568],
                },
            },
        },
        "authored_scenes": {"left": LEFT_SCENE, "right": RIGHT_EVENT},
        "boundary_and_reuse_audit": word_audit(LEFT_SCENE, RIGHT_EVENT),
        "reader_status": "not_run; candidate is inexact and no readability claim is made",
        "next_operator": "Move the real seam to a clause boundary that changes which local words own the first reflected characters; do not rephrase either scene or reuse the cut-561 event. Preserve this tape, then preflight a different parent cut before authoring again.",
        "provenance": {
            "lexical_source": "freshly authored causal scene and council response for this single seam probe",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "full_rendered_candidate_preserved": True,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "status": result["status"],
        "letters": len(tape),
        "growth": len(tape) - len(parent_tape),
        "local_cursor": f"{equation_cursor}/{len(right_tape)}",
        "first_mismatch": audit["first_mismatch"],
        "project_exact": project_exact,
        "sha256": forward_sha,
        "retained_mismatch_pairs": len(contradicted_unchanged_pairs),
    }, indent=2))


if __name__ == "__main__":
    main()
