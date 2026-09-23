#!/usr/bin/env python3
"""Audit an exact 568-letter seam rewrite rejected by the no-shortcut gate.

The paired left/right passage is authored before insertion and checked as one
character equation. It is retained as audit evidence only: every left token
maps to the character-reversal of one right token, including the self-
palindromic token ``sees``. The exact 588-letter tape is not an admissible
shortcut-free candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import is_palindrome, normalize


ROOT = Path(__file__).resolve().parents[1]
PARENT_REL = "runs/incumbent-560-outer-causal-scene-20261002.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
OUTPUT_REL = "runs/incumbent-568-linked-event-pair-growth-20260923.json"

LEFT_OLD = "Now, Noel, did I live?"
RIGHT_OLD = "Evil I did, Leon won."
LEFT_NEW = "Aidan stops Mara. Mara sees Ira."
RIGHT_NEW = "Ari sees Aram. Aram spots Nadia."
LEFT_START, LEFT_END = 148, 163
RIGHT_START, RIGHT_END = 405, 420


def regex_tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def ascii_byte_tape(text: str) -> bytes:
    """Separate normalizer: retain and lowercase only ASCII A-Z/a-z bytes."""
    out = bytearray()
    for byte in text.encode("utf-8"):
        if 65 <= byte <= 90:
            out.append(byte + 32)
        elif 97 <= byte <= 122:
            out.append(byte)
    return bytes(out)


def direct_mirror_mismatch(tape: bytes) -> int | None:
    for i in range(len(tape) // 2):
        if tape[i] != tape[len(tape) - 1 - i]:
            return i
    return None


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def node_independent_audit(rendered: str) -> dict[str, object]:
    source = r"""
const crypto = require('node:crypto');
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => input += chunk);
process.stdin.on('end', () => {
  const { rendered } = JSON.parse(input);
  const tape = rendered.toLowerCase().replace(/[^a-z]/g, '');
  let mismatch = null;
  for (let i = 0, j = tape.length - 1; i < j; i++, j--) {
    if (tape[i] !== tape[j]) { mismatch = { offset: i, left: tape[i], right: tape[j] }; break; }
  }
  const forward = crypto.createHash('sha256').update(tape, 'ascii').digest('hex');
  const reverse = crypto.createHash('sha256').update([...tape].reverse().join(''), 'ascii').digest('hex');
  process.stdout.write(JSON.stringify({ letters: tape.length, mismatch, forward, reverse, exact: tape.length > 0 && mismatch === null }));
});
"""
    result = subprocess.run(
        ["node", "-e", source],
        input=json.dumps({"rendered": rendered}),
        text=True,
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parent_record = json.loads((ROOT / PARENT_REL).read_text())
    parent_text = parent_record["rows"][0]["rendered"]
    parent_tape = regex_tape(parent_text)
    assert len(parent_tape) == 568
    assert sha256(parent_tape.encode("ascii")) == PARENT_SHA256
    assert is_palindrome(parent_text)
    assert direct_mirror_mismatch(ascii_byte_tape(parent_text)) is None

    assert parent_text.count(LEFT_OLD) == parent_text.count(RIGHT_OLD) == 1
    assert parent_tape[LEFT_START:LEFT_END] == regex_tape(LEFT_OLD) == "nownoeldidilive"
    assert parent_tape[RIGHT_START:RIGHT_END] == regex_tape(RIGHT_OLD) == "evilididleonwon"
    assert parent_tape[LEFT_START:LEFT_END][::-1] == parent_tape[RIGHT_START:RIGHT_END]

    left_tape, right_tape = regex_tape(LEFT_NEW), regex_tape(RIGHT_NEW)
    assert left_tape == "aidanstopsmaramaraseesira"
    assert right_tape == "ariseesaramaramspotsnadia"
    assert len(left_tape) == len(right_tape) == 25
    assert left_tape == right_tape[::-1]
    left_words = re.findall(r"[a-z]+", LEFT_NEW.lower())
    right_words = re.findall(r"[a-z]+", RIGHT_NEW.lower())
    token_reverse_pairs = [
        {"left": left, "right": right, "exact_reverse": left[::-1] == right, "self_palindromic_left": left == left[::-1]}
        for left, right in zip(left_words, reversed(right_words), strict=True)
    ]
    assert all(pair["exact_reverse"] for pair in token_reverse_pairs)
    assert any(pair["self_palindromic_left"] for pair in token_reverse_pairs)

    rendered = parent_text.replace(LEFT_OLD, LEFT_NEW, 1).replace(RIGHT_OLD, RIGHT_NEW, 1)
    child_tape = regex_tape(rendered)
    expected_tape = (
        parent_tape[:LEFT_START]
        + left_tape
        + parent_tape[LEFT_END:RIGHT_START]
        + right_tape
        + parent_tape[RIGHT_END:]
    )
    assert child_tape == expected_tape
    assert len(child_tape) == 588
    assert child_tape[:LEFT_START] == parent_tape[:LEFT_START]
    assert child_tape[LEFT_START + len(left_tape):LEFT_START + len(left_tape) + RIGHT_START - LEFT_END] == parent_tape[LEFT_END:RIGHT_START]
    assert child_tape[-(len(parent_tape) - RIGHT_END):] == parent_tape[RIGHT_END:]

    # Three mechanically distinct checks: project validator, ASCII byte walk,
    # and a separate Node.js normalizer/mirrored-pointer/hash implementation.
    project_tape = normalize(rendered)
    byte_tape = ascii_byte_tape(rendered)
    byte_mismatch = direct_mirror_mismatch(byte_tape)
    node_audit = node_independent_audit(rendered)
    assert project_tape == child_tape
    assert byte_tape.decode("ascii") == child_tape
    assert byte_mismatch is None
    assert is_palindrome(rendered)
    assert node_audit["letters"] == 588 and node_audit["exact"] is True
    assert node_audit["mismatch"] is None
    child_sha = sha256(byte_tape)
    assert child_sha == "957f2db34fde364866217a45c4bcc87c412a510b264490cc1a701c60c07651c0"
    assert node_audit["forward"] == node_audit["reverse"] == child_sha

    artifact = {
        "experiment_id": "incumbent-568-linked-event-pair-growth-20260923",
        "method": "audit of a mirrored clause-window rewrite that closes through word-by-word reverse pairs",
        "working_status": "audit_only_exact_rejected_wordwise_reverse_shortcut",
        "parent": {
            "artifact": PARENT_REL,
            "id": parent_record["working_length_incumbent"]["id"],
            "normalized_letter_length": len(parent_tape),
            "normalized_letter_sha256": PARENT_SHA256,
        },
        "seam": {
            "normalized_parent_offsets": [[LEFT_START, LEFT_END], [RIGHT_START, RIGHT_END]],
            "old_left": LEFT_OLD,
            "old_right": RIGHT_OLD,
            "old_left_tape": parent_tape[LEFT_START:LEFT_END],
            "old_right_tape": parent_tape[RIGHT_START:RIGHT_END],
            "old_equation_exact": True,
            "outside_replaced_windows_preserved": True,
        },
        "construction": {
            "left_rendered": LEFT_NEW,
            "left_tape": left_tape,
            "right_rendered": RIGHT_NEW,
            "right_tape": right_tape,
            "equation": f"{left_tape} = reverse({right_tape})",
            "equation_exact": left_tape == right_tape[::-1],
            "event_chain": [
                "Aidan stops Mara",
                "Mara sees Ira",
                "Ari sees Aram",
                "Aram spots Nadia",
            ],
            "grammar_note": "Each sentence is a finite transitive SVO clause; Mara and Aram respectively carry object-to-subject continuity across their two-clause local chains.",
            "new_letters_per_side": 25,
            "growth_over_parent": 20,
            "no_shortcut_gate": {
                "admitted": False,
                "reason": "Every left token is paired in reverse order with a right token whose letters are exactly reversed; 'sees' maps to itself and is self-palindromic.",
                "token_reverse_pairs": token_reverse_pairs,
            },
        },
        "novelty_preflight": {
            "status": "distinct_seam_and_event_pair_no_exact_archive_hits",
            "checked_phrases": [LEFT_NEW, RIGHT_NEW, "aidanstopsmaramaraseesira", "ariseesaramaramspotsnadia"],
            "scope": "retrospective exact-string and relation-phrase checks over repository runs, experiment scripts, docs, and data; not preregistered",
            "known_adjacent_construction": "incumbent-672-discourse-linked-reverse-chain-20260922 uses complete-boundary seam [48,520], not these offsets",
        },
        "candidate": {
            "rendered": rendered,
            "normalized_letter_length": len(child_tape),
            "normalized_letter_sha256": child_sha,
            "rendered_surface_length": len(rendered),
            "audit": {
                "project_validator_exact": is_palindrome(rendered),
                "project_normalizer_agrees": project_tape == child_tape,
                "ascii_byte_normalizer_agrees": byte_tape.decode("ascii") == child_tape,
                "direct_mirrored_byte_walk_exact": byte_mismatch is None,
                "first_mismatch_offset": byte_mismatch,
                "node_independent_audit": node_audit,
                "sha256_forward": sha256(byte_tape),
                "sha256_reverse": sha256(byte_tape[::-1]),
            },
            "provenance": {
                "parent_artifact": PARENT_REL,
                "parent_sha256": PARENT_SHA256,
                "transformation": "replace the unique normalized windows [148,163] and [405,420]; preserve every outside-parent letter and all surface outside the two unique source phrases",
            },
            "readability": {
                "human_certified": False,
                "claim": "Not admitted as a shortcut-free candidate. The inherited passage remains rough; no reader study was run.",
            },
        },
        "post_generation_followup_audit": {
            "candidate_length": 604,
            "candidate_sha256": "93b1728473422af77283d68d8277d3d4fe394cdef0e08245e6939bec4226edd6",
            "exact": True,
            "decision": "do_not_promote_or_count_as_a_new_method",
            "reason": "The [64,91]/[497,524] shell in this 588 child maps to the already explored 568 [64,91]/[477,504] geometry and repeats the 568 event-lattice/596 shell-repair operator; its six-clause blocks are also tokenwise reverse pairs.",
            "prior_artifacts": [
                "experiments/incumbent_568_repeated_shell_event_lattice_20261002.py",
                "experiments/incumbent_596_repeated_shell_repair_20261002.py",
            ],
        },
        "next_action": "Switch to an asymmetric character-level construction whose left/right word segmentations do not pair tokens by reversal; retain this 588 exact tape only as a rejected shortcut audit.",
    }
    output_path = ROOT / OUTPUT_REL
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "artifact": OUTPUT_REL,
        "letters": len(child_tape),
        "sha256": child_sha,
        "exact": node_audit["exact"] and byte_mismatch is None and is_palindrome(rendered),
        "rendered": rendered,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
