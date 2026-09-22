"""Replace most of the 498-letter incumbent with a boundary-shifting shell.

This experiment reopens the first complete-word seam whose retained center is
the 240-letter palindrome at parent offsets [129, 369).  The new shell is not
rendered as a list of independently closed clause pairs: ``A tub? He`` on the
left becomes ``Eh, but a`` on the right, changing sentence and constituent
boundaries.  A live ``now`` obligation also crosses the retained center and is
consumed as ``won`` in ``Leon won``.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.validator import is_palindrome, normalize


PARENT = ROOT / "runs" / "overhang-growth-from-240-20261001.json"
OUT = ROOT / "runs" / "incumbent-498-deep-clause-transducer-20261002.json"
PARENT_SHA256 = "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032"
EXPECTED_SHA256 = "4df622f18dc04d737f0c3644e52a03e7d391e1baaf0f6d8c179b1f3eb38f6a0d"
LEFT_CURSOR = 129
RIGHT_CURSOR = 369

LEFT_SHELL = (
    "Nadia delivers maps. Nora stops rats. A tub? He maps Leon. "
    "Nora delivers maps. Mara stops rats. A tub? He maps Aron. "
    "Aidan delivers maps. Mara stops rats. A tub? He maps Nora. "
    "Deliver no evil. Now,"
)

# The opening ``won`` is deliberately owned by this shell.  It follows the
# retained final token ``Leon`` and turns that boundary into ``Leon won.``
RIGHT_SHELL = (
    "won. Live on, reviled. Aron, spam. Eh, but a star spots Aram. "
    "Spam's reviled, Nadia. Nora, spam. Eh, but a star spots Aram. "
    "Spam's reviled, Aron. Noel, spam. Eh, but a star spots Aron. "
    "Spam's reviled, Aidan."
)

CENTER_RENDERED = (
    "Noel, did I live? Nora, was I evil? Noel, did I draw Mara? Was I God? "
    "Sara, did I live? Nora, I saw desserts. Noel, was I stressed? "
    "Nora, I saw deliver Noel. I saw diaper. Repaid was I, Leon. "
    "Reviled was I, Aron. Desserts I saw, Leon. Stressed was I, Aron. "
    "Evil I did, Aras. Dog I saw, Aram. Ward I did, Leon. "
    "Live I saw, Aron. Evil I did, Leon"
)


def independent_tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = independent_tape(text)
    mismatch = next(
        (
            {"offset": i, "left": tape[i], "right": tape[-1 - i]}
            for i in range(len(tape) // 2)
            if tape[i] != tape[-1 - i]
        ),
        None,
    )
    byte_mismatch = next(
        (
            i for i, (left, right) in enumerate(zip(tape.encode(), tape[::-1].encode()))
            if left != right
        ),
        None,
    )
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "independent_normalizer_agrees": tape == normalize(text),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "byte_pointer_exact": bool(tape) and byte_mismatch is None,
        "project_validator_exact": bool(is_palindrome(text)),
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def load_parent() -> tuple[str, str]:
    payload = json.loads(PARENT.read_text())
    row = max(payload["rows"], key=lambda item: item["audit"]["letters"])
    rendered = row["rendered"]
    tape = independent_tape(rendered)
    assert len(tape) == 498
    assert tape == tape[::-1]
    assert hashlib.sha256(tape.encode()).hexdigest() == PARENT_SHA256
    return rendered, tape


def build_payload() -> dict[str, object]:
    parent_rendered, parent_tape = load_parent()
    center_tape = parent_tape[LEFT_CURSOR:RIGHT_CURSOR]
    assert len(center_tape) == 240
    assert center_tape == center_tape[::-1]
    assert independent_tape(CENTER_RENDERED) == center_tape

    left_tape = independent_tape(LEFT_SHELL)
    right_tape = independent_tape(RIGHT_SHELL)
    assert len(left_tape) == 147
    assert right_tape == left_tape[::-1]

    rendered = f"{LEFT_SHELL} {CENTER_RENDERED} {RIGHT_SHELL}"
    result_audit = audit(rendered)
    assert result_audit["letters"] == 534
    assert result_audit["sha256_forward"] == EXPECTED_SHA256
    assert all(
        result_audit[key]
        for key in (
            "independent_normalizer_agrees",
            "two_pointer_exact",
            "byte_pointer_exact",
            "project_validator_exact",
            "sha_equal",
        )
    )

    # The key macro changes the grammatical boundary layout.  It is exact as
    # tape, but its two left sentences are not independently paired with the
    # single right continuation.
    left_bridge = "A tub? He"
    right_bridge = "Eh, but a"
    assert independent_tape(left_bridge)[::-1] == independent_tape(right_bridge)

    return {
        "experiment_id": "incumbent-498-deep-clause-transducer-20261002",
        "method": (
            "reopen the first 240-letter complete-word center of the verified "
            "498 parent and solve a boundary-shifting finite-clause shell"
        ),
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "sha256": PARENT_SHA256,
            "letters": 498,
            "rendered": parent_rendered,
            "independent_exact": True,
        },
        "search_state": {
            "left_cursor": LEFT_CURSOR,
            "right_cursor": RIGHT_CURSOR,
            "retained_parent_letters": len(center_tape),
            "removed_inherited_outer_letters": 498 - len(center_tape),
            "left_owner_before_center": "R",
            "residual_before_center": "won",
            "right_boundary_consumption": "Leon|won",
            "residual_after_shell": "",
        },
        "transducer": {
            "left_bridge": left_bridge,
            "left_bridge_tape": independent_tape(left_bridge),
            "right_bridge": right_bridge,
            "right_bridge_tape": independent_tape(right_bridge),
            "bridge_exact": True,
            "boundary_shift": (
                "left question plus new subject becomes a right interjection "
                "and conjunction inside a finite clause"
            ),
            "left_shell_letters": len(left_tape),
            "right_shell_letters": len(right_tape),
            "shell_reverse_exact": True,
        },
        "rows": [
            {
                "id": "depth129-finite-clause-transducer",
                "rendered": rendered,
                "audit": result_audit,
                "parent_artifact": str(PARENT.relative_to(ROOT)),
                "parent_sha256": PARENT_SHA256,
                "growth_over_parent": result_audit["letters"] - 498,
                "removed_inherited_outer_letters": 258,
                "retained_parent_letters": 240,
                "new_shell_letters_per_side": 147,
                "new_event_content": [
                    "three deliveries",
                    "three stopped-rat events",
                    "three mapping events",
                    "Leon wins",
                ],
                "provenance": {
                    "center_copied_from_parent_offsets": [LEFT_CURSOR, RIGHT_CURSOR],
                    "new_boundary_shifting_topology": True,
                    "finished_parent_tape_reversal": False,
                    "posthoc_character_repair": False,
                    "human_certified": False,
                },
                "worst_seam": {
                    "text": "the retained 240-letter Noel/Nora center",
                    "diagnosis": (
                        "exact and substantially smaller than before, but still "
                        "formulaic and often semantically discontinuous"
                    ),
                    "next_repair": (
                        "reopen a symmetric phrase span inside the 240-letter "
                        "center and require a finite event transition"
                    ),
                },
            }
        ],
        "stats": {
            "independently_exact_children": 1,
            "children_over_530": 1,
            "longest_letters": 534,
            "maximum_inherited_outer_letters_replaced": 258,
        },
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
