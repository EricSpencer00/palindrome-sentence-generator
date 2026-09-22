"""Record the exact but shortcut-rejected wide-center story replacement."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from llm_palindrome.admission import (
    has_distinct_content_words,
    has_repeated_nontrivial_unit,
    has_self_palindromic_proper_multiword_span,
    normalize_letters,
    tokenize,
)


PARENT = ROOT / "runs" / "incumbent-544-cross-boundary-seam-repair-20261002.json"
OUT = ROOT / "runs" / "incumbent-550-wide-center-story-20261002.json"
PARENT_SHA256 = "3040f0c4ac28002aa0edd7ce2fd920751b10e4a4f5430d82de3e46d09b3e7673"
EXPECTED_SHA256 = "f7846451d06b41025b216009bdc82e7d70a3d65d9b09ff454db6560c25cf2744"

OLD_WINDOW = (
    "Pat notes. Mara saw God. Sara, did I live? Nora, I saw desserts. "
    "Noel, was I stressed? Nadia delivers maps. Leon, I saw diaper. "
    "Repaid was I, Noel; spam's reviled, Aidan. Desserts I saw, Leon. "
    "Stressed was I, Aron. Evil I did, Aras. Dog was Aram. Seton, tap"
)
NEW_WINDOW = (
    "Now, Nora lived. Aidan saw Raila; Nadia saw Reeda. Aidan saw Namowa. "
    "Nora stops rats. Nora won Mara. Keep “S,” Kay. Yaks peek, Aram. "
    "Now, Aron, Star spots Aron. A woman was Nadia; a deer was Aidan. "
    "A liar was Nadia. “Devil Aron” won"
)


def proper_spans(text: str) -> list[dict[str, object]]:
    units = tokenize(text)
    spans = []
    for width in range(2, len(units) + 1):
        for start in range(len(units) - width + 1):
            end = start + width
            if start == 0 and end == len(units):
                continue
            tape = "".join(normalize_letters(unit) for unit in units[start:end])
            if tape and tape == tape[::-1]:
                spans.append({
                    "token_offsets": [start, end],
                    "tokens": list(units[start:end]),
                    "letters": len(tape),
                })
    return spans


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = parent_payload["rows"][0]
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    parent_rendered = str(parent["rendered"])
    assert parent_rendered.count(OLD_WINDOW) == 1

    rendered = parent_rendered.replace(OLD_WINDOW, NEW_WINDOW)
    result_audit = audit(rendered)
    assert result_audit["letters"] == 532
    assert result_audit["sha256_forward"] == EXPECTED_SHA256
    assert result_audit["two_pointer_exact"]
    assert result_audit["byte_pointer_exact"]
    assert result_audit["project_validator_exact"]

    units = tokenize(rendered)
    spans = proper_spans(rendered)
    assert len(spans) == 64
    assert has_self_palindromic_proper_multiword_span(units)
    assert has_repeated_nontrivial_unit(units)
    assert not has_distinct_content_words(units)

    row = {
        "id": "wide-center-story-532-rejected",
        "rendered": rendered,
        "audit": result_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_sha256": PARENT_SHA256,
        "source_498_artifact": "runs/overhang-growth-from-240-20261001.json",
        "source_498_sha256": "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032",
        "replacement": {
            "parent_normalized_offsets": [182, 368],
            "old_letters": 186,
            "new_letters": 168,
            "old_window": OLD_WINDOW,
            "new_window": NEW_WINDOW,
            "new_event_content": [
                "Nora lived",
                "Aidan saw three people",
                "Nora won Mara",
                "Kay keeps S",
                "Aram observes yaks",
            ],
        },
        "strict_admission": {
            "proper_palindromic_multiword_spans": spans,
            "proper_span_count": len(spans),
            "no_self_palindromic_proper_multiword_span": False,
            "no_repeated_nontrivial_unit": False,
            "distinct_content_words": False,
            "shortcut_clean": False,
            "human_certified": False,
            "status": "exact working-track child; rejected from reader promotion",
        },
        "next_operator": (
            "use the asymmetric smaps-owned window at normalized [261,294) "
            "and reject complementary internal token-boundary cuts online"
        ),
    }
    return {
        "experiment_id": "incumbent-550-wide-center-story-20261002",
        "method": "single authored wide-center story replacement with strict post-construction admission audit",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": 550,
            "sha256": PARENT_SHA256,
        },
        "stats": {
            "independently_exact_children": 1,
            "longest_letters": 532,
            "shortcut_clean_children": 0,
            "proper_span_count": len(spans),
        },
        "rows": [row],
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
