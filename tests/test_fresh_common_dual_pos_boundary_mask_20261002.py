import hashlib
import json
from pathlib import Path

from experiments.fresh_common_dual_pos_boundary_mask_20261002 import (
    TEMPLATES,
    boundary_mask_ok,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "fresh-common-dual-pos-boundary-mask-20261002.json"


def test_template_product_is_fresh_and_fully_exhausted() -> None:
    payload = json.loads(ARTIFACT.read_text())

    assert len(TEMPLATES) == 22
    assert payload["grammar"]["ordered_template_pairs"] == len(TEMPLATES) ** 2
    assert payload["result"]["states"] == 507871
    assert not payload["result"]["state_cap_reached"]
    assert payload["result"]["closures"] == 0
    assert payload["novelty_preflight"]["inherited_498_endpoint"] is False
    assert payload["candidate"] is None


def test_complementary_boundary_mask_rejects_the_diagnostic_shortcut() -> None:
    assert not boundary_mask_ok(("stops", "live"), ("evil", "spots"))
    assert boundary_mask_ok(("dog",), ("god",))


def test_committed_generator_digest_and_next_operator_are_pinned() -> None:
    payload = json.loads(ARTIFACT.read_text())
    source = ROOT / "experiments" / "fresh_common_dual_pos_boundary_mask_20261002.py"

    assert hashlib.sha256(source.read_bytes()).hexdigest() == payload["provenance"]["committed_generator_sha256"]
    assert payload["incumbent_specific_obstruction"]["next_operator"].startswith(
        "compile the same lexical domains into bidirectional character tries"
    )
