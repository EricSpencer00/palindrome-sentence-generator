import json
from pathlib import Path
from experiments.grapheme_chunk_finite_state_constructor_20260916 import audit, chunks, finite_state_clauses, two_pointer

ROOT = Path(__file__).parents[1]

def test_typed_chunks_and_fsm_emit_complete_clauses():
    rows = finite_state_clauses(4)
    assert rows and all(len(r["words"]) == 5 for r in rows)
    # Content words carry typed multi-character onset/rime chunks; one-letter
    # determiners are allowed as ordinary function-word terminals.
    assert all(any(len(part) > 1 for part in item)
               for r in rows for item, word in zip(r["chunks"], r["words"])
               if len(word) > 1)
    assert rows[0]["states"][:2] == ["START", "det"]

def test_independent_audit_and_residual_cross_boundaries():
    assert audit("A baker maps a gate.")["sha256_forward"] != audit("A baker maps a gate.")["sha256_reverse"]
    residual = two_pointer("the baker", "a quiet pilot")
    assert "residual_left" in residual and not residual["exact"]

def test_run_artifact_records_provenance_and_repair():
    path = ROOT / "runs/grapheme-chunk-finite-state-constructor-20260916.json"
    if path.exists():
        data = json.loads(path.read_text())
        assert data["provenance"]["catalogue_lookup"] is False
        assert data["first_residual_repair"]
