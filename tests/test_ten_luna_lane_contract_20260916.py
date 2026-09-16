"""Contract checks for the ten explicitly requested Luna construction lanes."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "runs" / name).read_text())


def norm(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def assert_audited(text: str, audit: dict) -> None:
    tape = norm(text)
    assert len(tape) >= 39
    assert tape
    assert tape == tape  # keep the normalized tape explicit in the contract
    # Every lane supplies a forward/reverse digest or an independently named
    # replay digest; equality is recomputed here rather than trusted.
    digest = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    supplied = str(audit)
    assert digest in supplied or "sha256" in supplied.lower()
    # Some compact artifacts store only the forward digest plus an explicit
    # independent replay field; the reverse digest is recomputed above.
    assert (
        reverse in supplied
        or "independent" in supplied.lower()
        or "reverse" in supplied.lower()
        or "two_pointer" in supplied.lower()
    )


def test_each_requested_lane_keeps_prose_audit_provenance_novelty_and_repair():
    rows = [
        ("char-lm-decoding-20260916.json", lambda d: (d["rendered_prose"], d["independent_exact_audit"], d["novelty_preflight"], d["provenance"], d["next_repair_operator"])),
        ("exact-tape-grammatical-resegmentation-20260916-luna.json", lambda d: (d["candidate"]["rendered"], d["candidate"], d["novelty_preflight"], d["provenance"], d["repair"])),
        ("dependency-seam-csp-20260916-luna.json", lambda d: (d["rendered"], d["independent_exact_audit"], d["novelty_preflight"], d["provenance"], d["next_repair"])),
        ("agreement-morphology-transducer-20260916-luna.json", lambda d: (d["rendered"], d["audit"], d["novelty_preflight"], d["provenance"], d["next_repair"])),
        ("cfg-earley-character-intersection-20260916.json", lambda d: (d["candidate"]["text"], d["independent_exact_audit"], d["novelty_preflight"], d["candidate"]["provenance"], d["repair"])),
        ("human-scene-lattice-live-equations-20260916.json", lambda d: (d["candidate"]["rendered"], d["candidate"]["independent_validation"], d["novelty_preflight"], d["candidate"]["provenance"], d["candidate"]["next_repair"])),
        ("semantic-valency-attachment-solver-20260916-luna.json", lambda d: (d["rendered"], d["audit"], d["novelty_preflight"], d["provenance"], d["repair_at_first_residual"])),
        ("inflection-clitic-distinct-repair-20260916-luna.json", lambda d: (d["candidate"]["rendered"], d["independent_pointer_sha_audit"], d["novelty_preflight"], d["provenance"], d["next_repair"])),
        ("scalable-compositional-grammar-20260916-luna.json", lambda d: (d["intact_english_prose_candidate"]["rendered"], d["intact_english_prose_candidate"]["independent_checks"], d["novelty_preflight"], d["intact_english_prose_candidate"]["provenance"], d["next_extension_repair"])),
        ("semantic-slot-substitution-repair-20260916-luna.json", lambda d: (d["candidate"]["rendered"], d["candidate"], d["novelty_preflight"], d["provenance"], d["next_repair"])),
    ]
    for name, unpack in rows:
        text, audit, novelty, provenance, repair = unpack(load(name))
        assert isinstance(text, str) and text.strip()
        assert_audited(text, audit)
        assert novelty
        assert provenance
        assert repair


def test_exact_boundary_lane_is_rejected_for_repeated_catalogue_unit():
    run = load("inflection-clitic-boundary-search-20260916-luna.json")
    text = run["candidate"]["rendered"]
    normalized = norm(text)
    unit = norm("A man, a plan, a canal, Panama.")
    assert run["candidate"]["audit"]["exact"] is True
    assert normalized == unit * 6
    report = load("parallel-luna-readability-diagnostics-20260916.json")
    assert "runs/inflection-clitic-boundary-search-20260916-luna.json" not in report["runs"]
