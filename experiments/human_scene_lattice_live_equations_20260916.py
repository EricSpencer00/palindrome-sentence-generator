"""Reproduce the human-authored scene-lattice live-equation probe.

The scene is authored as semantic slots first; the mirror debt is measured
while the slots are emitted.  This is deliberately a construction probe, not
an assertion that ordinary prose is already palindromic.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (
    "At first light, Mara carried the brass key across the flooded courtyard, "
    "unlocked the archive door, and waited while the rescued records dried."
)


def tape(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def audit(text: str) -> dict:
    t = tape(text)
    pointer = all(t[i] == t[-1 - i] for i in range(len(t) // 2))
    return {
        "letters": len(t),
        "two_pointer_exact": pointer,
        "sha256_normalized": hashlib.sha256(t.encode()).hexdigest(),
        "sha256_reversed": hashlib.sha256(t[::-1].encode()).hexdigest(),
        "hash_exact": hashlib.sha256(t.encode()).hexdigest()
        == hashlib.sha256(t[::-1].encode()).hexdigest(),
    }


def main() -> None:
    out = ROOT / "runs" / "human-scene-lattice-live-equations-20260916.json"
    payload = json.loads(out.read_text())
    payload["candidate"]["independent_validation"]["replay"] = audit(TEXT)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["candidate"]["independent_validation"], indent=2))


if __name__ == "__main__":
    main()
