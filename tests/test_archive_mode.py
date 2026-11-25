"""PALINDROME_ARCHIVE=1 revives v1-v3 for local replay; unset keeps every gate."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import json
from fastapi.testclient import TestClient
from server import app as appmod, archive, v3

c = TestClient(appmod.app)
out = {
    "enabled": archive.ENABLED,
    "retired": sorted(appmod.RETIRED_OUTPUT_PATHS),
    "v2_word": c.get("/api/v2/paragraph?mode=word&sentences=2").status_code,
    "v2_health_ok": c.get("/api/v2/health").json()["ok"],
    "v3_palindrome": c.get("/api/v3/palindrome?seed=1").status_code,
    "v3_bank": len(v3._load_bank()),
    "v4_generate": c.get("/api/v4/generate").status_code,
}
print(json.dumps(out))
"""


def _probe(**env) -> dict:
    full = {k: v for k, v in os.environ.items()
            if k not in ("PALINDROME_ARCHIVE", "PALINDROME_ENGINE")}
    full.update(PALINDROME_NO_WARM="1", **env)
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=full,
                          capture_output=True, text=True, check=True, timeout=300)
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_archive_is_off_by_default() -> None:
    got = _probe()
    assert got["enabled"] is False
    assert got["retired"] == sorted({
        "/api/generate", "/api/v2/paragraph", "/api/v2/generate",
        "/api/v3/composition", "/api/v3/palindrome", "/api/v3/refrain",
        "/api/v4/generate"})
    assert got["v2_word"] == 503
    assert got["v2_health_ok"] is False
    assert got["v3_palindrome"] == 503
    assert got["v4_generate"] == 503


def test_archive_flag_revives_v1_to_v3() -> None:
    got = _probe(PALINDROME_ARCHIVE="1")
    assert got["enabled"] is True
    assert got["retired"] == []
    assert got["v2_word"] == 200
    assert got["v2_health_ok"] is True
    assert got["v3_palindrome"] == 200
    assert got["v3_bank"] > 0
    # v4 never generated live; its own route keeps the gate.
    assert got["v4_generate"] == 503
