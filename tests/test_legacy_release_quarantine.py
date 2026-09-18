import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "program,args",
    [
        ("paper/build_release.py", ["--release-id", "anything"]),
        ("paper/validate_release.py", ["--release-id", "revision-2026-09-10"]),
        ("paper/build_anonymous_review_release.py", []),
        ("paper/validate_anonymous_review_release.py", []),
        ("paper/prepare_dataverse_upload.py", []),
    ],
)
def test_no_legacy_release_path_can_claim_current_evidence(program, args):
    completed = subprocess.run(
        [sys.executable, program, *args], text=True, capture_output=True, check=False
    )
    assert completed.returncode != 0
    assert "disabled" in completed.stderr
