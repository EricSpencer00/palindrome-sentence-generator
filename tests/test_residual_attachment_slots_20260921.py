import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs/residual-attachment-slots-20260921.json"


def test_residual_attachment_controls_are_complete_and_nonexact():
    subprocess.run(
        ["python3", "experiments/residual_attachment_slots_20260921.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(RUN.read_text())
    assert payload["exact_count"] == 0
    assert payload["stats"]["longest_letters"] >= 89
    rendered = [row["rendered"] for row in payload["rendered_candidates"]]
    assert all("which the foreman" not in text for text in rendered)
    assert all("that the survey team" not in text for text in rendered)
    assert all(row["provenance"]["fresh_complete_prose"] for row in payload["rendered_candidates"])
