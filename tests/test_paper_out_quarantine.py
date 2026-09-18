from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "out"


def test_all_legacy_paper_payload_folders_are_visibly_withdrawn():
    assert "quarantined legacy" in (OUT / "QUARANTINE.md").read_text().lower()

    folders = (
        "dataverse-upload",
        "evidence",
        "figures-qa",
        "final-prose",
        "good-draft-final",
        "naacl-controlled",
        "naacl-final",
        "naacl",
        "pre-final-prose",
        "pre-good-draft-abstract",
        "pre-pruning",
        "pruning",
        "revision-2026-09-10",
    )
    for folder in folders:
        marker = OUT / folder / "WITHDRAWN-DO-NOT-DISTRIBUTE.md"
        assert marker.is_file(), folder
        assert "do not distribute" in marker.read_text().lower(), folder


def test_legacy_reader_instructions_are_visibly_withdrawn():
    instructions = (
        ROOT
        / "runs"
        / "readability-length-study-2026-09-12"
        / "HUMAN-INSTRUCTIONS.md"
    ).read_text()
    assert instructions.startswith("# WITHDRAWN")
