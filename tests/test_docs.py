"""The README must describe the system that exists.

Its structure block listed five modules when there were sixteen, so a reader
following it landed on `search.py` and never found `themes.py`, `respace.py` or
`mining.py` — the ones that produce what the endpoint actually serves. Worse,
the Paragraphs section described the WORD-ORDER pipeline as the answer long
after the letter-level one had replaced it.

Prose drifts and no test can catch that. What a test can catch is a structure
block naming files that do not exist, or a module that ships with no mention —
which is how both defects started.
"""
import re
from pathlib import Path

import pytest

README = Path("README.md")
BLOCK_START = "## Project structure"
BLOCK_END = "## Known limits"


@pytest.fixture(scope="module")
def block():
    text = README.read_text()
    return text[text.index(BLOCK_START):text.index(BLOCK_END)]


class TestStructureBlock:
    def test_every_file_it_names_exists(self, block):
        named = re.findall(r"\s{2,}([a-z_0-9]+\.(?:py|json|txt))\s", block)
        assert named, "structure block names no files"
        missing = []
        for name in named:
            found = [p for p in Path(".").rglob(name)
                     if "venv" not in str(p) and "node_modules" not in str(p)]
            if not found:
                missing.append(name)
        assert not missing, missing

    def test_every_shipped_module_is_named(self, block):
        """A module nobody can find is a module nobody will maintain."""
        actual = {p.name for p in Path("llm_palindrome").glob("*.py")}
        actual -= {"__init__.py"}
        named = set(re.findall(r"([a-z_0-9]+\.py)", block))
        assert not (actual - named), sorted(actual - named)

    def test_every_shipped_data_file_is_named(self, block):
        actual = {p.name for p in Path("data").glob("*")
                  if p.suffix in {".json", ".txt"}}
        named = set(re.findall(r"([a-z_0-9]+\.(?:json|txt))", block))
        assert not (actual - named), sorted(actual - named)


class TestItDescribesWhatShips:
    def test_the_paragraph_section_leads_with_the_letter_mode(self):
        """It described the word-order mode as the answer for ~40 iterations."""
        text = README.read_text()
        section = text[text.index("## Paragraphs"):text.index("## Credit")]
        head = section[:800]
        assert "letters" in head.lower()
        assert "/api/v2/paragraph" in head

    def test_the_word_mode_is_named_as_the_easier_constraint(self):
        """Whatever else the prose says, it must not sell the easy one."""
        text = README.read_text().lower()
        assert "easier constraint" in text

    def test_the_shipped_example_is_actually_a_palindrome(self):
        """A README example that does not verify is the worst kind of stale."""
        from llm_palindrome.validator import is_palindrome

        text = README.read_text()
        section = text[text.index("## Paragraphs"):text.index("### How")]
        indented = [ln.strip() for ln in section.split("\n")
                    if ln.startswith("    ") and ln.strip()]
        assert indented, "no example block found"
        assert is_palindrome(" ".join(indented))


class TestQuotedCountsMatchShippedData:
    """Numbers in the README must be true of the files in the repo.

    The paragraph pipeline's figures were produced by commands run inline
    during development, so unlike the benchmark table they have no script
    behind them. That is fine for a measurement taken once; it is not fine for
    a number describing a file that can change. Rebuilding `data/lexicon.txt`
    or `data/mirror_pairs.json` silently falsifies the prose otherwise.

    These check the counts the README states against what is actually shipped.
    A rebuild that moves a number now fails here and the prose gets updated
    with it.
    """
    import json as _json
    from pathlib import Path as _P

    README_TEXT = _P("README.md").read_text()

    def quoted(self, pattern):
        import re
        found = re.search(pattern, self.README_TEXT)
        assert found, f"README no longer states {pattern}"
        return int(found.group(1).replace(",", ""))

    def test_mirror_pair_count(self):
        import json
        from pathlib import Path
        pairs = json.loads(Path("data/mirror_pairs.json").read_text())
        assert len(pairs) == self.quoted(r"mirror_pairs\.json\s+([\d,]+) mined")

    def test_lexicon_size(self):
        from pathlib import Path
        words = Path("data/lexicon.txt").read_text().split()
        assert len(words) == self.quoted(r"lexicon\.txt\s+([\d,]+) headwords")

    def test_centre_count(self):
        import json
        from pathlib import Path
        centres = json.loads(Path("data/centres.json").read_text())
        assert len(centres) == self.quoted(r"`data/centres\.json` \| ([\d,]+) ")

    def test_spelling_count(self):
        import json
        from pathlib import Path
        spelled = json.loads(Path("data/canon_spelled.json").read_text())
        assert len(spelled) == self.quoted(
            r"`data/canon_spelled\.json` \| ([\d,]+) catalogued")

    def test_novelty_reference_size(self):
        import json
        from pathlib import Path
        canon = json.loads(Path("data/known_palindromes.json").read_text())
        assert len(canon) == self.quoted(r"is_novel_palindrome` \(([\d,]+) entries")


class TestExperimentsStillRun:
    def test_every_experiment_imports(self):
        """A script that no longer imports cannot reproduce what it is cited
        for. This is the cheap half of that check; the expensive half is
        running them, which needs a GPU allocation."""
        import importlib.util
        import os
        from pathlib import Path

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        broken = []
        for path in sorted(Path("experiments").glob("*.py")):
            spec = importlib.util.spec_from_file_location("probe", path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except SystemExit:
                pass
            except Exception as exc:
                broken.append(f"{path.name}: {type(exc).__name__}")
        assert not broken, broken
