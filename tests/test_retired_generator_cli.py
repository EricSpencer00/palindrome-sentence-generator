from __future__ import annotations

import pytest


def test_legacy_generator_cli_is_retired() -> None:
    from llm_palindrome.generate import main

    with pytest.raises(SystemExit, match="retired"):
        main()
