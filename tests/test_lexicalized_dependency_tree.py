from experiments.lexicalized_dependency_tree import (
    DirectionalIndex,
    Template,
    checks,
    letters,
    solve_template,
)


def test_prefix_index_returns_exactly_the_compatible_options():
    forward = DirectionalIndex(("alpha", "alpine", "beta"), reverse=False)
    assert [option.text for option in forward.compatible("al")] == ["alpha", "alpine"]
    assert [option.text for option in forward.compatible("alphabet")] == ["alpha"]
    assert forward.compatible("z") == ()


def test_known_sentence_is_a_control_not_a_novel_admission():
    template = Template("control", "lets", "see", "visible object")
    # Temporarily narrow the lexical domains through a local subclass so the
    # exact solver must discover, rather than be handed, the complete tape.
    class ControlTemplate(Template):
        def slots(self):
            return (("marge",), ("lets",), ("norah",), ("see",), ("sharons",), ("telegram",))

    rows, stats = solve_template(ControlTemplate("control", "lets", "see", "visible object"))
    assert rows == [("marge", "lets", "norah", "see", "sharons", "telegram")]
    assert stats["closed_exact"] == 1
    text = template.render(rows[0])
    assert letters(text) == letters(text)[::-1]
    assert checks(text, {letters(text)})["local_catalogue_absent"] is False
    assert checks(text, set())["not_catalogue_family_derivative"] is False


def test_duplicate_or_self_palindromic_word_is_never_returned():
    class DegenerateTemplate(Template):
        def slots(self):
            return (("anna",), ("lets",), ("anna",), ("see",), ("annas",), ("telegram",))

    rows, _ = solve_template(DegenerateTemplate("degenerate", "lets", "see", "visible object"))
    assert rows == []
