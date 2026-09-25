from paper.audit_week_results import (
    build,
    escape_tex,
    render_representative_output,
)


def test_paper_prints_the_independently_audited_752_letter_endpoint():
    records = build()["results"]
    endpoint = next(row for row in records if row["id"] == "752-center-path")
    rendered = render_representative_output(records)

    assert endpoint["letters"] == 752
    assert endpoint["independently_exact"] is True
    assert endpoint["normalized_sha256"] == (
        "109c758ecfa0c0a2049c1dfb925d3820eb453a92836c0835d80f5106f222de3c"
    )
    assert escape_tex(endpoint["surface"]) in rendered
    assert endpoint["normalized_sha256"] in rendered
    assert "rough prose, not reader-validated" in rendered
