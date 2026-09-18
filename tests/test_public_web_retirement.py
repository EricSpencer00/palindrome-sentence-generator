from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_static_panama_route_is_a_small_withdrawal_page_not_legacy_output():
    for location in ("public", "dist"):
        page = (
            ROOT / "web" / location / "panama" / "index.html"
        ).read_text().lower()
        assert "legacy page is withdrawn" in page
        assert "a man, a plan" not in page
        assert len(page) < 2_500


def test_source_app_exposes_no_generation_endpoint_or_legacy_promise():
    source = (ROOT / "web" / "src" / "App.tsx").read_text()
    assert "/api/generate" not in source
    assert "/api/v2/generate" not in source
    assert "Generation is unavailable." in source
