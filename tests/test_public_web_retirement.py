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


def test_root_streams_the_v5_engine_and_v3_stays_off_the_root():
    source = (ROOT / "web" / "src" / "App.tsx").read_text()
    assert '"/api/generate"' in source
    assert 'const IS_V3 = PATH === "/v3" || PATH === "/dev"' in source
    proxy = (ROOT / "web" / "functions" / "api" / "[[path]].js").read_text()
    assert "(?:generate|v" not in proxy  # bare /api/generate passes the proxy
