from io import BytesIO
import json
from urllib.error import HTTPError

import pytest
from tools import collect_reddit_palindromes as collector
from tools.collect_reddit_palindromes import collect, parse_feed


FEED = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>t3_abc</id><title>Madam, I'm Adam!</title>
<author><name>/u/example</name><uri>https://www.reddit.com/user/example</uri></author>
<link href="https://www.reddit.com/r/palindromes/comments/abc/example/"/>
<published>2026-10-09T00:00:00Z</published></entry></feed>'''


def test_exact_title_and_credit_are_retained():
    row, = parse_feed(FEED)
    assert row['text'] == "Madam, I'm Adam!"
    assert row['author'] == 'u/example'
    assert row['source_url'].endswith('/abc/example/')
    assert row['published'] == '2026-10-09T00:00:00Z'


@pytest.mark.parametrize('body', [
    b'<html>Blocked</html>',
    FEED.replace(b'<name>/u/example</name>', b''),
    FEED.replace(b'/r/palindromes/', b'/r/unrelated/'),
    FEED.replace(b't3_abc', b't3_wrong'),
])
def test_missing_credit_or_wrong_source_fails(body):
    with pytest.raises(ValueError):
        parse_feed(body)


@pytest.mark.parametrize("existing", ["feed-posts.json", "feed-snapshots/page-01.xml"])
def test_existing_evidence_refuses_collection_before_network(tmp_path, monkeypatch, existing):
    artifact = tmp_path / existing
    artifact.parent.mkdir(parents=True, exist_ok=True)
    original = b"preserve this previous collection"
    artifact.write_bytes(original)

    def unexpected_network(*args, **kwargs):
        pytest.fail("existing output must be rejected before network access")

    monkeypatch.setattr(collector, "urlopen", unexpected_network)
    with pytest.raises(FileExistsError, match="collection already exists"):
        collect(tmp_path)
    assert artifact.read_bytes() == original


def test_second_page_rate_limit_preserves_partial_progress_and_request(tmp_path, monkeypatch):
    calls = []

    def request(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            return BytesIO(FEED)
        # Progress must already exist before the second request begins.
        saved = json.loads((tmp_path / "feed-posts.json").read_text())
        assert saved["rows"][0]["post_id"] == "t3_abc"
        raise HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(collector, "urlopen", request)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    with pytest.raises(HTTPError) as failure:
        collect(tmp_path, pages=3)
    assert failure.value.code == 429
    assert len(calls) == 2
    saved = json.loads((tmp_path / "feed-posts.json").read_text())
    assert saved["rows"] == parse_feed(FEED)
    assert saved["failed_request"] == {"url": calls[1], "http_status": 429}
    assert "after=t3_abc" in calls[1]
    assert "HTTPError" in saved["collection_error"]
    assert (tmp_path / "feed-snapshots/page-01.xml").read_bytes() == FEED
    assert not (tmp_path / "feed-snapshots/page-02.xml").exists()


def test_first_request_error_is_recorded_without_retry(tmp_path, monkeypatch):
    calls = []

    def request(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(collector, "urlopen", request)
    with pytest.raises(HTTPError):
        collect(tmp_path, pages=3)
    saved = json.loads((tmp_path / "feed-posts.json").read_text())
    assert saved["rows"] == []
    assert saved["failed_request"] == {"url": calls[0], "http_status": 403}
    assert len(calls) == 1
