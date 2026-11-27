"""Collect public r/palindromes titles with attribution from its Atom feed.

No login, posting, browser state, or private data is used. Stop on HTTP errors;
retain page snapshots and already collected records. Fetch at most 10 pages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

NS = {"a": "http://www.w3.org/2005/Atom"}
FEED = "https://www.reddit.com/r/palindromes/new/.rss"


def parse_feed(body: bytes) -> list[dict]:
    root = ElementTree.fromstring(body)
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise ValueError("response is not an Atom feed")
    rows = []
    for entry in root.findall("a:entry", NS):
        text = entry.findtext("a:title", namespaces=NS)
        author = entry.findtext("a:author/a:name", namespaces=NS)
        author_url = entry.findtext("a:author/a:uri", namespaces=NS)
        link = entry.find("a:link", NS)
        source_url = link.get("href") if link is not None else None
        post_id = entry.findtext("a:id", namespaces=NS)
        if not all((text, author, author_url, source_url, post_id)):
            raise ValueError("post is missing text or attribution; refusing incomplete credit")
        url = urlsplit(source_url)
        if url.hostname != "www.reddit.com" or not url.path.startswith("/r/palindromes/comments/"):
            raise ValueError("unexpected source outside r/palindromes")
        if post_id != "t3_" + url.path.split("/")[4]:
            raise ValueError("post ID and permalink disagree")
        rows.append(dict(text=text, author=author.lstrip("/"), author_url=author_url,
                         source_url=source_url, post_id=post_id,
                         published=entry.findtext("a:published", namespaces=NS),
                         updated=entry.findtext("a:updated", namespaces=NS)))
    return rows


def collect(out: Path, pages: int = 5) -> dict:
    if not 1 <= pages <= 10:
        raise ValueError("pages must be between 1 and 10")
    snapshots = out / "feed-snapshots"
    destination = out / "feed-posts.json"
    if destination.exists() or any(snapshots.glob("page-[0-9]*.xml")):
        raise FileExistsError("collection already exists; choose a new output directory to preserve it")
    out.mkdir(parents=True, exist_ok=True)
    snapshots.mkdir(exist_ok=True)
    result = dict(collected_at=datetime.now(timezone.utc).isoformat(),
                  source=FEED, scope="Public post titles only; not comments or images.",
                  credit="Usernames credit Reddit posters; original authorship is not independently established.",
                  requests=[], rows=[])
    seen, after = set(), None
    error = None
    current_url = None

    def save_progress():
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    try:
        for index in range(pages):
            params = {"limit": 100}
            if after:
                params["after"] = after
            url = FEED + "?" + urlencode(params)
            current_url = url
            request = Request(url, headers={"User-Agent": "palindrome-sentence-generator/0.1 (credited research collection)"})
            with urlopen(request, timeout=20) as response:
                body = response.read()
            page = snapshots / f"page-{index+1:02}.xml"
            page.write_bytes(body)
            rows = parse_feed(body)
            added = [row for row in rows if row["post_id"] not in seen]
            seen.update(row["post_id"] for row in added)
            result["rows"].extend(added)
            result["requests"].append(dict(url=url, snapshot=str(page.relative_to(out)),
                                            posts=len(rows), new_posts=len(added)))
            save_progress()
            print(f"Page {index+1}: {len(added)} new posts, {len(result['rows'])} total", flush=True)
            if not rows or not added:
                break
            after = rows[-1]["post_id"]
            if index + 1 < pages:
                time.sleep(1)
    except Exception as exc:
        error = exc
        result["collection_error"] = f"{type(exc).__name__}: {exc}"
        result["failed_request"] = {"url": current_url,
                                    "http_status": exc.code if isinstance(exc, HTTPError) else None}
    save_progress()
    if error is not None:
        raise error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=5)
    args = parser.parse_args()
    collect(args.out, args.pages)


if __name__ == "__main__":
    main()
