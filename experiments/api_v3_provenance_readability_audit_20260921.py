"""Audit the public v3 API's capacity claim against its rendered surface.

The service is useful inspiration for explicit source/provenance and exactness
metadata.  This probe deliberately keeps those properties separate from
readability: the response is independently normalized and checked, while no
programmatic score is allowed to certify English.  The full rendered response
is preserved in the run artifact for human inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "runs" / "api-v3-provenance-readability-audit-20260921.json"
DEFAULT_BASE = "https://palindrome.ericspencer.us"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def get_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "palindrome-research-audit/2026-09-21"}
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


def probe_route(url: str) -> dict:
    """Record deployment state without treating a 404/503 as an exception."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "palindrome-research-audit/2026-09-21"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            body = response.read(4096)
            return {"status": response.status, "content_type": response.headers.get("Content-Type"),
                    "body_prefix": body.decode("utf-8", errors="replace")[:400]}
    except urllib.error.HTTPError as exc:
        body = exc.read(4096)
        return {"status": exc.code, "content_type": exc.headers.get("Content-Type"),
                "body_prefix": body.decode("utf-8", errors="replace")[:400]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--letters", type=int, default=420)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = args.base.rstrip("/")
    health_url = f"{base}/api/v3/health"
    composition_url = (
        f"{base}/api/v3/composition?"
        + urllib.parse.urlencode({"letters": args.letters, "seed": args.seed})
    )
    health = get_json(health_url)
    composition = get_json(composition_url)
    text = composition["text"]
    normalized = normalize(text)
    plain_normalized = normalize(composition["plain"])
    chunks = composition.get("chunks", [])
    route_audit = {
        "/api/v3/health": probe_route(f"{base}/api/v3/health"),
        "/api/v3/composition": probe_route(composition_url),
        "/api/v4/health": probe_route(f"{base}/api/v4/health"),
        "/api/v4/method": probe_route(f"{base}/api/v4/method"),
    }

    payload = {
        "experiment_id": "api-v3-provenance-readability-audit-20260921",
        "request": {"health": health_url, "composition": composition_url},
        "deployment_route_audit": route_audit,
        "deployment_gate_status": {
            "repository_v4_generation": "gated",
            "live_v4_routes_present": any(
                route_audit[path]["status"] == 200
                for path in ("/api/v4/health", "/api/v4/method")
            ),
            "live_v3_generation_present": route_audit["/api/v3/composition"]["status"] == 200,
            "mismatch_requires_deployment_action": (
                route_audit["/api/v3/composition"]["status"] == 200
                and not any(route_audit[path]["status"] == 200
                            for path in ("/api/v4/health", "/api/v4/method"))
            ),
            "note": "A live v3 200 is legacy deployment state, not reader evidence or a promotion decision.",
        },
        "health": health,
        "composition": {
            "text": text,
            "plain": composition["plain"],
            "letters_server": composition.get("letters"),
            "letters_independent": len(normalized),
            "exact_independent": bool(normalized) and normalized == normalized[::-1],
            "plain_matches_text_letters": normalized == plain_normalized,
            "sha256_normalized_forward": hashlib.sha256(normalized.encode()).hexdigest(),
            "sha256_normalized_reverse": hashlib.sha256(normalized[::-1].encode()).hexdigest(),
            "words": composition.get("words"),
            "pairs": composition.get("pairs"),
            "requested_letters": composition.get("requested_letters"),
            "capacity_letters": composition.get("capacity_letters"),
            "distinct_chunks": composition.get("distinct_chunks"),
            "repeats": composition.get("repeats"),
            "chunk_sources": sorted({chunk.get("source") for chunk in chunks}),
        },
        "readability": {
            "status": "not_run",
            "rule": "programmatic measures diagnose only; no readability certification",
            "reader_test_required": True,
        },
        "inspiration": {
            "useful": ["explicit capacity", "source labels", "independent exactness boundary"],
            "not_adopted": ["nested mirror-pair composition", "catalogue fallback as generated prose"],
            "goal_relevance": "capacity and exactness are necessary but do not establish readable English",
        },
        "provenance": {
            "source": "live public API response",
            "response_preserved": True,
            "finished_tape_reversal_by_this_probe": False,
            "post_hoc_repair_by_this_probe": False,
        },
        "next_constructive_use": "Carry the API's capacity/provenance fields into a genuinely online grammar search, while keeping reader admission separate.",
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "letters": payload["composition"]["letters_independent"],
        "exact": payload["composition"]["exact_independent"],
        "words": payload["composition"]["words"],
        "pairs": payload["composition"]["pairs"],
        "capacity_letters": payload["composition"]["capacity_letters"],
        "chunk_sources": payload["composition"]["chunk_sources"],
        "output": str(OUT),
    }))


if __name__ == "__main__":
    main()
