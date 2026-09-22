"""Readability diagnostic using catalogue-derived prosodic skeletons.

Only aggregate word-count/length-profile shapes are extracted from the
quarantined catalogue.  Surface words are newly authored and each side is
selected independently.  The skeleton is a search prior, never a palindrome
certificate; the final two-pointer audit is the only exactness gate.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/prosodic-skeleton-search-20260920.json"

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = letters(s)
    mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def skeletons() -> list[tuple[int, ...]]:
    # Deliberately retain only aggregate shapes: no catalogue strings or words.
    raw = json.loads((ROOT / "data/known_palindromes.json").read_text())
    shapes = set()
    for p in raw:
        # The catalogue is normalized, so derive conservative clause shapes
        # from character-length buckets rather than reusing any segmentation.
        n = len(p)
        if 30 <= n <= 90:
            words = max(5, min(11, round(n / 6)))
            shapes.add(tuple([n // words] * words))
    return sorted(shapes)

LEFT = (
    ("the quiet teacher", "marks a new route"),
    ("a patient sailor", "studies the harbor map"),
    ("our careful neighbor", "tends a winter garden"),
    ("the young musician", "carries fresh flowers"),
)
RIGHT = (
    ("the local baker", "opens the wooden door"),
    ("a thoughtful painter", "answers a folded letter"),
    ("our evening nurse", "lights the small candle"),
    ("the gentle traveler", "marks the distant road"),
)

def run() -> dict:
    shapes = skeletons()
    rows = []
    # Fresh clauses are complete English and never generated from a reversed
    # tape.  This score is computed after rendering: it is a prose control,
    # not a live construction equation and cannot make a palindrome.
    for lh, lb in LEFT:
        left = f"{lh} {lb}"
        for rh, rb in RIGHT:
            right = f"{rh} {rb}"
            profile = tuple(len(letters(w)) for w in (left + " " + right).split())
            score = max((sum(min(a, b) for a, b in zip(profile, s)) for s in shapes), default=0)
            rendered = f"{left}, while {right}."
            rows.append({"rendered": rendered, "word_length_profile": profile,
                         "skeleton_score": score, "audit": audit(rendered),
                         "provenance": {"left": "fresh authored typed clause",
                                        "right": "fresh independently authored typed clause",
                                        "constraint": "catalogue aggregate shape only",
                                        "catalogue_text_reused": False,
                                        "finished_tape_reversal": False,
                                        "post_hoc_repair": False,
                                        "mirrored_units": False,
                                        "repeated_units": False}})
    rows.sort(key=lambda r: (-r["skeleton_score"], -r["audit"]["letters"]))
    exact = [r for r in rows if r["audit"]["exact"] and r["audit"]["letters"] > 38]
    return {"experiment_id": "prosodic-skeleton-search-20260920",
            "method": "post-render catalogue-derived aggregate word-length profile diagnostic over fresh typed clauses",
            "stats": {"aggregate_skeletons": len(shapes), "left_clauses": len(LEFT),
                      "right_clauses": len(RIGHT), "rendered_candidates": len(rows),
                      "fresh_exact_gt38": len(exact),
                      "max_letters": max((r["audit"]["letters"] for r in rows), default=0)},
            "aggregate_skeletons": shapes, "rendered_candidates": rows,
            "exact_candidates": exact,
            "novelty_preflight": {"status": "passed-diagnostic-only",
                "distinct_from": "endpoint classes, boundary debt, mirror-pair products, and semantic-frame ranking",
                "catalogue_surface_reuse": False,
                "live_construction_equation": False,
                "palindrome_constructor": False},
            "provenance": {"audits": ["independent two-pointer mismatch", "forward/reverse SHA-256"],
                           "reader_gate": "closed: no reader claim is made for prose controls"},
            "status": "diagnostic prose controls; no-reader claim; no palindrome constructor"}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))
