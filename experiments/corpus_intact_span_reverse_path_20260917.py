"""Corpus-scale intact-prose span graph with reversed-character residual edges.

This is a construction experiment, not a catalogue lookup: vertices are
attested sentence spans and an edge consumes the longest outside-in character
match against the reverse residual. Paths cannot reuse a source span.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "authored_sentences.txt"
OUT = ROOT / "runs" / "corpus-intact-span-reverse-path-20260917.json"
MAX_SIDE = 3

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    n = norm(text)
    i, j = 0, len(n) - 1
    ok = True
    while i < j:
        if n[i] != n[j]:
            ok = False
            break
        i += 1; j -= 1
    return {"normalized_length": len(n), "two_pointer_exact": ok,
            "sha256": hashlib.sha256(n.encode()).hexdigest()}

def extend(residual: str, span: str) -> tuple[str, str]:
    """Consume matching chars at residual's left against span's right."""
    k = 0
    while k < len(residual) and k < len(span) and residual[k] == span[-1-k]:
        k += 1
    return residual[k:], span[:-k] if k else span

def main() -> None:
    raw = [x.strip() for x in SOURCE.read_text().splitlines() if x.strip()]
    spans = [{"id": i, "text": s, "norm": norm(s)} for i, s in enumerate(raw)]
    # Keep intact corpus sentences as vertices; edges carry exact residuals.
    edges = []
    for a in spans:
        for b in spans:
            if a["id"] == b["id"]:
                continue
            residual, remainder = extend(a["norm"], b["norm"])
            consumed = len(a["norm"]) - len(residual)
            if consumed:
                edges.append({"from": a["id"], "to": b["id"],
                              "consumed": consumed, "residual": residual,
                              "remainder": remainder})
    # Search non-repeated paths on both sides, bounded to keep provenance clear.
    best = {"matched_chars": 0, "left_ids": [], "right_ids": [], "residual": ""}
    closures = []
    def visit(left_ids, right_ids, residual, matched):
        nonlocal best
        if matched > best["matched_chars"]:
            best = {"matched_chars": matched, "left_ids": left_ids[:],
                    "right_ids": right_ids[:], "residual": residual}
        if not residual and left_ids and right_ids:
            text = " ".join(spans[i]["text"] for i in left_ids + right_ids)
            report = audit(text)
            if report["two_pointer_exact"]:
                closures.append({"text": text, "left_ids": left_ids[:],
                                 "right_ids": right_ids[:], "audit": report})
            return
        if len(left_ids) >= MAX_SIDE or len(right_ids) >= MAX_SIDE:
            return
        used = set(left_ids) | set(right_ids)
        for b in spans:
            if b["id"] in used:
                continue
            nxt, _ = extend(residual or b["norm"], b["norm"])
            gained = len(residual or b["norm"]) - len(nxt)
            if gained or not residual:
                visit(left_ids, right_ids + [b["id"]], nxt, matched + gained)
    # Seed each left span as a residual and grow a distinct right-side path.
    for a in spans:
        visit([a["id"]], [], a["norm"][::-1], 0)
    result = {
        "representation": "intact-corpus-span graph; directed edges consume reversed-character residuals",
        "source": str(SOURCE.relative_to(ROOT)), "sentence_count": len(spans),
        "max_sentences_per_side": MAX_SIDE, "edge_count": len(edges),
        "search": "distinct-span outside-in residual path; no copied span or catalogue units",
        "closures": closures, "best_frontier": best,
        "independent_audit": "two-pointer character comparison plus SHA-256 normalized tape",
        "next_repair": "add a provenance-preserving bridge-span index keyed by residual prefix/suffix, then rerun with four-span paths; do not mutate corpus text",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"edges": len(edges), "closures": len(closures), "best": best}, indent=2))

if __name__ == "__main__":
    main()
