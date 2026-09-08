"""Measure hard semantic-edge capacity over v3 sentence mirror-pairs."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_palindrome.hierarchy import sentence_centres, sentence_pairs
from llm_palindrome.semantic_graph import paired_path
from llm_palindrome.validator import is_palindrome
from server import v3

HOST = "http://localhost:11434/api/embed"


def embed(model: str, texts: list[str], batch: int = 96) -> dict[str, list[float]]:
    out = {}
    for start in range(0, len(texts), batch):
        group = texts[start:start + batch]
        payload = json.dumps({"model": model, "input": group}).encode()
        req = urllib.request.Request(HOST, payload, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            vectors = json.load(response)["embeddings"]
        out.update(zip(group, vectors))
    return out


def material():
    v3.ensure_loaded()
    table, shapes, trigrams = v3._tables
    raw, seen = [], set()
    for row in v3._bank:
        if row["source"] != "generated":
            continue
        got = v3.harvest_pair(row["words"])
        if not got:
            continue
        left, right = got
        keys = " ".join(left), " ".join(right)
        if keys[0] == keys[1] or any(key in seen for key in keys):
            continue
        seen.update(keys)
        raw.append((left, right, row["source"]))
    pairs = [{"left": " ".join(left), "right": " ".join(right),
              "source": source} for left, right, source
             in sentence_pairs(raw, table, shapes, trigrams)]
    centres = sentence_centres(
        [row for row in v3._bank if row["source"] == "generated"],
        table, shapes, trigrams)
    return pairs, [" ".join(row["words"]) for row in centres]


def assemble(path, centre):
    words = []
    for pair in path:
        words += pair["left"].split()
    words += centre.split()
    for pair in reversed(path):
        words += pair["right"].split()
    text = " ".join(words)
    assert is_palindrome(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--thresholds", default="0.35,0.4,0.45,0.5,0.55")
    ap.add_argument("--want", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("runs/sentence_graph.json"))
    args = ap.parse_args()
    pairs, centres = material()
    texts = sorted({text for pair in pairs for text in (pair["left"], pair["right"])}
                   | set(centres))
    vectors = embed(args.model, texts)
    rows = []
    for threshold in [float(x) for x in args.thresholds.split(",")]:
        for seed in range(args.seeds):
            centre = centres[seed % len(centres)]
            path = paired_path(pairs, centre, vectors, threshold,
                               want=args.want, seed=seed)
            text = assemble(path, centre) if path else centre
            sentences = ([pair["left"] for pair in path] + [centre]
                         + [pair["right"] for pair in reversed(path)])
            rows.append({"threshold": threshold, "seed": seed,
                         "pairs": len(path), "words": len(text.split()),
                         "centre": centre, "sentences": sentences,
                         "rendered": ". ".join(s.capitalize() for s in sentences) + ".",
                         "text": text})
        at = [row for row in rows if row["threshold"] == threshold]
        print(threshold, "pairs", min(r["pairs"] for r in at),
              sum(r["pairs"] for r in at) / len(at),
              max(r["pairs"] for r in at), flush=True)
    result = {"model": args.model, "pair_bank": len(pairs),
              "centres": len(centres), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
