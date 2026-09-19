"""Bounded reverse-phrase lattice experiment.

Unlike word-order mirroring, each edge maps a *phrase* on the left to a
different phrase on the right.  A path is accepted only when the left tape is
an authored sentence; the right tape is then independently checked for exact
character symmetry and can be compared with a sentence bank.  This keeps the
construction honest: grammaticality is not inferred from palindrome status.
"""
from __future__ import annotations

import json, re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORD = re.compile(r"[a-z]+")

def words(s: str) -> tuple[str, ...]:
    return tuple(WORD.findall(s.lower()))

def norm(s: str) -> str:
    return "".join(words(s))

def search(sentences, pairs, max_edges=8):
    by_left = {}
    for p in pairs:
        by_left.setdefault(tuple(p["left"]), []).append(p)
    out = []
    for left in sentences:
        toks = words(left)
        frontier = {0: [()]}
        for i in range(len(toks)):
            for j in range(i + 1, len(toks) + 1):
                for edge in by_left.get(toks[i:j], ()):
                    for path in frontier.get(i, ()):
                        if len(path) < max_edges:
                            frontier.setdefault(j, []).append(path + (edge,))
        for path in frontier.get(len(toks), ()):
            right = tuple(w for edge in path for w in edge["right"])
            ltxt, rtxt = " ".join(toks), " ".join(right)
            out.append({"left": ltxt, "right": rtxt,
                        "edges": len(path), "exact": norm(ltxt + rtxt) == norm(ltxt + rtxt)[::-1],
                        "right_in_bank": right in sentence_set})
    return out

if __name__ == "__main__":
    pairs = json.loads((ROOT / "data/mirror_pairs.json").read_text())
    authored = [x.strip() for x in (ROOT / "data/authored_sentences.txt").read_text().splitlines() if x.strip()]
    composed = json.loads((ROOT / "data/composed_sentences.json").read_text())["sentences"]
    sentence_set = {words(x) for x in authored + composed}
    rows = search(authored + composed, pairs)
    exact = [r for r in rows if r["exact"]]
    result = {"method": "reverse_phrase_lattice", "sentences": len(authored + composed),
              "edges": len(pairs), "paths": len(rows), "exact_paths": len(exact),
              "two_sided_bank_paths": sum(r["right_in_bank"] for r in exact),
              "candidates": exact}
    out = ROOT / "artifacts/reverse_phrase_lattice.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in result if k != "candidates"}, indent=2))
