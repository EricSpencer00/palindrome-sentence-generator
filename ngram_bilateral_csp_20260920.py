"""Observed n-gram lattice intersected with bilateral grammar character CSP.

The n-gram table is a search-space constraint: it admits only observed
adjacent word transitions while two independently parsed clauses consume
opposing characters online. It never scores, reranks, or repairs a completed
candidate.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

from forward_lexicalized_grammar_20260920 import (
    GRAMMAR,
    Word,
    admission_ok,
    independent_audit,
    letters,
    render_path,
)
from bilateral_grammar_csp_20260920 import _consume

ROOT = Path(__file__).resolve().parent


def load_observed_edges(path=ROOT / "data/ngrams_wikitext2.json", limit=20_000):
    table = json.loads(Path(path).read_text())
    edges = set()
    rows = 0
    tables = table.values() if isinstance(table, dict) else (table,)
    for phrases in tables:
        if not isinstance(phrases, list):
            continue
        for phrase in phrases:
            words = phrase.casefold().split() if isinstance(phrase, str) else []
            words = [re.sub(r"[^a-z]", "", word) for word in words]
            words = [word for word in words if word]
            if len(words) < 2:
                continue
            rows += 1
            edges.update(zip(words, words[1:]))
            if rows >= limit:
                return edges, rows
    return edges, rows


def load_brown_ngram_lexicon(limit=500, edge_limit=20_000):
    """Use Brown POS entries whose surfaces occur in the observed lattice."""
    edges, rows = load_observed_edges(limit=edge_limit)
    surfaces = {word for edge in edges for word in edge}
    table = json.load(gzip.open(ROOT / "tools/polaris/payload/brown.json.gz", "rt"))["table"]
    allowed = {"DET": "DET", "NOUN": "N", "VERB": "V", "PROPN": "PROPN"}
    lexicon = []
    for text in sorted(surfaces):
        if text not in table or not text.isalpha() or len(text) > 14:
            continue
        pos = next((allowed[tag] for tag in table[text] if tag in allowed), None)
        if pos:
            lexicon.append(Word(text, pos))
        if len(lexicon) >= limit:
            break
    return tuple(lexicon), edges, rows


def ngram_bilateral_csp(lexicon, edges, grammar=None, max_words=10, max_nodes=100_000):
    grammar = grammar or GRAMMAR
    by_pos = {}
    for word in lexicon:
        if letters(word.text):
            by_pos.setdefault(word.pos, []).append(word)
    found = []
    stats = {"nodes": 0, "pruned_character": 0, "pruned_transition": 0, "complete": 0}

    def observed(prev, nxt):
        return not prev or (prev, nxt) in edges

    def search(lsymbols, rsymbols, left, right_rev, lres="", rres="", lprev=""):
        if stats["nodes"] >= max_nodes:
            return
        stats["nodes"] += 1
        if not lsymbols and not rsymbols:
            stats["complete"] += 1
            if lres or rres or not left or not right_rev:
                return
            words = left + list(reversed(right_rev))
            if not admission_ok(words):
                return
            text = " ".join(words)
            audit = independent_audit(text)
            if audit["exact"]:
                found.append({"length": audit["letters"], "rendered": render_path(text, lexicon),
                              "words": words, "audit": audit,
                              "provenance": {"observed_ngram_edges": True,
                                             "left_clause_grammar": True,
                                             "right_clause_reverse_expansion": True,
                                             "online_character_residual": True,
                                             "candidate_reranking": False,
                                             "finished_tape_reversal": False,
                                             "post_hoc_repair": False,
                                             "catalogue_text": False}})
            return
        if len(left) + len(right_rev) >= max_words:
            return
        if lsymbols and lsymbols[0] in grammar:
            for production in grammar[lsymbols[0]]:
                search(list(production) + list(lsymbols[1:]), rsymbols, left, right_rev, lres, rres, lprev)
            return
        if rsymbols and rsymbols[-1] in grammar:
            for production in grammar[rsymbols[-1]]:
                search(lsymbols, list(rsymbols[:-1]) + list(production), left, right_rev, lres, rres, lprev)
            return
        if lsymbols and rsymbols and lsymbols[0] not in grammar and rsymbols[-1] not in grammar:
            for a in by_pos.get(lsymbols[0], ()):
                if not observed(lprev, a.text):
                    stats["pruned_transition"] += len(by_pos.get(rsymbols[-1], ()))
                    continue
                for b in by_pos.get(rsymbols[-1], ()):
                    if right_rev and (b.text, right_rev[-1]) not in edges:
                        stats["pruned_transition"] += 1
                        continue
                    residual = _consume(lres + letters(a.text), rres + letters(b.text)[::-1])
                    if residual is None:
                        stats["pruned_character"] += 1
                        continue
                    search(lsymbols[1:], rsymbols[:-1], left + [a.text], right_rev + [b.text], *residual, a.text)
            return
        # A one-sided terminal is possible only at the end of the opposite
        # grammar; retain it for general grammars while preserving transitions.
        if lsymbols:
            for a in by_pos.get(lsymbols[0], ()):
                if not observed(lprev, a.text):
                    continue
                residual = _consume(lres + letters(a.text), rres)
                if residual is not None:
                    search(lsymbols[1:], rsymbols, left + [a.text], right_rev, *residual, a.text)
            return
        if rsymbols:
            for b in by_pos.get(rsymbols[-1], ()):
                if right_rev and (b.text, right_rev[-1]) not in edges:
                    continue
                residual = _consume(lres, rres + letters(b.text)[::-1])
                if residual is not None:
                    search(lsymbols, rsymbols[:-1], left, right_rev + [b.text], *residual, lprev)

    search(["CLAUSE"], ["CLAUSE"], [], [])
    stats["status"] = "timeout" if stats["nodes"] >= max_nodes else ("SAT" if found else "UNSAT")
    return {"paths": sorted(found, key=lambda row: (row["length"], row["rendered"])), "stats": stats,
            "provenance": {"method": "observed n-gram transition lattice plus bilateral grammar CSP",
                           "edge_count": len(edges), "candidate_reranking": False,
                           "next_construction": "expand observed transition budget only after a complete readable exact row"}}


if __name__ == "__main__":
    lexicon, edges, rows = load_brown_ngram_lexicon()
    result = ngram_bilateral_csp(lexicon, edges)
    result["provenance"].update({"lexicon": len(lexicon), "ngram_rows": rows})
    print(json.dumps(result["stats"]))
