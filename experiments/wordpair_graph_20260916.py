"""Small reproducible word-pair graph run with independent clause inventory."""
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.wordpair_graph import WordPair, search, render, tape, fingerprint

PAIRS = [
    WordPair("Quiet archivists preserve maps", "maps preserve quiet archivists"),
    WordPair("patient editors weigh every claim", "every claim weighs patient editors"),
    WordPair("gardeners water young cedar", "young cedar waters gardeners"),
    WordPair("clear notes guide future readers", "future readers guide clear notes"),
    WordPair("careful hands repair old clocks", "old clocks repair careful hands"),
]

def main():
    result = search(PAIRS, max_depth=5)
    rows=[]
    for kind, paths in (("closure", result["closures"]), ("best_intact_prose", [result["best"]])):
        for path in paths[:10]:
            text=render(path); normalized=tape(text)
            rows.append({"kind":kind,"text":text,"letters":len(normalized),
                         "exact":normalized == normalized[::-1],
                         "distinct_pairs":len({(p.left,p.right) for p in path}),
                         "tape_sha256":fingerprint(path),
                         "pos_valency_gate":all(p.valency == "clause" for p in path)})
    out={"method":"independent_word_pair_graph_v1",
         "inventory":len(PAIRS),"expansions":result["expansions"],
         "closures":len(result["closures"]),"candidates":rows,
         "provenance":{"inventory":"authored ordinary-English clause pairs",
                        "construction":"left appends; right prepends; residual ledger",
                        "forbidden":"catalogue lookup, self-palindrome, word-order reversal shortcut"},
         "audits":{"pointer":"independent tape concatenation and reverse equality",
                   "hash":"SHA-256 of normalized rendered tape",
                   "mechanical_gate":"exact tape, distinct pairs, clause valency",
                   "novelty":"candidate hashes are emitted for external corpus preflight"},
         "next_repair":"Replace the mirrored right clauses with independently authored short clauses whose character residual matches; current inventory yields no closure."}
    outpath=Path(__file__).parents[1]/"runs"/"wordpair-graph-2026-09-16.json"; outpath.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"output":str(outpath),"closures":len(result["closures"]),"best_letters":rows[-1]["letters"] if rows else 0}))
if __name__ == "__main__": main()
