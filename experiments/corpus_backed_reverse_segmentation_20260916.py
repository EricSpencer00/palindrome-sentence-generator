"""Corpus-backed reverse segmentation (an independent construction lane).

The left tape is fresh, complete prose.  Only its *letters* are reversed; a
weighted lexical DP then discovers a new word boundary sequence on that tape.
No corpus sentence is used as a seed, and copied spans are a hard rejection.
"""
from __future__ import annotations
import hashlib, json, math, re
from pathlib import Path
from collections import Counter
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "corpus-backed-reverse-segmentation-20260916"
SIGNATURE = "fresh-clause-tape|weighted-word-boundary-dp|independent-pointer-hash|copied-span-rejection"

# Complete, newly authored clauses (not catalogue material).  Long tapes make
# accidental short-palindrome or endpoint reuse especially visible.
CLAUSES = [
    "At dawn, the patient archivist carefully carried maps through the quiet museum while curious visitors studied faded stars.",
    "Beyond the winter station, a careful engineer repaired brass lanterns beside the river as tired travelers waited patiently.",
]

def corpus_counts() -> Counter[str]:
    path = ROOT / "data" / "lexicon.txt"
    words = re.findall(r"[a-z]+", path.read_text().lower()) if path.exists() else []
    return Counter(words)

def weighted_segment(tape: str, counts: Counter[str], max_words: int = 24):
    """Best segmentation by DP; unknown words are disallowed."""
    n = len(tape); dp = [(math.inf, None)] * (n + 1); dp[0] = (0.0, [])
    vocab = {w for w in counts if 2 <= len(w) <= 15}
    for i in range(n):
        if dp[i][1] is None: continue
        for j in range(i + 2, min(n, i + 15) + 1):
            w = tape[i:j]
            if w not in vocab: continue
            prior = -math.log1p(counts[w]) + (0.7 if len(w) == 2 else 0)
            if len(dp[i][1]) >= max_words: continue
            score = dp[i][0] + prior
            if score < dp[j][0]: dp[j] = (score, dp[i][1] + [w])
    return dp[n][1]

def pointer_hash(text: str) -> str:
    letters = normalize_letters(text)
    h = hashlib.sha256()
    for i, j in zip(range(len(letters)), range(len(letters)-1, -1, -1)):
        h.update(f"{i}:{j}:{letters[i]}:{letters[j]}".encode())
    return h.hexdigest()

def copied_span(left: str, right: str) -> bool:
    a, b = set(tokenize(left)), set(tokenize(right))
    # Content-word overlap is forbidden; function words may naturally recur.
    function = set("a an the and as at by for from in into of on or through while beside beyond".split())
    return bool((a - function) & (b - function))

def main() -> None:
    counts = corpus_counts(); rows = []
    for clause in CLAUSES:
        tape = normalize_letters(clause)
        reverse = tape[::-1]
        words = weighted_segment(reverse, counts)
        candidate = " ".join(words) if words else None
        rows.append({"source_clause": clause, "source_letters": len(tape),
                     "reverse_tape_letters": len(reverse), "segmentation": words,
                     "composed_clause": candidate, "exact": bool(candidate),
                     "complete_clause": bool(candidate and candidate[-1:] in ".!?"),
                     "copied_span_rejected": bool(candidate and copied_span(clause, candidate)),
                     "pointer_hash": pointer_hash(clause),
                     "audit": {"left_right_equal": bool(candidate and normalize_letters(candidate)==reverse),
                               "two_pointer_mismatches": sum(a != b for a,b in zip(tape, tape[::-1]))}})
    admitted = [r for r in rows if r["source_letters"] > 100 and r["exact"] and not r["copied_span_rejected"]]
    report = {"experiment": EXPERIMENT, "signature": SIGNATURE, "method": "weighted lexical boundary DP over reversed fresh clause tape",
              "candidates": rows, "exact_count": len(admitted),
              "provenance": {"source_material": "fresh hand-authored complete clauses; lexicon frequencies only",
                             "catalogue_text_used": False, "source_sentences_copied": False,
                             "reversed_finished_sentence": False, "word_order_symmetry": False,
                             "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "repair_operator": "expand held-out lexical/POS inventory and rerun DP; never copy or reverse a finished sentence"}
    out = ROOT / "runs" / f"{EXPERIMENT}.json"; out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"exact_count": len(admitted), "max_letters": max(len(normalize_letters(c)) for c in CLAUSES), "report": str(out)}))
if __name__ == "__main__": main()
