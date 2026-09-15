"""Bounded semantic sentence-pair alignment experiment.

Authors complete, complementary sentences independently (observation/report,
action/result, question/answer).  Paraphrase alternatives are selected before
emission; the two complete sentences are then aligned over their entire
normalized tapes, so word and sentence boundaries may cross the mirror.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ID = "semantic-sentence-pair-alignment"
SIGNATURE = ("independent-complete-semantic-sentence-pairs|paraphrase-alternative-"
             "selection|whole-tape-cross-boundary-alignment|complementary-meaning-"
             "frames|no-local-window-or-reverse-prefix")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/semantic-sentence-pair-alignment-20260915.json"

# Each row is a complete, independently authored sentence.  Alternatives keep
# the same meaning while changing lexical and syntactic material jointly.
PAIRS = [
    (["Nora watches the quiet harbor.", "Nora observes the still harbor."],
     ["The harbor keeps Nora safe.", "The still harbor shelters Nora."]),
    (["Mara carries a warm lantern.", "Mara brings a small lantern."],
     ["The lantern guides Mara home.", "A small lantern leads Mara home."]),
    (["Ira records the first snowfall.", "Ira notes the early snowfall."],
     ["The snowfall covers Ira's path.", "The early snow hides Ira's path."]),
    (["A patient nurse checks the child.", "The careful nurse examines the child."],
     ["The child trusts the patient nurse.", "The child relies on the careful nurse."]),
    (["The baker saves a loaf for Ana.", "The baker keeps bread for Ana."],
     ["Ana shares the loaf with the baker.", "Ana gives the bread back to the baker."]),
]

def norm(s: str) -> str:
    return "".join(c.lower() for c in s if "a" <= c.lower() <= "z")

def two_pointer(t: str) -> bool:
    t = norm(t)
    i, j = 0, len(t) - 1
    while i < j:
        if t[i] != t[j]: return False
        i += 1; j -= 1
    return bool(t)

def readability(s: str) -> dict:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", s)
    return {"words": len(words), "mean_word_length": round(sum(map(len, words)) / len(words), 2),
            "complete_sentence": bool(re.search(r"[.!?]$", s))}

def main() -> None:
    branches = 0; aligned_prefixes = 0; exact = []
    probes = []
    for lefts, rights in PAIRS:
        for left in lefts:
            for right in rights:
                branches += 1
                text = left + " " + right
                tape = norm(text)
                # Whole-tape alignment: compare every mirrored character,
                # including characters crossing the sentence boundary.
                matched = 0
                for a, b in zip(tape, reversed(tape)):
                    if a != b: break
                    matched += 1
                aligned_prefixes += matched
                row = {"text": text, "letters": len(tape), "matching_outer_pairs": matched,
                       "exact": two_pointer(text), "readability": readability(text),
                       "provenance": "hand-authored complete sentence pair; paraphrase row"}
                probes.append(row)
                if row["exact"]: exact.append(row)
    # No catalogue lookup, repetition, or self-palindromic span is used here;
    # exact outputs still require a separate admission review.
    payload = {
        "experiment_id": ID, "signature": SIGNATURE,
        "method": "independent complete complementary sentences; whole-tape character alignment over semantic paraphrase alternatives",
        "seed": "An aide rips nine memos; some men inspire Diana.",
        "semantic_frames": len(PAIRS), "branches": branches,
        "aligned_outer_pairs_total": aligned_prefixes, "exact_count": len(exact),
        "rendered_probes": probes, "rendered_candidates": exact,
        "independent_audit": [{"text": r["text"], "two_pointer": two_pointer(r["text"]),
            "normalized_sha256": hashlib.sha256(norm(r["text"]).encode()).hexdigest()} for r in exact],
        "readability_note": "Programmatic word/sentence diagnostics only; no readability certification.",
        "repair_operator": "semantic paraphrase substitution at either complete-sentence row, followed by full-tape re-alignment; next test should add tense/aspect and connective alternatives while preserving complementary meaning",
        "provenance": "All sentence alternatives authored for this run; no catalogue lookup or imported prose.",
        "output_excluded_fingerprint": hashlib.sha256((Path(__file__).read_text()+ID+SIGNATURE).encode()).hexdigest(),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"branches": branches, "exact_count": len(exact), "aligned_outer_pairs_total": aligned_prefixes}, indent=2))

if __name__ == "__main__": main()
