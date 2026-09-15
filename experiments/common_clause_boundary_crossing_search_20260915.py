"""Constructive search over ordinary typed clauses with free word boundaries.

The left side is generated from small, common English SVO/PP frames.  Its
reversed character tape is then segmented into a *different* typed frame; this
allows a reversal to cross word boundaries instead of requiring mirrored
words.  Every exact render is printed in the report and sent through the
shared mechanical gate.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
POOLS = {
    "det": ("a", "the", "this", "my"),
    "subj": ("artist", "captain", "editor", "teacher", "writer", "pilot", "doctor", "child"),
    "verb": ("draws", "finds", "holds", "keeps", "makes", "opens", "reads", "sends", "writes", "sees"),
    "obj": ("book", "door", "gift", "letter", "map", "note", "plan", "report", "story", "room"),
    "prep": ("in", "near", "on", "with", "by"),
    "place": ("garden", "office", "park", "studio", "town", "school"),
    "adv": ("today", "early", "gently", "often", "quietly", "slowly"),
}
FRAMES = (("det", "subj", "verb", "det", "obj"),
          ("det", "subj", "verb", "det", "obj", "prep", "det", "place"),
          ("det", "subj", "verb", "adv"),
          ("subj", "verb", "det", "obj"))

def segment(tape: str, frame: tuple[str, ...], pos=0, out=()):
    if len(out) == len(frame):
        return [out] if pos == len(tape) else []
    hits = []
    for word in POOLS[frame[len(out)]]:
        if tape.startswith(word, pos): hits += segment(tape, frame, pos + len(word), out + (word,))
    return hits

def run():
    exact = []
    generated = 0
    for frame in FRAMES:
        for words in itertools.product(*(POOLS[r] for r in frame)):
            generated += 1
            left = " ".join(words)
            rev = normalize_letters(left)[::-1]
            for right_frame in FRAMES:
                for right in segment(rev, right_frame):
                    text = left.capitalize() + "; " + " ".join(right) + "."
                    tape = normalize_letters(text)
                    if len(tape) < 39 or tape != tape[::-1]: continue
                    checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
                    exact.append({"text": text, "letters": len(tape), "left_frame": frame,
                                  "right_frame": right_frame, "left_words": words,
                                  "right_words": right, "mechanical_checks": checks,
                                  "mechanically_eligible": all(checks.values())})
    unique = {r["text"]: r for r in exact}
    return {"status": "complete", "config": {"frames": FRAMES, "minimum_letters": 39,
            "operator": "reverse left clause tape, independently segment right typed clause",
            "catalogue_text": False}, "generated_left_clauses": generated,
            "exact_survivors": len(unique), "mechanically_eligible": sum(r["mechanically_eligible"] for r in unique.values()),
            "rendered_exact_survivors": list(unique.values()),
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "material": "small authored common-word pools; no sentence corpus or known palindrome units"},
            "reader_gate": "Mechanical eligibility is not a readability claim; any survivor requires blinded intact-prose review."}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): ap.error("refusing to overwrite output")
    result = run(); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("generated_left_clauses", "exact_survivors", "mechanically_eligible")}, indent=2))
