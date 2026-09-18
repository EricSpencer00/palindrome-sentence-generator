"""A small constructive repair: add a semantically licensed clause boundary.

The endpoint words ``now``/``won`` are a reverse pair.  Inserting a sentence
boundary after ``inspire`` keeps the mirrored text as intact prose rather than
as an ungrammatical object-verb sequence.
This is a diagnostic candidate, not yet a reader-study promotion: it extends
the current seed and therefore must be tested against the no-wrapper policy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters

CANDIDATE = "Now, an aide rips nine memos; some men inspire. Diana won."

def audit(text: str) -> dict:
    letters = normalize_letters(text)
    return {"text": text, "letters": len(letters),
            "exact": letters == letters[::-1],
            "normalized": letters,
            "provenance": "authored endpoint reverse-pair repair (now/won); seed expanded with new clause"}

if __name__ == "__main__":
    print(audit(CANDIDATE))
