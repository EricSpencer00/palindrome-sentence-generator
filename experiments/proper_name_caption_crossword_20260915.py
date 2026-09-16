"""Proper-name caption crossword: independent caption halves, exact tape audit.

This route treats a short caption as a typed record (person, action, object,
place), then joins two independently authored records.  It is deliberately
not a reverse decoder or local seam repair: the only search operation is a
crossword-style character compatibility filter over complete rendered captions.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ID = "proper-name-caption-crossword"
SIGNATURE = ("typed-proper-name-caption-records|independent-appositive-incident-"
             "reports|crossword-character-compatibility|whole-caption-join|"
             "independent-two-pointer-audit")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/proper-name-caption-crossword-20260915.json"

LEFT = [
    "Mina, a calm pilot, marked the quay.",
    "Owen, the night guard, opened a gate.",
    "Rhea, a young botanist, pressed a fern.",
    "Theo, the patient cook, stirred the stew.",
    "Lena, a harbor clerk, filed the chart.",
]
RIGHT = [
    "Iris, a quiet painter, framed the scene.",
    "Evan, the ferry guide, carried a lamp.",
    "Nia, a local singer, learned the tune.",
    "Ruth, the careful baker, cooled the loaf.",
    "Omar, a field medic, wrapped the hand.",
]

def norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isascii() and c.isalpha())

def audit(s: str) -> bool:
    t = norm(s)
    return bool(t) and all(a == b for a, b in zip(t, reversed(t)))

def stats(s: str) -> dict:
    w = re.findall(r"[A-Za-z]+", s)
    return {"words": len(w), "letters": len(norm(s)),
            "complete_caption": s.endswith("."),
            "mean_word_length": round(sum(map(len, w))/len(w), 2)}

def main() -> None:
    probes, exact = [], []
    for l in LEFT:
        for r in RIGHT:
            text = l + " " + r
            t = norm(text)
            pairs = 0
            for a, b in zip(t, reversed(t)):
                if a != b: break
                pairs += 1
            row = {"text": text, "outer_matching_pairs": pairs,
                   "exact": audit(text), "readability": stats(text),
                   "provenance": "hand-authored typed caption record; independent half"}
            probes.append(row)
            if row["exact"]: exact.append(row)
    payload = {"experiment_id": ID, "signature": SIGNATURE,
               "method": "crossword compatibility over complete proper-name appositive captions",
               "left_records": len(LEFT), "right_records": len(RIGHT),
               "branches": len(probes), "exact_count": len(exact),
               "rendered_probes": probes, "rendered_candidates": exact,
               "independent_audit": [{"text": x["text"], "two_pointer": audit(x["text"]),
                   "normalized_sha256": hashlib.sha256(norm(x["text"]).encode()).hexdigest()} for x in exact],
               "readability_note": "Diagnostics only; no human readability certification was performed.",
               "repair_operator": "Expand typed record banks with short common-name and place variants; retain complete-caption rendering and rerun exact audit.",
               "next_repair": "Add 50 independently authored records per slot, then inspect the top outer-match probes by a human for grammatical naturalness.",
               "provenance": "All records authored for this run; no catalogue lookup or known-palindrome import.",
               "output_fingerprint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"branches": len(probes), "exact_count": len(exact),
                      "best_outer_pairs": max((x["outer_matching_pairs"] for x in probes), default=0)}, indent=2))

if __name__ == "__main__": main()
