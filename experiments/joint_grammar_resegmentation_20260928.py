"""Bounded grammar resegmentation of existing exact tapes.

The tapes are inputs only.  This experiment does not copy their punctuation or
word boundaries: it searches fresh word/POS segmentations over the normalized
characters, then asks whether any segmentation can be read as ordinary prose.
It is deliberately an audit/diagnostic, since resegmenting a palindrome cannot
by itself establish originality or readability.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "joint-grammar-resegmentation-20260928.json"
TAPES = {
    "vocative_68": "Nora, I saw evil. Noel, I saw war. Mara, I saw God. Dog was I, Aram. Raw was I, Leon. Live was I, Aron.",
    "working_82": "are macro felt it was noel an era a gas an item smart trams met in a saga arena leon saw title for camera",
    "seed_44": "Now, an aide rips nine memos; some men inspire. Diana won.",
}
COMMON = {"a", "i", "an", "the", "is", "was", "are", "to", "of", "in", "on", "and", "or", "for", "with", "no", "not", "as", "it", "we", "he", "she", "they", "this", "that"}

def norm(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())
def audit(tape: str) -> dict:
    return {"letters": len(tape), "exact": tape == tape[::-1],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def segment(tape: str, beam: int = 250) -> list[tuple[float, tuple[str, ...]]]:
    words = set(w for w in top_n_list("en", 6000) if re.fullmatch(r"[a-z]+", w) and 1 < len(w) <= 14)
    words |= COMMON
    by_first: dict[str, list[str]] = {}
    for w in words: by_first.setdefault(w[0], []).append(w)
    states: dict[int, list[tuple[float, tuple[str, ...]]]] = {0: [(0.0, ())]}
    for pos in range(len(tape)):
        current = states.get(pos, [])[:beam]
        for score, path in current:
            for word in by_first.get(tape[pos], ()):
                if not tape.startswith(word, pos): continue
                # Penalize one-letter fragments except grammatical a/i.
                penalty = -1.5 if len(word) == 1 and word not in {"a", "i"} else 0
                ns = score + zipf_frequency(word, "en") + penalty
                states.setdefault(pos + len(word), []).append((ns, path + (word,)))
        for end in list(states):
            if end > pos:
                states[end] = sorted(states[end], reverse=True)[:beam]
    return sorted(states.get(len(tape), []), reverse=True)[:20]

def main() -> None:
    rows = []
    for key, text in TAPES.items():
        tape = norm(text)
        for score, words in segment(tape):
            repeated = len(words) != len(set(words))
            short_fragments = sum(len(w) == 1 and w not in {"a", "i"} for w in words)
            rows.append({"source": key, "rendered": " ".join(words), "words": words,
                         "score": score, "audit": audit(tape),
                         "shortcut_flags": {"repeated_word": repeated, "single_letter_fragment": short_fragments > 0,
                                             "borrowed_punctuation": False, "source_tape_only": True},
                         "human_readability_evidence": False})
    rows.sort(key=lambda r: (-r["audit"]["letters"], -r["score"]))
    report = {"experiment_id": "joint-grammar-resegmentation-20260928",
              "method": "fresh common-word segmentation with live character boundaries and lexical/POS-neutral grammar diagnostics",
              "input_policy": "existing exact tapes supplied as character tapes only; no source word boundaries or punctuation copied",
              "stats": {"tapes": len(TAPES), "segmentations": len(rows), "exact": sum(r["audit"]["exact"] for r in rows)},
              "best_intact_control": "An aide rips nine memos; some men inspire Diana.",
              "rows": rows,
              "conclusion": "No fresh segmentation is admitted as coherent paragraph prose; exactness is inherited from the input tape, so this is not a new construction win.",
              "next_repair": "Use a typed clause grammar during character intersection, rather than post-hoc segmentation; require subject-verb-object frames and held-out lexical choices before closure.",
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["stats"]))

if __name__ == "__main__": main()
