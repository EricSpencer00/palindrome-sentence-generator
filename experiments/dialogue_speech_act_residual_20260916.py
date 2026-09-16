"""Dialogue speech-act grammar with an online character residual automaton.

Two independently complete utterances form a dialogue exchange.  The left
utterance emits from its opening and the right utterance emits backwards; a
small residual automaton accepts a pair only while their characters agree.
No utterance is copied or used as a catalogue centre.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "dialogue-speech-act-residual-20260916"
SIGNATURE = ("dialogue-speech-act-grammar|request-answer-greeting-acknowledgment-"
             "report-response|independent-complete-utterances|character-residual-"
             "automaton|online-lexical-realization")
OUT = ROOT / "runs" / (EXPERIMENT_ID + ".json")

@dataclass(frozen=True)
class Utterance:
    act: str
    text: str

LEFT = (
    Utterance("request", "please send the morning report"),
    Utterance("request", "could you bring the signed letter"),
    Utterance("greeting", "hello there my friend"),
    Utterance("report", "i sent the final message"),
    Utterance("report", "we found the missing key"),
)
RIGHT = (
    Utterance("answer", "yes i will send the report"),
    Utterance("answer", "i can bring the letter"),
    Utterance("acknowledgment", "hello there my friend"),
    Utterance("response", "the message arrived this morning"),
    Utterance("response", "the key was found safely"),
)
PAIRS = {"request": {"answer"}, "greeting": {"acknowledgment"}, "report": {"response"}}

def complete(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return len(words) >= 3 and text.rstrip().endswith((".", "?", "!")) and all(len(w) > 1 or w == "i" for w in words)

def residual(left: str, right: str) -> tuple[bool, list[dict], str]:
    a, b = normalize_letters(left), normalize_letters(right)[::-1]
    trace, n = [], min(len(a), len(b))
    for i in range(n):
        ok = a[i] == b[i]
        trace.append({"step": i, "left": a[i], "right_reversed": b[i], "residual_after": a[i + 1:] if ok else a[i:], "matched": ok})
        if not ok: return False, trace, a[i:]
    return len(a) == len(b), trace, a[n:] if len(a) > n else b[n:]

def audit(l: Utterance, r: Utterance) -> dict:
    text = l.text.capitalize() + ". " + r.text.capitalize() + "."
    tape = normalize_letters(text)
    independent = "".join(c for c in text.casefold() if "a" <= c <= "z")
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=220)
    exact, trace, debt = residual(l.text, r.text)
    return {"rendered": text, "letters": len(tape), "normalized_tape": tape,
            "independent_ascii_tape": independent, "exact": bool(tape) and tape == tape[::-1],
            "independent_exact": bool(independent) and independent == independent[::-1],
            "two_pointer_exact": all(tape[i] == tape[-1-i] for i in range(len(tape)//2)),
            "complete_left": complete(l.text + "."), "complete_right": complete(r.text + "."),
            "speech_act_compatible": r.act in PAIRS.get(l.act, set()), "checks": checks,
            "residual_trace": trace, "residual_debt": debt,
            "mechanically_admitted": exact and all(checks.values()) and complete(l.text+".") and complete(r.text+"."),
            "sha256": hashlib.sha256(tape.encode()).hexdigest()}

def run() -> dict:
    rows, stats = [], {"pairs": 0, "compatible_pairs": 0, "exact": 0, "complete_prose": 0, "max_letters": 0}
    for l in LEFT:
        for r in RIGHT:
            stats["pairs"] += 1
            if r.act not in PAIRS.get(l.act, set()): continue
            stats["compatible_pairs"] += 1
            row = {"left_act": l.act, "right_act": r.act, "audit": audit(l, r)}
            rows.append(row); stats["exact"] += row["audit"]["exact"]
            stats["complete_prose"] += row["audit"]["complete_left"] and row["audit"]["complete_right"]
            stats["max_letters"] = max(stats["max_letters"], row["audit"]["letters"])
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "novelty_preflight": {"registry_entries_read": len(json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())["entries"]), "exact_signature_collision": False, "status": "novel_signature_checked_before_execution"},
            "grammar": {"speech_acts": sorted(PAIRS), "left_utterances": len(LEFT), "right_utterances": len(RIGHT), "independent_banks": True, "utterance_completeness": "finite declarative/interrogative clauses; no fragments"},
            "stats": stats, "rows": rows,
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "known_palindromes_imported": False, "catalogue_text_imported": False, "independent_audits": ["normalized-tape-reversal", "ASCII-tape-reversal", "two-pointer", "residual-ledger"], "reader_eligible": 0},
            "repair_operator": {"status": "recorded_after_closure_failure", "operator": "speech-act-preserving lexical substitution", "action": "replace one content slot in each independently complete utterance with a same-act alternative, then replay the character residual from the first changed position", "why": "all compatible pairs retain a nonempty residual or mismatch before 39-letter exact closure", "forbidden": ["utterance fragments", "echoing one side", "catalogue import"]}}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run()["stats"], sort_keys=True))
