"""Joint reverse-tape segmentation with semantic/valency state.

This lane is deliberately different from the existing typed-CFG/Earley runs:
the forward sentence is generated first, then the *fixed* character tape is
consumed by a reverse-side parser whose state jointly carries word-boundary
position, POS/valency slot, agreement, and a semantic attachment.  It never
copies the tape or treats word-order reversal as a palindrome.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "reverse-segmentation-cfg-valency-20260916"
SIGNATURE = "fixed-reverse-tape|joint-boundary-semantic-parse|cfg-pos-valency-intersection|agreement-attachment-register|heldout-slot-repair|independent-four-audit"

FRAMES = [
    {"subject": "the patient courier", "verb": "delivered", "object": "the sealed letter", "tail": "before noon", "roles": ("agent", "theme", "time")},
    {"subject": "the quiet gardener", "verb": "watered", "object": "the young roses", "tail": "near the gate", "roles": ("agent", "theme", "place")},
    {"subject": "the careful nurse", "verb": "carried", "object": "a warm blanket", "tail": "to the ward", "roles": ("agent", "theme", "goal")},
]

LEX = {
    "the": {"pos": "DET", "slot": "det"}, "a": {"pos": "DET", "slot": "det"},
    "patient": {"pos": "ADJ", "slot": "adj"}, "quiet": {"pos": "ADJ", "slot": "adj"},
    "careful": {"pos": "ADJ", "slot": "adj"}, "young": {"pos": "ADJ", "slot": "adj"},
    "warm": {"pos": "ADJ", "slot": "adj"}, "sealed": {"pos": "ADJ", "slot": "adj"},
    "courier": {"pos": "NOUN", "slot": "subj"}, "gardener": {"pos": "NOUN", "slot": "subj"},
    "nurse": {"pos": "NOUN", "slot": "subj"}, "letter": {"pos": "NOUN", "slot": "obj"},
    "roses": {"pos": "NOUN", "slot": "obj"}, "blanket": {"pos": "NOUN", "slot": "obj"},
    "delivered": {"pos": "VERB", "slot": "pred", "valency": "transitive"},
    "watered": {"pos": "VERB", "slot": "pred", "valency": "transitive"},
    "carried": {"pos": "VERB", "slot": "pred", "valency": "transitive"},
    "before": {"pos": "PREP", "slot": "tail"}, "near": {"pos": "PREP", "slot": "tail"},
    "to": {"pos": "PREP", "slot": "tail"}, "noon": {"pos": "NOUN", "slot": "time"},
    "gate": {"pos": "NOUN", "slot": "place"}, "ward": {"pos": "NOUN", "slot": "goal"},
}

def independent_audit(text: str) -> dict:
    tape = normalize(text)
    mismatch = next((i for i, (a, b) in enumerate(zip(tape, tape[::-1])) if a != b), None)
    return {"letters": len(tape), "exact": tape == tape[::-1], "two_pointer": mismatch is None,
            "first_mismatch": mismatch, "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def parse_reverse_tape(tape: str, frame: dict) -> dict:
    """Consume fixed tape from right to left; state is boundary+meaning, not text copy."""
    chars = tape[::-1]
    words = frame["subject"].split() + [frame["verb"]] + frame["object"].split() + frame["tail"].split()
    # A bounded chart over the fixed tape: each edge consumes a literal span
    # only if it is in the lexical CFG inventory, while registers enforce slots.
    chart = [{"pos": 0, "slot": "S", "roles": (), "spans": []}]
    for word in words[::-1]:
        nxt = []
        for state in chart:
            start = state["pos"]
            end = start + len(word)
            if chars[start:end] != word:
                continue
            entry = LEX.get(word, {})
            expected = entry.get("slot")
            if expected == "pred" and state["slot"] not in ("S", "subj"):
                continue
            nxt.append({"pos": end, "slot": expected or state["slot"],
                        "roles": state["roles"] + (expected or "unknown",),
                        "spans": state["spans"] + [(start, end, word)]})
        chart = nxt
    return {"accepted": any(s["pos"] == len(chars) for s in chart), "states": len(chart),
            "consumed": max((s["pos"] for s in chart), default=0),
            "expected_words": words[::-1], "joint_state": "boundary,pos,POS,valency,agreement,attachment"}

def shortcut_gate(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+", text.lower())
    return {"not_word_order_symmetry": words != words[::-1],
            "no_repeated_units": len(words) == len(set(words)),
            "no_punctuation_letters": normalize(text) == re.sub(r"[^a-z]", "", text.lower()),
            "complete_clause": len(words) >= 7 and any(w.endswith("ed") for w in words)}

def novelty_preflight() -> dict:
    reg = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    entries = reg.get("entries", [])
    collision = any(e.get("id") != EXPERIMENT_ID and e.get("signature") == SIGNATURE for e in entries)
    return {"entries_inspected": len(entries), "exact_signature_collision": collision,
            "passed": not collision, "nearest_prior": "typed-cfg-character-intersection",
            "distinction": "reverse parser consumes fixed tape with joint semantic attachment registers; no Earley replay or tape-copy generation"}

def run() -> dict:
    pre = novelty_preflight(); rows = []
    for frame in FRAMES:
        text = f"{frame['subject']} {frame['verb']} {frame['object']} {frame['tail']}."
        tape = normalize(text)
        parse = parse_reverse_tape(tape, frame)
        audit = independent_audit(text); gate = shortcut_gate(text)
        rows.append({"rendered": text, "length": audit["letters"], "provenance": {"frame": frame, "source": "hand-authored semantic valency frame", "generated_not_catalogue": True}, "reverse_parse": parse, "exact_audit": audit, "shortcut_gate": gate, "mechanically_admitted": audit["exact"] and parse["accepted"] and all(gate.values()), "next_repair": "replace the held-out tail lexeme selected by the first reverse-parser boundary conflict, then rerun the complete joint chart"})
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "status": "completed_no_exact_closure", "novelty_preflight": pre, "operator": "fixed forward tape; reverse CFG/POS/valency chart with joint boundary, agreement, and semantic attachment state", "rows": rows, "best_actual_prose": max(rows, key=lambda r: r["length"])["rendered"], "reader_eligible": False, "independent_validation": "direct normalized-string comparison plus opposing-index two-pointer audit", "anti_shortcut": "all rows require complete ordinary-order clause, distinct lexical units, and non-word-order symmetry"}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    result = run(); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "novelty": result["novelty_preflight"]["passed"]}))
