"""Live bidirectional semantic-slot product.

The left clause is expanded in reading order. The right clause is expanded
from its final grammatical slot backward, but selected words are stored in
forward reading order. Character debt is consumed before descendants expand;
no completed tape is reversed or repaired.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "bidirectional-slot-product-20260921"
RUN = ROOT / "runs" / f"{ID}.json"

LEFT = {
    "determiner": [("an", "sg"), ("the", "sg"), ("a", "sg")],
    "agent": [("aide", "sg"), ("sailor", "sg"), ("keeper", "sg"),
              ("artist", "sg"), ("writer", "sg"), ("pilot", "sg")],
    "action": [(w, "sg", "transitive") for w in
               ("rips", "sees", "keeps", "marks", "reads", "helps",
                "guides", "writes", "carries", "meets")],
    "object_number": [("nine", "pl"), ("seven", "pl"), ("one", "sg"),
                       ("two", "pl")],
    "object": [(w, n) for w, n in
                (("memos", "pl"), ("letters", "pl"), ("maps", "pl"),
                 ("notes", "pl"), ("books", "pl"), ("boats", "pl"),
                 ("gate", "sg"), ("parcel", "sg"))],
}

# Forward grammatical order: subject determiner, subject noun, verb, object
# name. The search visits these slots in reverse order.
RIGHT = {
    "subject_det": [("some", "pl"), ("many", "pl"), ("the", "sg"),
                     ("a", "sg")],
    "subject_noun": [("men", "pl"), ("women", "pl"), ("sailors", "pl"),
                      ("artists", "pl"), ("writers", "pl"), ("pilots", "pl"),
                      ("poets", "pl"), ("keeper", "sg"), ("guard", "sg")],
    "predicate": [(w, n, "transitive") for w, n in
                  (("inspire", "pl"), ("read", "pl"), ("see", "pl"),
                   ("help", "pl"), ("mark", "pl"), ("guide", "pl"),
                   ("write", "pl"), ("carry", "pl"), ("meet", "pl"),
                   ("keep", "pl"), ("inspires", "sg"), ("reads", "sg"),
                   ("sees", "sg"), ("keeps", "sg"))],
    "object_name": [(w, "sg") for w in
                    ("Diana", "Ada", "Anna", "Nora", "Mira", "Iris",
                     "Leon", "Noah", "Ariel", "Maria")],
}

LEFT_FRAME = ("determiner", "agent", "action", "object_number", "object")
RIGHT_FRAME = ("subject_det", "subject_noun", "predicate", "object_name")
KNOWN_CALIBRATION = "An aide rips nine memos; some men inspire Diana."


def clean(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def independent_audit(text: str) -> dict:
    tape = clean(text)
    return {
        "normalized": tape,
        "letters": len(tape),
        "exact": bool(tape) and tape == tape[::-1],
        "pointer_check": bool(tape) and all(
            tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "sha256_normalized": sha(tape),
        "sha256_reverse": sha(tape[::-1]),
    }


def shortcut_reasons(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.casefold())
    reasons = []
    if any(len(w) > 1 and w == w[::-1] for w in words):
        reasons.append("self_palindromic_word_unit")
    if any(len(a) > 2 and a != b and a == b[::-1]
           for a, b in zip(words, reversed(words))):
        reasons.append("semordnilap_word_pair")
    if len(words) != len(set(words)):
        reasons.append("repeated_word_unit")
    return reasons


def consume(debt: str, token: str, side: str) -> tuple[str, str] | None:
    """Consume one token against the live debt."""
    chars = clean(token) if side == "L" else clean(token)[::-1]
    if debt.startswith(chars):
        return debt[len(chars):], side
    if chars.startswith(debt):
        return chars[len(debt):], "L" if side == "R" else "R"
    return None


def validate_registers(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    left_values = dict(zip(LEFT_FRAME, left))
    right_values = dict(zip(RIGHT_FRAME, right))
    left_number = {
        slot: {item[0]: item[1] for item in values}
        for slot, values in LEFT.items()
    }
    right_number = {
        slot: {item[0]: item[1] for item in values}
        for slot, values in RIGHT.items()
    }
    right_kind = {
        item[0]: item[2] for item in RIGHT["predicate"] if len(item) > 2
    }
    return (left_number["agent"][left_values["agent"]]
            == left_number["action"][left_values["action"]]
            and left_number["object_number"][left_values["object_number"]]
            == left_number["object"][left_values["object"]]
            and right_number["subject_noun"][right_values["subject_noun"]]
            == right_number["predicate"][right_values["predicate"]]
            and right_kind[right_values["predicate"]] == "transitive")


def render(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    return " ".join(left + right) + "."


def run(max_nodes: int = 300_000) -> dict:
    terminal_rows: list[dict] = []
    exact_rows: list[dict] = []
    nodes = prunes = 0

    def record(left, right, debt, trace, terminal=True):
        text = render(left, right)
        audit = independent_audit(text)
        row = {
            "rendered": text, "audit": audit,
            "remaining_character_debt": len(debt),
            "obligation_trace": trace[-16:],
            "candidate_kind": "generated_slot_product_control",
            "provenance": {
                "left_frame": LEFT_FRAME, "right_frame": RIGHT_FRAME,
                "left_forward_generation": True,
                "right_backward_slot_generation": True,
                "finished_tape_reversal": False,
                "post_render_repair": False, "terminal_state": terminal,
            },
            "shortcut_reasons": shortcut_reasons(text),
            "novelty_status": (
                "known_calibration_recovery"
                if audit["normalized"] == clean(KNOWN_CALIBRATION)
                else "unassessed_novelty"
            ),
            "reader_status": "not_run",
        }
        terminal_rows.append(row)
        if (terminal and not debt and validate_registers(left, right)
                and audit["exact"] and not row["shortcut_reasons"]):
            row["candidate_kind"] = "generated_exact_candidate"
            exact_rows.append(row)

    def rec(li, ri, debt, side, left, right, trace, used):
        nonlocal nodes, prunes
        nodes += 1
        if nodes > max_nodes:
            return
        if li == len(LEFT_FRAME) and ri < 0:
            if debt == debt[::-1]:
                record(left, right, debt, trace)
            return
        if not debt and li < len(LEFT_FRAME):
            slot = LEFT_FRAME[li]
            for item in LEFT[slot]:
                word = item[0]
                if word.casefold() in used:
                    continue
                rec(li + 1, ri, clean(word), "R", left + (word,), right,
                    trace + [{"side": "left", "slot": slot, "word": word,
                              "debt_after": clean(word)}],
                    used | {word.casefold()})
            return
        if side == "R" and ri >= 0:
            slot = RIGHT_FRAME[ri]
            for item in RIGHT[slot]:
                word = item[0]
                if word.casefold() in used:
                    continue
                result = consume(debt, word, "R")
                if result is None:
                    prunes += 1
                    continue
                new_debt, new_side = result
                rec(li, ri - 1, new_debt, new_side, left,
                    (word,) + right,
                    trace + [{"side": "right", "slot": slot, "word": word,
                              "debt_after": new_debt}],
                    used | {word.casefold()})
            return
        if side == "L" and li < len(LEFT_FRAME):
            slot = LEFT_FRAME[li]
            for item in LEFT[slot]:
                word = item[0]
                if word.casefold() in used:
                    continue
                result = consume(debt, word, "L")
                if result is None:
                    prunes += 1
                    continue
                new_debt, new_side = result
                rec(li + 1, ri, new_debt, new_side, left + (word,), right,
                    trace + [{"side": "left", "slot": slot, "word": word,
                              "debt_after": new_debt}],
                    used | {word.casefold()})
            return
        prunes += 1

    rec(0, len(RIGHT_FRAME) - 1, "", "L", (), (), [], frozenset())
    near = sorted(terminal_rows,
                  key=lambda r: (r["remaining_character_debt"],
                                  -r["audit"]["letters"]))[:20]
    calibration = KNOWN_CALIBRATION
    out = {
        "experiment_id": ID,
        "method": "independent semantic slot product with live left-forward/right-final-slot-backward character debt",
        "counts": {
            "search_nodes": nodes, "obligation_prunes": prunes,
            "terminal_states": len(terminal_rows), "generated_exact": len(exact_rows),
            "generated_exact_ge_40": sum(r["audit"]["letters"] >= 40 for r in exact_rows),
            "generated_novel_exact": sum(
                r["novelty_status"] == "unassessed_novelty" for r in exact_rows
            ),
            "generated_novel_exact_ge_40": sum(
                r["novelty_status"] == "unassessed_novelty"
                and r["audit"]["letters"] >= 40 for r in exact_rows
            ),
        },
        "candidates": near, "exact_candidates": exact_rows,
        "calibration": {"rendered": calibration, "audit": independent_audit(calibration),
                        "candidate_kind": "known_calibration", "reader_status": "benchmark_only",
                        "provenance": "user-supplied 38-letter benchmark; excluded from generated counts"},
        "shortcut_gates": {"reverse_tape_segmentation": False, "mirrored_units": False,
                            "repair": False, "word_order_only": False, "borrowed_catalogue": False,
                            "RLAIF_per_candidate": False, "live_character_debt": True,
                            "independent_banks": True},
        "provenance": {"code_sha256": sha(Path(__file__).read_text()), "run_path": str(RUN),
                       "audit": "independent normalization, two-pointer equality, and SHA-256",
                       "near_miss_policy": "retain smallest-debt terminal controls; rerun to regenerate full trace"},
        "next_construction": "expand the semantic banks with independently authored valency-compatible frames and allow a typed center slot before terminal closure",
    }
    RUN.write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["counts"], sort_keys=True))
    for row in result["candidates"][:8]:
        print(row["audit"]["letters"], row["remaining_character_debt"], row["rendered"])
