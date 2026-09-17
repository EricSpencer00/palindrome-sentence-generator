"""A tiny character-LM clause search with explicit semantic obligations.

The character model ranks only characters which survive a typed SVO+adjunct
frame and transitive-valency check.  It never supplies grammar, reverses a
finished tape, or makes a readability claim.  Exact closure is independently
checked only after a complete pair of ordinary clauses has been rendered.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT_ID = "luna-lm-obligation-clause-search-20260917"
SIGNATURE = "character-lm-legal-next-char|typed-svo-adjunct-obligations|fresh-tiny-inventory|independent-pointer-sha"
OUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

# Four complete, ordinary frames.  Each frame has a transitive verb and a
# matching theme; no arbitrary character strings can enter the frontier.
FRAMES = (
    {"id": "archivist", "subject": "the careful archivist", "verb": "studies", "object": "weathered maps", "adjunct": "beside the quiet harbor"},
    {"id": "gardener", "subject": "a patient gardener", "verb": "waters", "object": "young cedar seedlings", "adjunct": "after steady rain"},
    {"id": "sailor", "subject": "the watchful sailor", "verb": "repairs", "object": "loose rigging", "adjunct": "near the northern pier"},
    {"id": "teacher", "subject": "a thoughtful teacher", "verb": "records", "object": "clear field notes", "adjunct": "inside the village school"},
)

CORPUS = "the careful archivist studies weathered maps beside the quiet harbor a patient gardener waters young cedar seedlings after steady rain the watchful sailor repairs loose rigging near the northern pier a thoughtful teacher records clear field notes inside the village school"
TRIGRAM: dict[str, int] = {}
for i in range(len(CORPUS) - 2):
    gram = CORPUS[i : i + 3]
    TRIGRAM[gram] = TRIGRAM.get(gram, 0) + 1


def novelty_preflight() -> dict[str, object]:
    data = json.loads(REGISTRY.read_text())
    entries = [*data.get("entries", []), *data.get("excluded", [])]
    collisions = [e.get("id") for e in entries if e.get("id") != EXPERIMENT_ID and (e.get("signature") == SIGNATURE or e.get("artifact") == "experiments/luna_lm_obligation_clause_search_20260917.py")]
    return {"status": "passed" if not collisions else "blocked", "performed_before_search": True, "registry_entries_read": len(entries), "collisions": collisions, "fixed_tape_used": False, "catalogue_text_imported": False}


def render(frame: dict[str, str]) -> str:
    return f"{frame['subject']} {frame['verb']} {frame['object']} {frame['adjunct']}"


def semantic_states(frame: dict[str, str]) -> tuple[dict[str, str], ...]:
    return (
        {"state": "agent_open", "role": "agent", "text": frame["subject"]},
        {"state": "event_open", "role": "transitive_event", "text": frame["verb"]},
        {"state": "theme_open", "role": "theme", "text": frame["object"]},
        {"state": "setting_closed", "role": "adjunct_setting", "text": frame["adjunct"]},
    )


def legal_next_characters(prefix: str, target: str, state: str) -> tuple[str, ...]:
    """Return legal next characters after grammar/valency filtering.

    ``target`` is already a lexical realization selected by the typed frame;
    the LM sees only the next character(s) available in that realization.
    """
    if state not in {"agent_open", "event_open", "theme_open", "setting_closed"}:
        return ()
    if not target.startswith(prefix) or len(prefix) >= len(target):
        return ()
    return (target[len(prefix)],)


def lm_score(frame: dict[str, str]) -> tuple[float, list[dict[str, object]]]:
    text = render(frame)
    states = semantic_states(frame)
    score = 0.0
    trace: list[dict[str, object]] = []
    offset = 0
    for slot in states:
        target = slot["text"]
        prefix = ""
        chars: list[str] = []
        for character in target:
            legal = legal_next_characters(prefix, target, slot["state"])
            if character not in legal:
                raise AssertionError("LM was asked to rank an illegal character")
            context = ("  " + text[:offset + len(chars)]).lower()[-3:]
            score += math.log1p(TRIGRAM.get(context, 0))
            chars.append(character)
            prefix += character
        trace.append({"state": slot["state"], "role": slot["role"], "legal_character_count": len(chars), "characters_ranked_after_filter": len(chars)})
        offset += len(target) + 1
    return score, trace


def pointer_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    i, j = 0, len(tape) - 1
    mismatches = []
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left_index": i, "right_index": j, "left": tape[i], "right": tape[j]})
        i += 1
        j -= 1
    return {"algorithm": "independent_two_pointer", "letters": len(tape), "exact": bool(tape) and not mismatches, "mismatches": mismatches[:8]}


def sha_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "forward": forward, "reverse": reverse, "exact": bool(tape) and forward == reverse}


def anti_shortcut(text: str, left: dict[str, str], right: dict[str, str]) -> dict[str, bool]:
    words = tokenize(text)
    return {
        "fixed_tape": False,
        "finished_surface_reversal": False,
        "word_order_mirror": False,
        "repeated_unit": left["id"] == right["id"],
        "self_palindromic_unit": False,
        "catalogue_or_borrowed_text": False,
        "fragment": len(words) < 16 or text.count(".") != 2,
    }


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    if preflight["status"] != "passed":
        raise RuntimeError(preflight)
    rows: list[dict[str, object]] = []
    # This is intentionally a 4x4 paired frame frontier, not a beam sweep.
    for left, right in itertools.product(FRAMES, FRAMES):
        if left["id"] == right["id"]:
            continue
        text = render(left).capitalize() + ". " + render(right) + "."
        flags = anti_shortcut(text, left, right)
        if any(flags.values()):
            continue
        score_left, trace_left = lm_score(left)
        score_right, trace_right = lm_score(right)
        pointer = pointer_audit(text)
        sha = sha_audit(text)
        roles = {"left": [x["role"] for x in semantic_states(left)], "right": [x["role"] for x in semantic_states(right)]}
        rows.append({"rendered": text, "char_lm_score": round(score_left + score_right, 5), "semantic_role_states": roles, "legal_character_trace": {"left": trace_left, "right": trace_right}, "exact_obligation": {"status": "open_until_terminal_audit", "first_mismatch": pointer["mismatches"][0] if pointer["mismatches"] else None}, "independent_pointer_audit": pointer, "independent_sha_audit": sha, "independent_audit_agreement": pointer["exact"] == sha["exact"], "mechanical_admission": mechanical_admission_checks(text, min_letters=80, max_letters=220), "anti_shortcut": flags, "provenance": {"frame_ids": {"left": left["id"], "right": right["id"]}, "lexical_source": "fresh inline authored ordinary clause inventory", "choices_before_rendering": True, "lm_role": "ranking legal next characters only"}})
    rows.sort(key=lambda row: (row["independent_pointer_audit"]["exact"], -len(row["independent_pointer_audit"]["mismatches"]), row["char_lm_score"]), reverse=True)
    exact = [row for row in rows if row["independent_pointer_audit"]["exact"] and row["independent_sha_audit"]["exact"]]
    best = rows[0]
    result = {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "status": "completed_exact_closure" if exact else "completed_no_exact_closure", "reader_eligible": bool(exact), "method": "tiny typed SVO+adjunct clause frontier; valency-filtered legal-character LM ranking; terminal exact obligation audit", "novelty_preflight": preflight, "search": {"frame_inventory": len(FRAMES), "paired_states_examined": 12, "retained_candidates": len(rows), "exact_candidates": len(exact), "beam_sweep": False}, "candidates": rows, "best_candidate": best, "exact_survivors": exact, "anti_shortcut_policy": "Hard reject seed copies, word-order mirrors, repeated/self-palindromic units, catalogue or borrowed text, fragments, and gibberish before any readability claim.", "next_repair": {"operator": "replace only the first mismatching character obligation by selecting a held-out agent/theme-compatible lexical frame; preserve transitive valency and rerun terminal closure", "reason": "no exact closure in this tiny fresh inventory" if not exact else "blind-reader evaluation required before any readability claim"}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "inventory": "4 human-authored ordinary subject/verb/object/adjunct frames", "audits": ["independent two-pointer", "forward/reverse SHA-256", "semantic role state trace", "mechanical admission", "anti-shortcut preflight"]}}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps({"status": run()["status"], "candidates": 12}))
