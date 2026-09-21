"""Typed pronoun/accessibility successor for executable discourse CSP."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/executable-discourse-pronoun-accessibility-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "executable-discourse-pronoun-accessibility-20260920"
SIGNATURE = "finite-model-precondition|recipient-theme-state|typed-pronoun-accessibility|joint-character-orbit"

WORLD = {
    ("sealed", "map"), ("closed", "cabinet"), ("key_at", "table"), ("flag_down", "flag"),
}
EVENTS = (
    {"id": "send_map", "subject": "archivist", "verb": "sent", "recipient": "courier", "theme": "map",
     "pre": (("sealed", "map"),), "effects": (("has", "courier", "map"),)},
    {"id": "study_map", "subject": "courier", "verb": "studied", "recipient": "archivist", "theme": "map",
     "pre": (("has", "courier", "map"),), "effects": (("read", "courier", "map"),)},
    {"id": "find_key", "subject": "nurse", "verb": "found", "recipient": "keeper", "theme": "key",
     "pre": (("key_at", "table"),), "effects": (("has", "nurse", "key"),)},
    {"id": "open_cabinet", "subject": "nurse", "verb": "opened", "recipient": "keeper", "theme": "cabinet",
     "pre": (("has", "nurse", "key"), ("closed", "cabinet")), "effects": (("open", "cabinet"),)},
    {"id": "raise_flag", "subject": "sailor", "verb": "raised", "recipient": "keeper", "theme": "flag",
     "pre": (("flag_down", "flag"),), "effects": (("up", "flag"),)},
    {"id": "answer_signal", "subject": "keeper", "verb": "answered", "recipient": "sailor", "theme": "signal",
     "pre": (("up", "flag"),), "effects": (("acknowledged", "keeper", "signal"),)},
)
DETS = ("the", "a")
PRONOUNS = {"archivist": "he", "courier": "him", "nurse": "she", "keeper": "him", "sailor": "him", "map": "it", "key": "it", "cabinet": "it", "flag": "it", "signal": "it"}


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    tape = letters(text); i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]: i += 1; j -= 1
    return {"letters": len(tape), "pointer_exact": bool(tape) and i >= j,
            "first_mismatch": None if i >= j else [i, j, tape[i], tape[j]],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
                "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}


def preflight() -> dict:
    d = json.loads(REGISTRY.read_text())
    rows = [x for x in d.get("entries", []) + d.get("excluded", []) if x.get("id") != EXPERIMENT_ID]
    return {"status": "passed" if not any(x.get("signature") == SIGNATURE for x in rows) else "blocked",
            "signature_overlaps": [x["id"] for x in rows if x.get("signature") == SIGNATURE],
            "excluded_routes": ["finished-tape reversal", "post-hoc repair", "catalogue/API text"]}


def apply_effects(facts: set[tuple[str, ...]], effects: tuple[tuple[str, ...], ...]) -> set[tuple[str, ...]]:
    out = set(facts)
    for effect in effects:
        out.add(effect)
        if effect[0] == "open": out.discard(("closed",) + tuple(effect[1:]))
        if effect[0] == "up": out.discard(("flag_down",) + tuple(effect[1:]))
    return out


def join(first: dict, second: dict) -> tuple[bool, str, set[tuple[str, ...]]]:
    if first["id"] == second["id"]:
        return False, "distinct discourse events required", set()
    facts = set(WORLD)
    if any(p not in facts for p in first["pre"]):
        return False, "first event precondition is false", facts
    facts = apply_effects(facts, first["effects"])
    if not any(p in first["effects"] for p in second["pre"]):
        return False, "recipient/theme event has no enabled precondition", facts
    missing = [p for p in second["pre"] if p not in facts]
    return (not missing, "missing second-event precondition" if missing else "", facts)


def accessible_forms(event: dict, mentioned: set[str]) -> list[dict]:
    """Only allow pronouns after their antecedent is mentioned."""
    forms = [{"subject": event["subject"], "recipient": event["recipient"], "theme": event["theme"], "pronouns": []}]
    for role in ("subject", "recipient", "theme"):
        entity = event[role]
        if entity in mentioned:
            for f in list(forms):
                g = dict(f); g[role] = PRONOUNS[entity]; g["pronouns"] = f["pronouns"] + [role]; forms.append(g)
    return forms


def words(event: dict, det: str, form: dict) -> tuple[str, ...]:
    subject, recipient, theme = form["subject"], form["recipient"], form["theme"]
    sp = (subject,) if subject in PRONOUNS.values() else (det, subject)
    rp = (recipient,) if recipient in PRONOUNS.values() else (det, recipient)
    tp = (theme,) if theme in PRONOUNS.values() else (det, theme)
    if event["id"] == "send_map":
        return sp + (event["verb"],) + rp + tp
    return sp + (event["verb"],) + tp


def consume(left: str, right: str, pending: str) -> tuple[bool, str, int]:
    a, b = pending + letters(left), letters(right)[::-1]
    n = min(len(a), len(b))
    return (a[:n] == b[:n], a[n:] if a[:n] == b[:n] else pending, n)


def render(first: dict, second: dict, d1: str, d2: str, f1: dict, f2: dict) -> str:
    return " ".join(words(first, d1, f1)) + ", then " + " ".join(words(second, d2, f2)) + "."


def run() -> dict:
    pre = preflight()
    if pre["status"] != "passed": raise RuntimeError(pre)
    semantic_prunes = character_prunes = expanded = 0
    valid, controls, rows, witnesses = [], [], [], []
    for first in EVENTS:
        for second in EVENTS:
            expanded += 1
            ok, why, facts = join(first, second)
            if not ok:
                semantic_prunes += 1
                if len(witnesses) < 12: witnesses.append({"first": first["id"], "second": second["id"], "witness": why, "roles_unresolved": True})
                continue
            valid.append((first, second, facts))
            for d1 in DETS:
                for d2 in DETS:
                    f1s = accessible_forms(first, set()); f2s = accessible_forms(second, {first["subject"], first["recipient"], first["theme"]})
                    for f1 in f1s:
                      for f2 in f2s:
                        left, right = words(first, d1, f1), tuple(reversed(words(second, d2, f2)))
                        pending = ""; ok_chars = True; checked = 0
                        for lw, rw in zip(left, right):
                            ok_chars, pending, n = consume(lw, rw, pending); checked += n
                            if not ok_chars: character_prunes += 1; break
                        if ok_chars and not pending:
                            text = render(first, second, d1, d2, f1, f2); a = audit(text)
                            rows.append({"rendered": text, "audit": a, "roles": {"first": [first["recipient"], first["theme"]], "second": [second["recipient"], second["theme"]]}, "binding": {"first": f1["pronouns"], "second": f2["pronouns"]}, "characters_checked": checked})
    for first, second, facts in valid[:3]:
        f1 = accessible_forms(first, set())[0]; f2 = accessible_forms(second, {first["subject"], first["recipient"], first["theme"]})[0]
        text = render(first, second, "the", "the", f1, f2)
        controls.append({"rendered": text, "audit": audit(text), "reader_eligible": False,
                         "provenance": {"recipient_theme_state": True, "preconditions_satisfied": True, "catalogue_text": False}})
    exact = [x for x in rows if x["audit"]["pointer_exact"] and x["audit"]["sha256_forward"] == x["audit"]["sha256_reverse"] and x["audit"]["letters"] > 38]
    out = {"experiment_id": EXPERIMENT_ID, "method": "joint executable recipient/theme event CSP with shared character-orbit obligations",
           "config": {"events": len(EVENTS), "determinants": len(DETS), "recipient_theme_before_lexicalization": True, "typed_pronoun_binding_before_lexicalization": True, "post_search_scoring": False},
           "stats": {"event_pairs_expanded": expanded, "semantic_valid_pairs": len(valid), "semantic_prunes": semantic_prunes, "character_prunes": character_prunes, "rendered_candidates": len(rows), "prose_controls": len(controls), "exact_gt38": len(exact), "max_control_letters": max((x["audit"]["letters"] for x in controls), default=0)},
           "rendered_candidates": rows, "prose_controls": controls, "exact_candidates": exact, "semantic_rejection_witnesses": witnesses, "novelty_preflight": pre,
           "provenance": {"independent_audit": ["two-pointer", "forward/reverse SHA-256"], "recipient_theme_roles": True, "typed_pronoun_accessibility": True, "semantic_pruning_before_render": True, "finished_tape_reversal": False, "post_hoc_repair": False, "catalogue_text": False},
           "next_operator": "add discourse focus and agreement features to the accessibility state before lexical emission", "status": "fresh exact >38 requires blinded reading" if exact else "no exact >38 closure; typed binding controls retained"}
    RUN.write_text(json.dumps(out, indent=2) + "\n"); return out


if __name__ == "__main__": print(json.dumps(run(), indent=2))
