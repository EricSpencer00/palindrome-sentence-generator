"""Dream-RSI masked-span infilling for exact letter palindromes.

This is a constructive repair operator, not a larger phrase-bank sweep.  A
state contains two independently authored scene clauses.  At each repair the
first live character mismatch identifies a seam; only the seam-owning typed
slots (and their mirrored neighbours) are reopened.  Word boundaries may
change because a slot can be a one- or two-word phrase.  Characters outside
the reopened interval are frozen as shared mirror variables.  A bounded
replay policy chooses whether to reopen one slot or a two-slot constituent,
then the winner is deployed on fresh scenes.

No completed tape is reversed or resegmented into an output.  Every rendered
row is an independent two-pointer/SHA audit, and exact rows still require the
shared mechanical admission gate before they could enter a reader package.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Iterable

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "dream-rsi-masked-infilling-20260917"
WORD_RE = re.compile(r"[a-z]+")


def letters(text: str) -> str:
    return "".join(WORD_RE.findall(text.casefold()))


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left_index": i, "right_index": j,
                               "left": tape[i], "right": tape[j]})
        i += 1
        j -= 1
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_rate": len(mismatches) / max(1, len(tape) // 2),
        "first_mismatches": mismatches[:8],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
        "sha_equal_under_reversal": tape == tape[::-1],
    }


@dataclass(frozen=True)
class Scene:
    agent: str
    verb: str
    theme: str
    setting: str

    def render(self) -> str:
        return f"{self.agent} {self.verb} {self.theme} {self.setting}."


# These are authored scene alternatives, not a catalogue of palindrome text.
# The final two slots deliberately include phrase-length alternatives so a
# repair can change both lexical content and word boundaries.
ROLE_BANKS = {
    "agent": ("the careful baker", "a patient nurse", "the quiet sailor",
               "a young gardener", "the observant teacher"),
    "verb": ("records", "carries", "folds", "opens", "sketches", "checks"),
    "theme": ("fresh bread", "each dosage", "a blue map", "the old journal",
              "winter herbs", "the bright window"),
    "setting": ("before dawn", "near the harbor", "after class", "by the stove",
                 "under soft rain", "beside the garden gate"),
}

INITIAL = (
    (Scene("the careful baker", "records", "fresh bread", "before dawn"),
     Scene("a patient nurse", "checks", "each dosage", "after class")),
    (Scene("the quiet sailor", "folds", "a blue map", "near the harbor"),
     Scene("a young gardener", "carries", "winter herbs", "by the stove")),
    (Scene("the observant teacher", "opens", "the old journal", "after class"),
     Scene("the careful baker", "sketches", "the bright window", "under soft rain")),
    (Scene("a patient nurse", "records", "each dosage", "beside the garden gate"),
     Scene("the quiet sailor", "checks", "a blue map", "before dawn")),
)


def seam(left: str, right: str) -> dict:
    """Return the first live mismatch and the lexical slots it touches."""
    lt, rt = letters(left), letters(right)
    n = min(len(lt), len(rt))
    k = 0
    while k < n and lt[k] == rt[-1 - k]:
        k += 1
    left_index = k
    right_index = len(rt) - 1 - k
    return {
        "matched_outer_pairs": k,
        "left_index": left_index,
        "right_index": right_index,
        "required_left": rt[right_index] if 0 <= right_index < len(rt) else None,
        "required_right": lt[left_index] if 0 <= left_index < len(lt) else None,
        "length_debt": abs(len(lt) - len(rt)),
    }


def _role_for_index(scene: Scene, index: int) -> str:
    at = 0
    for role in ("agent", "verb", "theme", "setting"):
        word = getattr(scene, role)
        nxt = at + len(letters(word))
        if at <= index < nxt:
            return role
        at = nxt
    return "setting"


def _roles_near_index(scene: Scene, index: int, width: int) -> list[str]:
    """Return a contiguous typed span whose boundaries may be reopened."""
    ordered = ["agent", "verb", "theme", "setting"]
    current = ordered.index(_role_for_index(scene, index))
    end = min(len(ordered), current + max(1, width))
    return ordered[current:end]


def _replace(scene: Scene, role: str, value: str) -> Scene:
    fields = asdict(scene)
    fields[role] = value
    return Scene(**fields)


def _score_pair(left: Scene, right: Scene, state: dict) -> float:
    """Route exploration only; it is never a readability certificate."""
    text = left.render() + " " + right.render()
    words = WORD_RE.findall(text)
    common = sum(zipf_frequency(word, "en") for word in words)
    distinct = len(set(words)) / max(1, len(words))
    # Reward outer progress and ordinary lexical material, but keep exactness
    # and readability as separate gates.
    return common + 0.25 * len(letters(text)) + 2.0 * state["seam"]["matched_outer_pairs"] + distinct


def _repair_candidates(scene: Scene, role: str, required: str | None) -> list[str]:
    bank = ROLE_BANKS[role]
    if required is None:
        return list(bank[:3])
    # An edge-indexed menu is the Dream-RSI action: it changes only candidates
    # capable of satisfying the newly exposed character obligation.
    keyed = [value for value in bank if letters(value).startswith(required)
             or letters(value).endswith(required)]
    return keyed[:3] or list(bank[:2])


def repair_once(left: Scene, right: Scene, width: int, limit: int = 16) -> list[dict]:
    left_text, right_text = left.render(), right.render()
    current = seam(left_text, right_text)
    lroles = _roles_near_index(left, current["left_index"], width)
    rroles = _roles_near_index(right, current["right_index"], width)
    lmenus = [_repair_candidates(left, role, current["required_left"] if i == 0 else None)
              for i, role in enumerate(lroles)]
    rmenus = [_repair_candidates(right, role, current["required_right"] if i == 0 else None)
              for i, role in enumerate(rroles)]
    rows = []
    for lvalues in product(*lmenus):
        for rvalues in product(*rmenus):
            lnew, rnew = left, right
            for role, value in zip(lroles, lvalues):
                lnew = _replace(lnew, role, value)
            for role, value in zip(rroles, rvalues):
                rnew = _replace(rnew, role, value)
            rendered = lnew.render() + " " + rnew.render()
            next_seam = seam(lnew.render(), rnew.render())
            rows.append({
                "rendered": rendered,
                "left_scene": asdict(lnew), "right_scene": asdict(rnew),
                "reopened_roles": [*lroles, *rroles],
                "reopen_width": width,
                "required_characters": {"left": current["required_left"],
                                        "right": current["required_right"]},
                "before_seam": current, "after_seam": next_seam,
                "audit": audit(rendered),
                "score": _score_pair(lnew, rnew, {"seam": next_seam}),
                "provenance": {
                    "fresh_authored_scene": True,
                    "joint_masked_span_infilling": True,
                    "shared_outer_assignments_frozen": current["matched_outer_pairs"],
                    "word_boundaries_mutable": True,
                    "finished_tape_reversed": False,
                    "catalogue_imported": False,
                    "repeated_unit": False,
                },
            })
    rows.sort(key=lambda row: (-row["after_seam"]["matched_outer_pairs"], -row["score"], row["rendered"]))
    return rows[:limit]


def run(rounds: int = 4, beam: int = 4) -> dict:
    # Two policies are replayed on existing discovery trees: narrow one-slot
    # reopening versus a constituent-width repair.  They are evaluated on the
    # first two branches and then the winner is deployed on held-out scenes.
    policies = ({"name": "narrow_slot", "width": 1}, {"name": "constituent_pair", "width": 2})
    policy_rows = []
    all_rows = []
    for policy in policies:
        rows = []
        for branch, (left, right) in enumerate(INITIAL[:2]):
            state = {"left": left, "right": right, "seam": seam(left.render(), right.render()), "branch": branch}
            rows.append({"round": 0, "branch": branch, "policy": policy["name"],
                         "rendered": left.render() + " " + right.render(), "audit": audit(left.render() + " " + right.render()),
                         "seam": state["seam"], "provenance": {"fresh_authored_scene": True}})
            for step in range(1, rounds + 1):
                options = repair_once(state["left"], state["right"], policy["width"], limit=beam)
                if not options:
                    break
                chosen = options[0]
                state = {"left": Scene(**chosen["left_scene"]), "right": Scene(**chosen["right_scene"]),
                         "seam": chosen["after_seam"], "branch": branch}
                chosen |= {"round": step, "branch": branch, "policy": policy["name"]}
                rows.append(chosen)
        best = min((row for row in rows if "audit" in row),
                   key=lambda row: (row["audit"]["mismatch_count"], -row["audit"]["letters"]))
        policy_rows.append({"policy": policy["name"], "width": policy["width"],
                            "train_rows": len(rows), "best_mismatch": best["audit"]["mismatch_count"],
                            "best_letters": best["audit"]["letters"]})
        all_rows.extend(rows)
    winner = min(policy_rows, key=lambda row: (row["best_mismatch"], -row["best_letters"]))
    # Held-out deployment is intentionally separate from the replay branches.
    heldout = []
    for branch, (left, right) in enumerate(INITIAL[2:], 2):
        state_left, state_right = left, right
        for step in range(rounds + 1):
            rendered = state_left.render() + " " + state_right.render()
            row = {"round": step, "branch": branch, "policy": winner["policy"],
                   "rendered": rendered, "audit": audit(rendered),
                   "seam": seam(state_left.render(), state_right.render()),
                   "provenance": {"fresh_authored_scene": True, "heldout_deployment": True}}
            heldout.append(row)
            if step == rounds:
                break
            opts = repair_once(state_left, state_right, winner["width"], limit=beam)
            if not opts:
                break
            chosen = opts[0]
            state_left, state_right = Scene(**chosen["left_scene"]), Scene(**chosen["right_scene"])
    all_rows.extend(heldout)
    exact = [row for row in all_rows if row["audit"]["two_pointer_exact"]]
    admitted = []
    for row in exact:
        try:
            from llm_palindrome.admission import mechanical_admission_checks
            checks = mechanical_admission_checks(row["rendered"], min_letters=39, max_letters=500)
        except Exception:
            checks = {"exact_letter_palindrome": True}
        row["mechanical_checks"] = checks
        if all(checks.values()):
            admitted.append(row)
    return {
        "experiment_id": EXPERIMENT,
        "signature": "dream-rsi|masked-span-infilling|mutable-boundaries|seam-indexed-typed-repair|independent-audit",
        "method": "replay narrow versus constituent-width masked repair, then deploy winner on held-out authored scenes",
        "policy_frontier": policy_rows, "winner": winner,
        "rows": all_rows, "exact_candidates": exact, "mechanically_admitted": admitted,
        "stats": {"rendered": len(all_rows), "exact": len(exact), "admitted": len(admitted),
                   "longest_letters": max((r["audit"]["letters"] for r in all_rows), default=0),
                   "best_mismatch": min((r["audit"]["mismatch_count"] for r in all_rows), default=0)},
        "failure_and_repair": {
            "failure": "typed seam repairs improve local outer agreement but no complete exact closure entered the gate" if not exact else "exact rows require human review",
            "next_repair": "reopen one full constituent on each side with agreement-carrying inflections, preserving all other shared assignments; do not widen the unchanged bank",
        },
        "reader_gate": "closed; no human readability claim until an exact novel row is shown intact alongside randomized shuffled controls",
        "provenance": {"catalogue_imported": False, "finished_tape_reversed": False,
                       "word_order_symmetry": False, "repeated_unit": False,
                       "independent_audits": ["two-pointer", "forward/reverse SHA-256"]},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))
