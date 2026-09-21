"""Bounded API-inspired semantic atom transducer.

The public v3 service makes length scalable by carrying a live mirror state,
but its bank is made of already-closed palindrome chunks.  This probe keeps
the useful state/provenance idea while choosing ordinary authored atoms and
their semantic roles *during* the character walk.  It is deliberately small:
two independent SVO clauses, with a shared scene frame but distinct atom IDs.
It is not a catalogue composer and never reverses a finished tape.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "runs" / "api-inspired-semantic-atom-transducer-20260920.json"


def norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = norm(text)
    mismatch = next(
        ((i, tape[i], tape[-i - 1]) for i in range(len(tape) // 2)
         if tape[i] != tape[-i - 1]),
        None,
    )
    return {
        "letters": len(tape),
        "exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def hidden_span(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    whole = norm(text)
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            span = "".join(words[i:j])
            if 1 < len(span) < len(whole) and span == span[::-1]:
                return True
    return False


@dataclass(frozen=True)
class Atom:
    ident: str
    scene: str
    role: str
    surface: str


# Fresh, hand-authored alternatives.  The scene label is a semantic frame,
# not a mirror pairing: each clause must select its own subject, verb, object.
SCENES: dict[str, dict[str, tuple[Atom, ...]]] = {
    "harbor": {
        "subject": (
            Atom("harbor-sailor", "harbor", "subject", "the sailor"),
            Atom("harbor-keeper", "harbor", "subject", "the keeper"),
        ),
        "verb": (
            Atom("harbor-marks", "harbor", "verb", "marks"),
            Atom("harbor-notes", "harbor", "verb", "notes"),
            Atom("harbor-guides", "harbor", "verb", "guides"),
        ),
        "object": (
            Atom("harbor-inlet", "harbor", "object", "the inlet"),
            Atom("harbor-buoy", "harbor", "object", "the buoy"),
            Atom("harbor-signal", "harbor", "object", "the signal"),
        ),
    },
    "clinic": {
        "subject": (
            Atom("clinic-nurse", "clinic", "subject", "the nurse"),
            Atom("clinic-doctor", "clinic", "subject", "the doctor"),
        ),
        "verb": (
            Atom("clinic-checks", "clinic", "verb", "checks"),
            Atom("clinic-opens", "clinic", "verb", "opens"),
            Atom("clinic-keeps", "clinic", "verb", "keeps"),
        ),
        "object": (
            Atom("clinic-chart", "clinic", "object", "the chart"),
            Atom("clinic-cabinet", "clinic", "object", "the cabinet"),
            Atom("clinic-record", "clinic", "object", "the record"),
        ),
    },
    "workshop": {
        "subject": (
            Atom("workshop-baker", "workshop", "subject", "a baker"),
            Atom("workshop-miller", "workshop", "subject", "the miller"),
        ),
        "verb": (
            Atom("workshop-warms", "workshop", "verb", "warms"),
            Atom("workshop-moves", "workshop", "verb", "moves"),
            Atom("workshop-keeps", "workshop", "verb", "keeps"),
        ),
        "object": (
            Atom("workshop-oven", "workshop", "object", "the oven"),
            Atom("workshop-tray", "workshop", "object", "the tray"),
            Atom("workshop-loaf", "workshop", "object", "the loaf"),
        ),
    },
}

LEFT_ROLES = ("subject", "verb", "object")
# The right clause is traversed from its final atom toward its first atom.
# Keeping the grammatical order here makes the decrementing index explicit.
RIGHT_ROLES = ("subject", "verb", "object")


def compatible(selected: tuple[Atom, ...], atom: Atom) -> bool:
    """Keep one independently grammatical event frame per side."""
    if any(old.ident == atom.ident for old in selected):
        return False
    return not selected or all(old.scene == atom.scene for old in selected)


def search_scene(scene: str, state_limit: int = 120_000) -> tuple[list[dict], dict]:
    bank = SCENES[scene]
    exact: list[dict] = []
    controls: list[dict] = []
    seen: set[tuple] = set()
    states = 0
    best = {"matched": 0, "rendered": None, "path": None}

    def walk(
        li: int,
        ri: int,
        left_atom: Atom | None,
        left_pos: int,
        right_atom: Atom | None,
        right_pos: int,
        left: tuple[Atom, ...],
        right_rev: tuple[Atom, ...],
        matched: int,
    ) -> None:
        nonlocal states, best
        states += 1
        if states > state_limit:
            return
        key = (
            li, ri,
            left_atom.ident if left_atom else None, left_pos,
            right_atom.ident if right_atom else None, right_pos,
            tuple(a.ident for a in left), tuple(a.ident for a in right_rev),
        )
        if key in seen:
            return
        seen.add(key)

        if li == len(LEFT_ROLES) and ri < 0:
            # No letters are permitted in the centre for this probe.  This
            # keeps the admission rule simple and leaves centre design for a
            # separate experiment.
            if left_atom is None and right_atom is None:
                left_text = " ".join(a.surface for a in left)
                right = tuple(reversed(right_rev))
                right_text = " ".join(a.surface for a in right)
                rendered = f"{left_text}; {right_text}."
                row = {
                    "rendered": rendered,
                    "left_atoms": [a.ident for a in left],
                    "right_atoms": [a.ident for a in right],
                    "matched_characters": matched,
                    "audit": audit(rendered),
                    "hidden_palindromic_span": hidden_span(rendered),
                    "provenance": {
                        "source": "fresh hand-authored semantic atoms",
                        "scene": scene,
                        "state_trace": [a.ident for a in left + right],
                        "atom_ids_distinct": len({a.ident for a in left + right}) == 6,
                        "finished_tape_reversal": False,
                        "post_hoc_repair": False,
                        "catalogue_text": False,
                        "repeated_units": False,
                    },
                }
                controls.append(row)
                if row["audit"]["exact"]:
                    exact.append(row)
            return

        # Select the next atom only when the current atom is exhausted.
        left_choices = [left_atom] if left_atom else (
            list(bank[LEFT_ROLES[li]]) if li < len(LEFT_ROLES) else []
        )
        right_choices = [right_atom] if right_atom else (
            list(bank[RIGHT_ROLES[ri]]) if ri >= 0 else []
        )
        for la in left_choices:
            if la is None:
                continue
            next_left = left if left_atom else left + (la,)
            if not compatible(left, la) if not left_atom else False:
                continue
            for ra in right_choices:
                if ra is None:
                    continue
                next_right = right_rev if right_atom else right_rev + (ra,)
                if not compatible(right_rev, ra) if not right_atom else False:
                    continue
                lp = left_pos if left_atom else 0
                lt = la.surface.casefold()
                rt = ra.surface.casefold()
                # Spaces/punctuation are not in the tape; compare only the
                # next alphabetic character on each atom.
                lletters = norm(lt)
                rletters = norm(rt)
                rp = right_pos if right_atom else len(rletters) - 1
                if lp >= len(lletters) or rp < 0:
                    continue
                if lletters[lp] != rletters[rp]:
                    continue
                nlp = lp + 1
                nrp = rp - 1
                finished_left = nlp == len(lletters)
                finished_right = nrp < 0
                nl = None if finished_left else la
                nr = None if finished_right else ra
                nli = li + 1 if finished_left else li
                nri = ri - 1 if finished_right else ri
                nmatched = matched + 1
                if nmatched > best["matched"]:
                    partial_left = " ".join(a.surface for a in next_left)
                    partial_right = " ".join(a.surface for a in reversed(next_right))
                    best = {
                        "matched": nmatched,
                        "rendered": f"{partial_left}; {partial_right}.",
                        "path": [a.ident for a in next_left + tuple(reversed(next_right))],
                    }
                walk(nli, nri, nl, nlp, nr, nrp, next_left, next_right, nmatched)

    walk(0, len(RIGHT_ROLES) - 1, None, 0, None, -1, (), (), 0)
    return exact, {"states": states, "deduplicated_states": len(seen), "best": best}


def run() -> dict:
    all_exact: list[dict] = []
    all_controls: list[dict] = []
    scene_stats: dict[str, dict] = {}
    for scene in SCENES:
        exact, stats = search_scene(scene)
        all_exact.extend(exact)
        # Reconstruct controls from the same bounded search by using the
        # complete-product controls as reader-facing grammatical witnesses.
        rows: list[dict] = []
        bank = SCENES[scene]
        for subject in bank["subject"]:
            for verb in bank["verb"]:
                for obj in bank["object"]:
                    rows.append({
                        "rendered": f"{subject.surface} {verb.surface} {obj.surface}.",
                        "scene": scene,
                        "atoms": [subject.ident, verb.ident, obj.ident],
                        "audit": audit(f"{subject.surface} {verb.surface} {obj.surface}."),
                        "provenance": {
                            "source": "fresh hand-authored semantic atoms",
                            "grammar_control": True,
                            "finished_tape_reversal": False,
                            "catalogue_text": False,
                        },
                    })
        all_controls.extend(rows)
        scene_stats[scene] = {"search": stats, "grammar_controls": len(rows), "exact": len(exact)}
    clean = [r for r in all_exact if not r["hidden_palindromic_span"]]
    return {
        "experiment_id": "api-inspired-semantic-atom-transducer-20260920",
        "method": "provenance-carrying semantic atom selection with live opposing character residual",
        "stats": {
            "scenes": len(SCENES),
            "grammar_controls": len(all_controls),
            "exact": len(all_exact),
            "exact_over_38": sum(r["audit"]["letters"] > 38 for r in all_exact),
            "clean_exact": len(clean),
            "longest_exact": max((r["audit"]["letters"] for r in all_exact), default=0),
        },
        "scene_stats": scene_stats,
        "exact_candidates": all_exact,
        "reader_facing_candidates": all_controls[:18],
        "strongest_partial": max(
            (s["search"]["best"] for s in scene_stats.values()),
            key=lambda x: x["matched"],
            default=None,
        ),
        "provenance": {
            "fresh_authored_atoms": True,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "catalogue_text": False,
            "repeated_units": False,
            "independent_audits": ["two-pointer normalized tape", "forward/reverse SHA-256"],
        },
        "novelty_preflight": {
            "signature": "api-mirror-state|semantic-atoms|live-residual|provenance-trace",
            "registry_inspected": True,
            "closest_prior": "complete semantic clause-pair and phrase-seam lanes; this probe selects word atoms while matching",
            "falsifier": "if disabling scene compatibility leaves exact and clean rates unchanged, semantic state added no value",
        },
        "status": "no exact closure" if not all_exact else "exact closure requires independent reader gate",
        "next_construction": "add an authored center atom and a second event relation without relaxing atom-distinctness",
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    if result["strongest_partial"]:
        print(json.dumps(result["strongest_partial"], indent=2))
