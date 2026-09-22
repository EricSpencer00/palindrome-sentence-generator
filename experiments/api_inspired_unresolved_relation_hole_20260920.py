"""Bilateral grammar frontier with an unresolved lexical relation hole.

This is the next state after the fixed center-relation product.  Each side
chooses subject/verb/object atoms lazily while the connective remains a set of
authored terminals with a known semantic type but unknown letters.  Character
obligations filter that set as soon as either frontier reaches the hole.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from experiments.api_inspired_semantic_atom_transducer_20260920 import (
    SCENES,
    Atom,
    audit,
    hidden_span,
    norm,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "api-inspired-unresolved-relation-hole-20260920.json"
LEFT_ROLES = ("subject", "verb", "object")
RIGHT_ROLES = ("subject", "verb", "object")


@dataclass(frozen=True)
class RelationTerminal:
    ident: str
    surface: str
    semantic_type: str


RELATIONS = (
    RelationTerminal("while", "while", "simultaneous"),
    RelationTerminal("before", "before", "temporal"),
    RelationTerminal("after", "after", "temporal"),
    RelationTerminal("because", "because", "causal"),
    RelationTerminal("despite", "despite", "contrast"),
)


def relation_ok(rel: RelationTerminal, left: tuple[Atom, ...], right: tuple[Atom, ...]) -> bool:
    if len(left) < 3 or len(right) < 3:
        return True
    if rel.semantic_type == "simultaneous":
        return left[1].scene == right[1].scene
    if rel.semantic_type == "temporal":
        return left[1].ident != right[1].ident
    if rel.semantic_type == "causal":
        return left[2].ident != right[2].ident
    if rel.semantic_type == "contrast":
        return left[0].ident != right[0].ident
    return False


def word_letters(atom: Atom) -> str:
    return norm(atom.surface)


def relation_letters(options: tuple[RelationTerminal, ...], pos: int, reverse: bool) -> dict[str, tuple[RelationTerminal, ...]]:
    """Group unresolved terminals by the next character at a cursor."""
    grouped: dict[str, list[RelationTerminal]] = {}
    for rel in options:
        tape = norm(rel.surface)
        i = len(tape) - 1 - pos if reverse else pos
        if 0 <= i < len(tape):
            grouped.setdefault(tape[i], []).append(rel)
    return {k: tuple(v) for k, v in grouped.items()}


def search_scene(scene: str, state_limit: int = 80_000) -> tuple[list[dict], dict]:
    bank = SCENES[scene]
    exact: list[dict] = []
    seen: set[tuple] = set()
    states = 0
    pruned = 0
    best = {"matched": 0, "rendered": None, "reason": "no live character pair"}

    def walk(
        li: int,
        ri: int,
        left_atom: Optional[Atom],
        left_pos: int,
        right_atom: Optional[Atom],
        right_pos: int,
        left: tuple[Atom, ...],
        right_rev: tuple[Atom, ...],
        hole_options: tuple[RelationTerminal, ...],
        hole_len: Optional[int],
        left_hole_pos: int,
        right_hole_pos: int,
        matched: int,
    ) -> None:
        nonlocal states, pruned, best
        states += 1
        if states > state_limit:
            return
        key = (
            li, ri,
            left_atom.ident if left_atom else None, left_pos,
            right_atom.ident if right_atom else None, right_pos,
            tuple(a.ident for a in left), tuple(a.ident for a in right_rev),
            tuple(r.ident for r in hole_options), hole_len,
            left_hole_pos, right_hole_pos,
        )
        if key in seen:
            return
        seen.add(key)

        left_role_done = li == len(LEFT_ROLES) and left_atom is None
        right_role_done = ri < 0 and right_atom is None
        left_done = left_role_done and hole_len is not None and left_hole_pos >= hole_len
        right_done = right_role_done and hole_len is not None and right_hole_pos >= hole_len
        if left_done and right_done and hole_options:
            right = tuple(reversed(right_rev))
            for rel in hole_options:
                rendered = f"{' '.join(a.surface for a in left)} {rel.surface} {' '.join(a.surface for a in right)}."
                row = {
                    "rendered": rendered,
                    "scene": scene,
                    "left_atoms": [a.ident for a in left],
                    "relation_id": rel.ident,
                    "right_atoms": [a.ident for a in right],
                    "matched_characters": matched,
                    "audit": audit(rendered),
                    "hidden_palindromic_span": hidden_span(rendered),
                    "provenance": {
                        "fresh_hand_authored_atoms": True,
                        "unresolved_relation_hole": True,
                        "relation_semantic_type": rel.semantic_type,
                        "relation_filtered_online": True,
                        "atom_ids_distinct": len({a.ident for a in left + right}) == 6,
                        "finished_tape_reversal": False,
                        "post_hoc_repair": False,
                        "catalogue_text": False,
                        "repeated_units": False,
                    },
                }
                if row["audit"]["exact"] and not row["hidden_palindromic_span"]:
                    exact.append(row)
            return
        if (left_role_done and right_role_done and
                ((hole_len is None) or left_hole_pos >= hole_len or right_hole_pos >= hole_len)):
            # One side crossed the unresolved hole without the other; do not
            # invent a center or silently reuse a completed clause.
            pruned += 1
            return

        # Pick a relation length only when a frontier first reaches the hole;
        # its identity remains unresolved and is filtered by characters.
        if hole_len is None and left_role_done and right_role_done:
            lengths = sorted({len(norm(r.surface)) for r in RELATIONS})
            for length in lengths:
                opts = tuple(r for r in RELATIONS if len(norm(r.surface)) == length)
                walk(li, ri, left_atom, left_pos, right_atom, right_pos,
                     left, right_rev, opts, length, 0, 0, matched)
            return

        # Atoms are selected lazily from the next grammar role.
        if left_atom is None and not left_role_done:
            left_choices = list(bank[LEFT_ROLES[li]])
        else:
            left_choices = [left_atom] if left_atom else [None]
        if right_atom is None and not right_role_done:
            right_choices = list(bank[RIGHT_ROLES[ri]])
        else:
            right_choices = [right_atom] if right_atom else [None]

        for la in left_choices:
            if la is None and not left_role_done:
                continue
            next_left = left if left_atom or la is None else left + (la,)
            if la is not None and not left_atom and any(a.ident == la.ident for a in left + tuple(reversed(right_rev))):
                continue
            for ra in right_choices:
                if ra is None and not right_role_done:
                    continue
                next_right = right_rev if right_atom or ra is None else right_rev + (ra,)
                if ra is not None and not right_atom and any(a.ident == ra.ident for a in left + tuple(reversed(right_rev))):
                    continue
                if hole_len is None:
                    # No relation letters are exposed until both grammar
                    # stacks reach the hole; compare ordinary atom letters.
                    lp = left_pos if left_atom else 0
                    rp = right_pos if right_atom else (len(word_letters(ra)) - 1 if ra else -1)
                    ll = word_letters(la) if la else ""
                    rr = word_letters(ra) if ra else ""
                    if not ll or not rr or lp >= len(ll) or rp < 0 or ll[lp] != rr[rp]:
                        pruned += 1
                        continue
                    nl = None if lp + 1 == len(ll) else la
                    nr = None if rp == 0 else ra
                    nli = li + 1 if nl is None else li
                    nri = ri - 1 if nr is None else ri
                    if matched + 1 > best["matched"]:
                        best = {
                            "matched": matched + 1,
                            "rendered": f"{' '.join(a.surface for a in next_left)}; {' '.join(a.surface for a in reversed(next_right))}.",
                            "reason": "live atom seam",
                        }
                    walk(nli, nri, nl, lp + 1 if nl else 0, nr, rp - 1 if nr else -1,
                         next_left, next_right, hole_options, hole_len,
                         left_hole_pos, right_hole_pos, matched + 1)
                else:
                    # Once a side reaches the hole, compare its known letter
                    # to the opposite atom or to the other side's hole.
                    left_in_hole = left_role_done
                    right_in_hole = right_role_done
                    ll = word_letters(la) if la else ""
                    rr = word_letters(ra) if ra else ""
                    lp = left_pos if left_atom else 0
                    rp = right_pos if right_atom else (len(rr) - 1 if ra else -1)
                    lgroups = relation_letters(hole_options, left_hole_pos, False) if left_in_hole else {ll[lp]: hole_options} if la and lp < len(ll) else {}
                    rgroups = relation_letters(hole_options, right_hole_pos, True) if right_in_hole else {rr[rp]: hole_options} if ra and rp >= 0 else {}
                    common = set(lgroups) & set(rgroups)
                    for char in common:
                        opts = tuple(r for r in lgroups[char] if r in rgroups[char])
                        if not opts:
                            continue
                        nl = None if (not left_in_hole and lp + 1 == len(ll)) else la
                        nr = None if (not right_in_hole and rp == 0) else ra
                        nli = li + 1 if (not left_in_hole and nl is None) else li
                        nri = ri - 1 if (not right_in_hole and nr is None) else ri
                        if matched + 1 > best["matched"]:
                            best = {
                                "matched": matched + 1,
                                "rendered": f"{' '.join(a.surface for a in next_left)}; {' '.join(a.surface for a in reversed(next_right))}.",
                                "reason": "relation-hole seam",
                            }
                        walk(nli, nri, nl, lp + 1 if (la and not left_in_hole and nl) else 0,
                             nr, rp - 1 if (ra and not right_in_hole and nr) else -1,
                             next_left, next_right, opts, hole_len,
                             left_hole_pos + int(left_in_hole), right_hole_pos + int(right_in_hole), matched + 1)

    walk(0, len(RIGHT_ROLES) - 1, None, 0, None, -1, (), (), RELATIONS, None, 0, 0, 0)
    return exact, {"states": states, "deduplicated_states": len(seen), "pruned": pruned, "best": best}


def run() -> dict:
    all_exact: list[dict] = []
    scene_stats = {}
    for scene in SCENES:
        exact, stats = search_scene(scene)
        all_exact.extend(exact)
        scene_stats[scene] = stats
    controls = (
        "The sailor marks the inlet while the keeper notes the buoy.",
        "The nurse checks the chart before the doctor opens the cabinet.",
        "A baker warms the oven after the miller moves the tray.",
    )
    return {
        "experiment_id": "api-inspired-unresolved-relation-hole-20260920",
        "method": "lazy bilateral grammar atoms with unresolved semantic relation terminal",
        "stats": {
            "scenes": len(SCENES),
            "exact": len(all_exact),
            "exact_over_38": sum(r["audit"]["letters"] > 38 for r in all_exact),
            "clean_exact": len(all_exact),
            "ordinary_controls": len(controls),
            "states": sum(s["states"] for s in scene_stats.values()),
            "pruned": sum(s["pruned"] for s in scene_stats.values()),
        },
        "scene_stats": scene_stats,
        "exact_candidates": all_exact,
        "ordinary_controls": [{
            "rendered": text,
            "audit": audit(text),
            "provenance": {"fresh_authored_control": True, "reader_eligible": False},
        } for text in controls],
        "reader_facing_candidates": [],
        "provenance": {
            "fresh_authored_inventory": True,
            "relation_chosen_after_character_filter": True,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "catalogue_text": False,
            "repeated_units": False,
            "independent_audits": ["two-pointer normalized tape", "forward/reverse SHA-256"],
        },
        "novelty_preflight": {
            "signature": "api-mirror-state|lazy-grammar-stacks|unresolved-relation-hole|online-filter",
            "registry_inspected": True,
            "distinct_from": ["complete center-relation frame product", "complete semantic seam pair"],
            "falsifier": "if leaving the relation terminal unresolved changes no reachable state versus pre-resolved terminals, retire this representation",
        },
        "status": "no exact closure" if not all_exact else "exact closure requires independent reader gate",
        "next_construction": "permit variable-length grammar continuations after the relation hole while retaining online option filtering",
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
