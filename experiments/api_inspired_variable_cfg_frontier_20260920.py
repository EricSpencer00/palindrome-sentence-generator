"""Variable-length CFG frontier under a live opposing-character equation.

The previous atom and relation-hole probes fixed an SVO shape before the
outer walk.  Here each side carries an unresolved grammar stack.  Optional
adjectival noun phrases and prepositional continuations are expanded only as
their next terminal is needed; scene and agreement registers travel with the
same state.  No completed sentence is reversed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "api-inspired-variable-cfg-frontier-20260920.json"
NONTERMINALS = {"S", "NP_S", "VP", "NP_O", "PP"}


@dataclass(frozen=True)
class Token:
    ident: str
    category: str
    surface: str
    scene: str
    number: str = "sg"


TOKENS = (
    Token("harbor-sailor", "SUBJ", "the sailor", "harbor"),
    Token("harbor-keeper", "SUBJ", "the keeper", "harbor"),
    Token("harbor-aide", "SUBJ", "an aide", "harbor"),
    Token("harbor-calm-sailor", "SUBJ_MOD", "the calm sailor", "harbor"),
    Token("harbor-young-keeper", "SUBJ_MOD", "the young keeper", "harbor"),
    Token("harbor-marks", "VERB", "marks", "harbor"),
    Token("harbor-guides", "VERB", "guides", "harbor"),
    Token("harbor-keeps", "VERB", "keeps", "harbor"),
    Token("harbor-inlet", "OBJ", "the inlet", "harbor"),
    Token("harbor-buoy", "OBJ", "the buoy", "harbor"),
    Token("harbor-signal", "OBJ", "the signal", "harbor"),
    Token("harbor-quiet-inlet", "OBJ_MOD", "the quiet inlet", "harbor"),
    Token("harbor-near", "PREP", "near", "harbor"),
    Token("harbor-beside", "PREP", "beside", "harbor"),
    Token("harbor-shore", "PLACE", "the shore", "harbor"),
    Token("harbor-tower", "PLACE", "the tower", "harbor"),
    Token("clinic-nurse", "SUBJ", "the nurse", "clinic"),
    Token("clinic-doctor", "SUBJ", "the doctor", "clinic"),
    Token("clinic-kind-nurse", "SUBJ_MOD", "the kind nurse", "clinic"),
    Token("clinic-checks", "VERB", "checks", "clinic"),
    Token("clinic-opens", "VERB", "opens", "clinic"),
    Token("clinic-keeps", "VERB", "keeps", "clinic"),
    Token("clinic-chart", "OBJ", "the chart", "clinic"),
    Token("clinic-cabinet", "OBJ", "the cabinet", "clinic"),
    Token("clinic-record", "OBJ", "the record", "clinic"),
    Token("clinic-quiet-chart", "OBJ_MOD", "the quiet chart", "clinic"),
    Token("clinic-near", "PREP", "near", "clinic"),
    Token("clinic-under", "PREP", "under", "clinic"),
    Token("clinic-gate", "PLACE", "the gate", "clinic"),
    Token("clinic-yard", "PLACE", "the yard", "clinic"),
    Token("workshop-baker", "SUBJ", "a baker", "workshop"),
    Token("workshop-miller", "SUBJ", "the miller", "workshop"),
    Token("workshop-young-baker", "SUBJ_MOD", "a young baker", "workshop"),
    Token("workshop-warms", "VERB", "warms", "workshop"),
    Token("workshop-moves", "VERB", "moves", "workshop"),
    Token("workshop-keeps", "VERB", "keeps", "workshop"),
    Token("workshop-oven", "OBJ", "the oven", "workshop"),
    Token("workshop-tray", "OBJ", "the tray", "workshop"),
    Token("workshop-loaf", "OBJ", "the loaf", "workshop"),
    Token("workshop-quiet-oven", "OBJ_MOD", "the quiet oven", "workshop"),
    Token("workshop-near", "PREP", "near", "workshop"),
    Token("workshop-beside", "PREP", "beside", "workshop"),
    Token("workshop-mill", "PLACE", "the mill", "workshop"),
    Token("workshop-yard", "PLACE", "the yard", "workshop"),
)


def norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-i - 1]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": forward,
            "sha256_reverse": reverse, "sha_equal": forward == reverse}


def hidden_span(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    whole = norm(text)
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            span = "".join(words[i:j])
            if 1 < len(span) < len(whole) and span == span[::-1]:
                return True
    return False


TOKENS_BY_CATEGORY: dict[str, tuple[Token, ...]] = {}
for token in TOKENS:
    TOKENS_BY_CATEGORY.setdefault(token.category, []).append(token)
TOKENS_BY_CATEGORY = {k: tuple(v) for k, v in TOKENS_BY_CATEGORY.items()}


def productions(symbol: str, reverse: bool) -> tuple[tuple[str, ...], ...]:
    if not reverse:
        return {
            "S": (("NP_S", "VP"),),
            "NP_S": (("SUBJ",), ("SUBJ_MOD",)),
            "VP": (("VERB", "NP_O"), ("VERB", "NP_O", "PP")),
            "NP_O": (("OBJ",), ("OBJ_MOD",)),
            "PP": (("PREP", "PLACE"),),
        }[symbol]
    # The reverse stream consumes the final surface terminal first.
    return {
        "S": (("VP", "NP_S"),),
        "NP_S": (("SUBJ",), ("SUBJ_MOD",)),
        "VP": (("NP_O", "VERB"), ("PP", "NP_O", "VERB")),
        "NP_O": (("OBJ",), ("OBJ_MOD",)),
        "PP": (("PLACE", "PREP"),),
    }[symbol]


@dataclass(frozen=True)
class State:
    left_stack: tuple[str, ...] = ("S",)
    right_stack: tuple[str, ...] = ("S",)
    left_token: Optional[Token] = None
    right_token: Optional[Token] = None
    left_pos: int = 0
    right_pos: int = -1
    left_tokens: tuple[Token, ...] = ()
    right_tokens_rev: tuple[Token, ...] = ()
    left_scene: Optional[str] = None
    right_scene: Optional[str] = None
    left_number: Optional[str] = None
    right_number: Optional[str] = None
    matched: int = 0


def run(state_limit: int = 180_000) -> dict:
    frontier = [State()]
    seen: set[tuple] = set()
    states = 0
    pruned = {"character": 0, "scene": 0, "agreement": 0, "duplicate": 0, "terminal": 0}
    exact: list[dict] = []
    near: list[dict] = []
    best = {"matched": 0, "rendered": None, "reason": "no compatible outer character"}

    def visit(st: State) -> None:
        nonlocal states, best
        if states >= state_limit:
            return
        states += 1
        key = (st.left_stack, st.right_stack,
               st.left_token.ident if st.left_token else None, st.left_pos,
               st.right_token.ident if st.right_token else None, st.right_pos,
               tuple(t.ident for t in st.left_tokens),
               tuple(t.ident for t in st.right_tokens_rev), st.left_scene,
               st.right_scene, st.left_number, st.right_number)
        if key in seen:
            return
        seen.add(key)

        # Expand unresolved grammar continuations before choosing terminals.
        if st.left_token is None and st.left_stack and st.left_stack[0] in NONTERMINALS:
            symbol = st.left_stack[0]
            for rhs in productions(symbol, False):
                visit(State(left_stack=rhs + st.left_stack[1:], right_stack=st.right_stack,
                            right_token=st.right_token, right_pos=st.right_pos,
                            left_tokens=st.left_tokens, right_tokens_rev=st.right_tokens_rev,
                            left_scene=st.left_scene, right_scene=st.right_scene,
                            left_number=st.left_number, right_number=st.right_number,
                            matched=st.matched))
            return
        if st.right_token is None and st.right_stack and st.right_stack[0] in NONTERMINALS:
            symbol = st.right_stack[0]
            for rhs in productions(symbol, True):
                visit(State(left_stack=st.left_stack, right_stack=rhs + st.right_stack[1:],
                            left_token=st.left_token, left_pos=st.left_pos,
                            left_tokens=st.left_tokens, right_tokens_rev=st.right_tokens_rev,
                            left_scene=st.left_scene, right_scene=st.right_scene,
                            left_number=st.left_number, right_number=st.right_number,
                            matched=st.matched))
            return

        if not st.left_stack and not st.right_stack and st.left_token is None and st.right_token is None:
            rendered = f"{' '.join(t.surface for t in st.left_tokens)}; {' '.join(t.surface for t in reversed(st.right_tokens_rev))}."
            row = {"rendered": rendered, "matched": st.matched, "audit": audit(rendered),
                   "hidden_palindromic_span": hidden_span(rendered),
                   "provenance": {"grammar_stacks_unresolved": True,
                                  "left_tokens": [t.ident for t in st.left_tokens],
                                  "right_tokens": [t.ident for t in reversed(st.right_tokens_rev)],
                                  "fresh_authored_inventory": True,
                                  "finished_tape_reversal": False, "post_hoc_repair": False,
                                  "catalogue_text": False, "repeated_units": False}}
            near.append(row)
            if row["audit"]["exact"] and not row["hidden_palindromic_span"]:
                exact.append(row)
            return
        if not st.left_stack or not st.right_stack:
            pruned["terminal"] += 1
            return

        # Select lexical terminals lazily, respecting semantic state.
        left_choices = (TOKENS_BY_CATEGORY[st.left_stack[0]]
                        if st.left_token is None else (st.left_token,))
        right_choices = (TOKENS_BY_CATEGORY[st.right_stack[0]]
                         if st.right_token is None else (st.right_token,))
        for lt in left_choices:
            if st.left_token is None:
                if st.left_scene and lt.scene != st.left_scene:
                    pruned["scene"] += 1; continue
                if any(t.ident == lt.ident for t in st.left_tokens + st.right_tokens_rev):
                    pruned["duplicate"] += 1; continue
                if lt.category == "VERB" and st.left_number and lt.number != st.left_number:
                    pruned["agreement"] += 1; continue
                new_left = st.left_tokens + (lt,)
                left_scene = st.left_scene or lt.scene
                left_number = lt.number if lt.category in {"SUBJ", "SUBJ_MOD"} else st.left_number
                lp = 0
                left_stack = st.left_stack[1:]
            else:
                new_left, left_scene, left_number, lp, left_stack = st.left_tokens, st.left_scene, st.left_number, st.left_pos, st.left_stack
            ll = norm(lt.surface)
            for rt in right_choices:
                if st.right_token is None:
                    if st.right_scene and rt.scene != st.right_scene:
                        pruned["scene"] += 1; continue
                    if (left_scene and rt.scene != left_scene) or any(t.ident == rt.ident for t in new_left + st.right_tokens_rev):
                        pruned["duplicate"] += 1; continue
                    if rt.category == "VERB" and st.right_number and rt.number != st.right_number:
                        pruned["agreement"] += 1; continue
                    new_right = st.right_tokens_rev + (rt,)
                    right_scene = st.right_scene or rt.scene
                    right_number = rt.number if rt.category in {"SUBJ", "SUBJ_MOD"} else st.right_number
                    rp = len(norm(rt.surface)) - 1
                    right_stack = st.right_stack[1:]
                else:
                    new_right, right_scene, right_number, rp, right_stack = st.right_tokens_rev, st.right_scene, st.right_number, st.right_pos, st.right_stack
                rr = norm(rt.surface)
                if lp >= len(ll) or rp < 0 or ll[lp] != rr[rp]:
                    pruned["character"] += 1
                    continue
                matched = st.matched + 1
                if matched > best["matched"]:
                    best = {"matched": matched,
                            "rendered": f"{' '.join(t.surface for t in new_left)}; {' '.join(t.surface for t in reversed(new_right))}.",
                            "reason": "live grammar terminal"}
                visit(State(left_stack=left_stack, right_stack=right_stack,
                            left_token=None if lp + 1 == len(ll) else lt,
                            right_token=None if rp == 0 else rt,
                            left_pos=0 if lp + 1 == len(ll) else lp + 1,
                            right_pos=-1 if rp == 0 else rp - 1,
                            left_tokens=new_left, right_tokens_rev=new_right,
                            left_scene=left_scene, right_scene=right_scene,
                            left_number=left_number, right_number=right_number,
                            matched=matched))

    visit(State())
    return {
        "experiment_id": "api-inspired-variable-cfg-frontier-20260920",
        "method": "lazy variable-length CFG stacks with live opposing-character and agreement/scene constraints",
        "stats": {"states": states, "deduplicated_states": len(seen),
                  "pruned": pruned, "terminal_derivations": len(near),
                  "exact": len(exact), "exact_over_38": sum(r["audit"]["letters"] > 38 for r in exact),
                  "clean_exact": len(exact), "max_matched_prefix": best["matched"]},
        "best_partial": best,
        "rendered_candidates": near[:24],
        "exact_candidates": exact,
        "ordinary_controls": [
            {"rendered": x, "audit": audit(x), "provenance": {"fresh_authored_control": True}}
            for x in ("The sailor marks the inlet near the shore.",
                       "The nurse checks the chart under the gate.",
                       "A baker warms the oven beside the mill.")
        ],
        "reader_facing_candidates": [],
        "provenance": {"fresh_authored_inventory": True,
                       "grammar_expanded_before_terminal_choice": True,
                       "finished_tape_reversal": False, "post_hoc_repair": False,
                       "catalogue_text": False, "repeated_units": False,
                       "independent_audits": ["two-pointer normalized tape", "forward/reverse SHA-256"]},
        "novelty_preflight": {"signature": "api-mirror-state|lazy-cfg-stacks|optional-modifier-pp|scene-agreement-register",
                              "registry_inspected": True,
                              "distinct_from": ["fixed SVO atom seam", "complete clause product", "relation-hole-only frontier"],
                              "falsifier": "if optional grammar expansions never change reachable character depth versus fixed SVO, retire the CFG layer"},
        "status": "no exact closure" if not exact else "exact closure requires independent reader gate",
        "next_construction": "add an authored relative clause production only if it increases live matched depth without introducing fragments",
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))
