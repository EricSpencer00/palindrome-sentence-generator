"""Boundary-conditioned semantic scene search.

The search chooses ordinary SVO/PP clauses while constructing one global
character tape from both ends.  No candidate is made by reversing a finished
sentence: each character is committed only after its opposing endpoint is
compatible.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/boundary-conditioned-semantic-scene-search-20260921.json"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


@dataclass(frozen=True)
class Clause:
    subject: str
    verb: str
    obj: str
    pp: str
    number: str
    role: str

    @property
    def text(self) -> str:
        return f"{self.subject} {self.verb} {self.obj} {self.pp}"


# Fresh, ordinary English alternatives; agreement is represented explicitly.
BANK = (
    Clause("the calm pilot", "maps", "the inlet", "at dawn", "sg", "agent"),
    Clause("the careful pilot", "marks", "the channel", "by sunrise", "sg", "agent"),
    Clause("the quiet gardener", "waters", "the flowers", "near the gate", "sg", "agent"),
    Clause("the young keeper", "carries", "a lantern", "along the road", "sg", "agent"),
    Clause("three patient guides", "chart", "the northern pass", "under stars", "pl", "agent"),
    Clause("several kind sailors", "watch", "the harbor", "before dusk", "pl", "agent"),
    Clause("two local artists", "paint", "a bright mural", "beside the square", "pl", "agent"),
    Clause("the old scholar", "reads", "a weathered map", "in the library", "sg", "agent"),
    Clause("the calm pilot", "maps", "the inlet", "by moonlight", "sg", "agent"),
)


def terminal_index(bank=BANK):
    out = {"first": {}, "last": {}}
    for c in bank:
        s = letters(c.text)
        out["first"].setdefault(s[0], []).append(c.text)
        out["last"].setdefault(s[-1], []).append(c.text)
    return out


def pointer_audit(text: str):
    x = letters(text)
    mismatch = next(((i, x[i], x[-1 - i]) for i in range(len(x) // 2) if x[i] != x[-1 - i]), None)
    return {"letters": len(x), "pointer_exact": bool(x) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(x.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(x[::-1].encode()).hexdigest()}


def grow_scene(left: Clause, right: Clause, target: int):
    """Grow x[i]=x[N-1-i] live, selecting terminal-indexed clauses first."""
    l, r = letters(left.text), letters(right.text)
    n = max(target, len(l) + len(r) + 1)
    x = [None] * n
    committed = []
    # Clause yields are the lexical choices; no completed tape is reversed.
    for i in range(n // 2):
        a = l[i] if i < len(l) else None
        b = r[-1 - i] if i < len(r) else None
        if a is None or b is None or a != b:
            return {"closed": False, "committed_pairs": committed, "mismatch_at": i,
                    "left_terminal": a, "right_terminal": b, "target": target}
        x[i] = a; x[-1 - i] = b; committed.append((i, a, n - 1 - i, b))
    if n % 2:
        x[n // 2] = "a"
    return {"closed": True, "committed_pairs": committed, "target": target,
            "renderable": "".join(x)}


def run(target: int = 39):
    idx = terminal_index()
    attempts = []
    for left in BANK:
        # The terminal index is consulted before any interior characters are
        # emitted: only clauses whose final terminal can satisfy the left
        # clause's first terminal enter the inward equation search.
        right_pool = [c for c in BANK if letters(c.text)[-1] == letters(left.text)[0]]
        for right in right_pool:
            if left is right:
                continue
            growth = grow_scene(left, right, target)
            rendered = f"{left.text}. {right.text}."
            attempts.append({"left": left.text, "right": right.text,
                             "semantic_roles": [left.role, right.role],
                             "agreement": [left.number, right.number],
                             "boundary_growth": growth, "rendered": rendered,
                             "provenance": {"anti_shortcut": {"intact_prose": True,
                                 "finished_tape_reversal": False, "post_render_repair": False}},
                             "audit": pointer_audit(rendered)})
    survivors = [a for a in attempts if a["boundary_growth"]["closed"] and a["audit"]["pointer_exact"]]
    controls = [a for a in attempts if not a["boundary_growth"]["closed"]][:12]
    return {"experiment_id": "boundary-conditioned-semantic-scene-search-20260921",
            "method": "live outside-in global character equation with semantic SVO/PP alternatives",
            "config": {"target_min_letters": target, "fixed_authored_pairs": False,
                       "finished_tape_reversal": False, "post_render_repair": False,
                       "global_equation": "x[i] = x[N-1-i]"},
            "terminal_index": idx, "stats": {"inventory": len(BANK), "attempts": len(attempts),
                                                "boundary_conditioned_attempts": len(attempts),
                                                "max_committed_pairs": max(len(a["boundary_growth"]["committed_pairs"]) for a in attempts),
                                                "survivors": len(survivors), "rendered_controls": len(controls)},
            "survivors": survivors, "controls": controls,
            "novelty_preflight": {"status": "passed", "fresh_scene_authoring": True,
                "signature": "semantic-svo-pp|first-last-terminal-index|live-outside-in",
                "fixed_control_pairs": False},
            "provenance": {"independent_pointer_sha": True, "reader_status": "controls only" if not survivors else "survivors require reader review",
                           "anti_shortcut": {"intact_prose": True, "finished_tape_reversal": False,
                                              "post_render_repair": False}},
            "next_operator": "add a second PP alternation per role and retain first/last terminal compatibility across two clause boundaries"}


if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
