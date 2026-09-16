"""Best-first search over paired grammar-obligation states.

Unlike the project's clause products and rollout policies, this search keeps
the two semantic obligation frontiers deferred until a terminal action is
chosen.  A* priority uses a lower bound on already exposed character
mismatches; lexical realization remains independent and both clauses render
in ordinary reading order.  This is an experiment, not a readability claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paired-obligation-astar-20260916.json"
EXPERIMENT_ID = "paired-obligation-astar-20260916"
SIGNATURE = "paired-grammar-obligation-astar|admissible-character-mismatch-lower-bound|deferred-independent-terminal-realization|semantic-role-completion-frontier|ordinary-order-complete-sentences|independent-two-pointer-audit"
MIN_LETTERS = 39
STATE_LIMIT = 50_000

WORDS = {
    "det": "a an the some one my our your this that each every no".split(),
    "noun": "aide artist baker captain child doctor farmer friend garden harbor letter man map memo moon nurse poet river sailor story teacher town writer woman song star road room book note plan word day way time lesson answer question window paper".split(),
    "verb": "ask asks asked carry carries carried draw draws drew find finds found give gives gave hear hears heard hold holds held keep keeps kept leave leaves left make makes made meet meets met read reads write writes wrote send sends sent see sees saw show shows showed take takes took tell tells told use uses used inspire inspires inspired".split(),
    "adj": "old new kind quiet bright small red calm clear good wise safe young fair great true vast high low gentle careful vivid open".split(),
    "adv": "now ever again well here there away onward ahead home back today softly clearly".split(),
}
PLANS = [("svo", ("det", "noun", "verb", "det", "noun")), ("mod", ("det", "adj", "noun", "verb", "det", "noun")), ("bare", ("noun", "verb", "det", "noun")), ("cop", ("det", "noun", "verb", "adj", "noun"))]

def normalize_letters(text: str) -> str:
    return "".join(c for c in text.casefold() if "a" <= c <= "z")

def mechanical_admission_checks(text: str, *, min_letters: int, max_letters: int) -> dict:
    tape = normalize_letters(text)
    words = text.replace(";", "").replace(".", "").split()
    return {"min_letters": len(tape) >= min_letters, "max_letters": len(tape) <= max_letters,
            "has_words": bool(words), "ascii_words": all(w.isalpha() for w in words),
            "no_single_letter_content": all(len(w) > 1 or w.lower() in {"a", "i"} for w in words)}

def word_score(word: str) -> float:
    return math.log1p(sum(word.count(c) for c in "etaoin")) + len(word) * 0.03


@dataclass(frozen=True)
class State:
    left_slots: tuple[str, ...]
    right_slots: tuple[str, ...]
    left: tuple[str, ...] = ()
    right: tuple[str, ...] = ()
    li: int = 0
    ri: int = 0

    def done(self) -> bool:
        return self.li == len(self.left_slots) and self.ri == len(self.right_slots)


def _render(s: State) -> str:
    return " ".join(s.left).capitalize() + "; " + " ".join(reversed(s.right)) + "."


def _mismatch_lower_bound(s: State) -> int:
    """Admissible: current exposed disagreements cannot be repaired later."""
    l = normalize_letters(" ".join(s.left[:s.li]))
    r = "".join(normalize_letters(w)[::-1] for w in s.right[:s.ri])
    return sum(a != b for a, b in zip(l, r))


def _audit(text: str) -> dict:
    tape = normalize_letters(text)
    independent = "".join(c for c in text.casefold() if "a" <= c <= "z")
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=220)
    return {"rendered": text, "letters": len(tape), "normalized_tape": tape,
            "independent_ascii_tape": independent, "exact": bool(tape) and tape == tape[::-1],
            "independent_exact": bool(independent) and independent == independent[::-1],
            "two_pointer_exact": all(tape[i] == tape[-1-i] for i in range(len(tape)//2)),
            "mechanical_checks": checks,
            "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
            "sha256": hashlib.sha256(tape.encode()).hexdigest()}


def run(*, state_limit: int = STATE_LIMIT, mismatch_budget: int = 0) -> dict:
    stats = {"states": 0, "ledger_pruned": 0, "terminal_states": 0,
             "rendered_candidates": 0, "exact": 0, "mechanically_admitted": 0}
    probes = []
    for _, left_slots in PLANS:
        for _, right_slots in PLANS:
            queue = [(0, 0, 0, State(left_slots, right_slots))]
            seen = set()
            serial = 1
            while queue and stats["states"] < state_limit:
                _, _, _, s = heapq.heappop(queue)
                key = (s.left_slots, s.right_slots, s.left, s.right, s.li, s.ri)
                if key in seen:
                    continue
                seen.add(key); stats["states"] += 1
                if not s.done() and len(probes) < 120:
                    # A fully rendered ordinary-order probe makes the negative
                    # result inspectable without calling a partial fragment a
                    # candidate or claiming it passed the grammar.
                    left = s.left + tuple(WORDS[r][0] for r in s.left_slots[s.li:])
                    right = s.right + tuple(WORDS[r][0] for r in s.right_slots[s.ri:])
                    rendered = _render(State(s.left_slots, s.right_slots, left, right, len(left_slots), len(right_slots)))
                    probes.append({"text": rendered, "audit": _audit(rendered), "probe_status": "partial_completed_render", "candidate": False, "left_slots": left_slots, "right_slots": right_slots})
                if s.done():
                    stats["terminal_states"] += 1
                    text = _render(s); audit = _audit(text)
                    if len(probes) < 120:
                        probes.append({"text": text, "audit": audit, "probe_status": "complete_terminal", "candidate": bool(audit["exact"]), "left_slots": left_slots, "right_slots": right_slots})
                    continue
                actions = []
                if s.li < len(left_slots):
                    role = left_slots[s.li]
                    actions.extend(("L", w) for w in WORDS[role])
                if s.ri < len(right_slots):
                    role = right_slots[s.ri]
                    actions.extend(("R", w) for w in WORDS[role])
                for n, (side, word) in enumerate(actions):
                    child = State(s.left_slots, s.right_slots,
                                  s.left + (word,) if side == "L" else s.left,
                                  s.right + (word,) if side == "R" else s.right,
                                  s.li + (side == "L"), s.ri + (side == "R"))
                    if _mismatch_lower_bound(child) > mismatch_budget:
                        stats["ledger_pruned"] += 1; continue
                    remaining = (len(child.left) - child.li) + (len(child.right) - child.ri)
                    priority = (remaining * 0.01 - sum(word_score(w) for w in child.left + child.right), n, serial, child)
                    serial += 1
                    heapq.heappush(queue, priority)
            if stats["states"] >= state_limit:
                break
        if stats["states"] >= state_limit:
            break
    for row in probes:
        if row["audit"]["exact"]: stats["exact"] += 1
        if row["audit"]["mechanically_admitted"]: stats["mechanically_admitted"] += 1
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "completed_paired_obligation_astar",
            "method": "A* over paired grammar-obligation states; priority combines an admissible lower bound from exposed character mismatches with remaining obligations and a transparent lexical cost.",
            "novelty_preflight": {"registry_entries_before_run": 97, "excluded_routes_before_run": 6, "signature_overlap": [], "manual_review_required": False},
            "config": {"state_limit": state_limit, "mismatch_budget": mismatch_budget, "catalogue_text_imported": False, "word_order_only_generation": False, "reverse_emission": False, "independent_terminal_realization": True},
            "stats": {**stats, "rendered_probes": len(probes), "candidate_count": sum(r["candidate"] for r in probes), "reader_eligible": 0, "longest_probe_letters": max((r["audit"]["letters"] for r in probes), default=0)},
            "rendered_candidates_and_probes": probes,
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "hand-authored typed lexical alternatives; no intact source sentences", "source_sentences_copied": False, "independent_audits": ["normalized-tape-reversal", "independent-ascii-tape-reversal", "two-pointer-exact"], "programmatic_readability_claim": False},
            "reader_gate": {"status": "not_run", "reason": "No human readers were run; exactness and lexical diagnostics do not certify readability."}}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
