"""Mine role-safe lexical additions only at residuals reached by exact search.

The baseline dual-plan solver supplies the grammar and live character debt.
This experiment never widens a slot globally first: a word or multiword
constituent enters a role only if its emitted letters are prefix-compatible
with an observed residual for that role and orientation.  Expanded plans are
then replayed through the same exact solver and central admission gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dual_plan_residual_search_20260914 import (
    PLANS,
    Plan,
    ascii_audit,
    cancel,
    search_pair,
)
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


ROLE_CANDIDATES = {
    "person": (
        "actor", "agent", "clerk", "driver", "leader", "maker", "mother",
        "father", "neighbor", "owner", "player", "reader", "student",
        "visitor", "worker", "young artist", "careful editor", "quiet nurse",
    ),
    "person_pl": (
        "actors", "agents", "clerks", "drivers", "leaders", "makers",
        "mothers", "fathers", "neighbors", "owners", "players", "readers",
        "students", "visitors", "workers", "young artists", "quiet nurses",
    ),
    "thing": (
        "article", "basket", "bicycle", "box", "branch", "chair", "child",
        "cloth", "color", "desk", "easel", "engine", "flower", "garden",
        "glass", "loaf", "model", "mural", "paper", "portrait", "receipt",
        "result", "rose", "seed", "shop", "table", "tool", "window",
        "brief note", "clear map", "fresh loaf", "old letter", "small box",
    ),
    "thing_pl": (
        "articles", "baskets", "bicycles", "boxes", "branches", "chairs",
        "colors", "engines", "flowers", "loaves", "models", "murals",
        "papers", "portraits", "receipts", "results", "roses", "seeds",
        "tables", "tools", "windows", "brief notes", "clear maps",
    ),
    "name": (
        "Ada", "Anna", "Anne", "Ava", "Leon", "Mara", "Nina", "Otis",
        "Ron", "Sara", "Tessa",
    ),
    "vt_sg": (
        "answers", "brings", "builds", "carries", "cleans", "draws",
        "gives", "holds", "leaves", "makes", "marks", "meets", "moves",
        "notices", "paints", "repairs", "shows", "studies", "takes",
        "thanks", "tells", "uses", "visits", "watches",
    ),
    "vt_pl": (
        "answer", "bring", "build", "carry", "clean", "draw", "give",
        "hold", "leave", "make", "mark", "meet", "move", "notice",
        "paint", "repair", "show", "study", "take", "thank", "tell",
        "use", "visit", "watch",
    ),
    "vt_past": (
        "answered", "brought", "built", "carried", "cleaned", "drew",
        "gave", "held", "left", "made", "marked", "met", "moved",
        "noticed", "painted", "repaired", "showed", "studied", "took",
        "thanked", "told", "used", "visited", "watched",
    ),
    "base_vt": (
        "answer", "bring", "build", "carry", "clean", "draw", "give",
        "hold", "leave", "make", "mark", "meet", "move", "notice",
        "paint", "repair", "show", "study", "take", "thank", "tell",
        "use", "visit", "watch",
    ),
    "adj_person": ("brave", "gentle", "helpful", "skilled", "thoughtful"),
    "adj_thing": ("bright", "clean", "detailed", "solid", "useful"),
    "adverb": ("again", "inside", "later", "slowly", "together"),
    "prep": ("after", "before", "beside", "inside", "under"),
}


def trace_pair(left: Plan, right: Plan, *, state_budget: int = 50_000) -> list[dict]:
    """Return unique reachable nonempty residual states for lexical mining."""
    stack = [(0, len(right.roles) - 1, "", 0, 0)]
    seen = set()
    trace = []
    while stack and len(seen) < state_budget:
        li, ri, residual, owner, matched = stack.pop()
        key = (li, ri, residual, owner)
        if key in seen:
            continue
        seen.add(key)
        if residual:
            next_role = right.roles[ri] if owner == 1 and ri >= 0 else (
                left.roles[li] if owner == -1 and li < len(left.roles) else None
            )
            trace.append({"left_plan": left.name, "right_plan": right.name,
                          "left_index": li, "right_index": ri,
                          "residual": residual, "owner": owner,
                          "next_role": next_role, "matched_letters": matched})
        if owner == 0:
            if li < len(left.roles):
                for word in left.pools[li]:
                    stack.append((li + 1, ri, normalize_letters(word), 1, matched))
        elif owner == 1:
            if ri >= 0:
                for word in right.pools[ri]:
                    emitted = normalize_letters(word)[::-1]
                    outcome = cancel(residual, emitted, 1)
                    if outcome is not None:
                        debt, debt_owner = outcome
                        stack.append((li, ri - 1, debt, debt_owner,
                                      matched + min(len(residual), len(emitted))))
        elif li < len(left.roles):
            for word in left.pools[li]:
                emitted = normalize_letters(word)
                outcome = cancel(residual, emitted, -1)
                if outcome is not None:
                    debt, debt_owner = outcome
                    stack.append((li + 1, ri, debt, debt_owner,
                                  matched + min(len(residual), len(emitted))))
    return trace


def compatible_emission(residual: str, word: str, owner: int) -> str | None:
    emitted = normalize_letters(word)[::-1] if owner == 1 else normalize_letters(word)
    outcome = cancel(residual, emitted, owner)
    return None if outcome is None else outcome[0]


def mine_additions(traces: list[dict]) -> tuple[dict[str, tuple[str, ...]], list[dict]]:
    additions: dict[str, set[str]] = defaultdict(set)
    witnesses = []
    seen = set()
    for state in traces:
        role = state["next_role"]
        if role not in ROLE_CANDIDATES:
            continue
        key = (role, state["residual"], state["owner"])
        if key in seen:
            continue
        seen.add(key)
        for word in ROLE_CANDIDATES[role]:
            after = compatible_emission(state["residual"], word, state["owner"])
            if after is None:
                continue
            additions[role].add(word)
            witnesses.append({**state, "candidate": word,
                              "candidate_tape": normalize_letters(word),
                              "residual_after": after})
    return ({role: tuple(sorted(words)) for role, words in additions.items()},
            witnesses)


def expand_plans(additions: dict[str, tuple[str, ...]]) -> tuple[Plan, ...]:
    expanded = []
    for plan in PLANS:
        pools = tuple(
            tuple(dict.fromkeys(pool + additions.get(role, ())))
            for role, pool in zip(plan.roles, plan.pools)
        )
        expanded.append(Plan(plan.name, plan.roles, pools))
    return tuple(expanded)


def run(*, trace_budget: int = 50_000, replay_budget: int = 250_000) -> dict:
    traces = []
    for left in PLANS:
        for right in PLANS:
            traces.extend(trace_pair(left, right, state_budget=trace_budget))
    additions, witnesses = mine_additions(traces)
    expanded = expand_plans(additions)
    exact = []
    replay_states = 0
    exhausted = 0
    for left in expanded:
        for right in expanded:
            rows, stats = search_pair(left, right, state_budget=replay_budget)
            replay_states += stats["states"]
            exhausted += int(stats["budget_exhausted"])
            for row in rows:
                audit = ascii_audit(row["text"])
                checks = mechanical_admission_checks(row["text"], min_letters=39,
                                                      max_letters=180)
                exact.append({**row, "audit": audit, "mechanical_checks": checks,
                              "mechanically_eligible": all(checks.values()),
                              "reader_status": "human-unreviewed"})
    unique = {row["audit"]["normalized"]: row for row in exact}
    eligible = [row for row in unique.values() if row["mechanically_eligible"]]
    return {
        "status": "residual_indexed_lexical_expansion_complete",
        "config": {"trace_budget_per_pair": trace_budget,
                   "replay_budget_per_pair": replay_budget,
                   "plan_count": len(expanded), "grammar_during_search": True},
        "trace_states": len(traces), "mined_roles": additions,
        "mining_witnesses": witnesses,
        "replay": {"states": replay_states, "budget_exhausted_pairs": exhausted},
        "unique_exact_closures": len(unique), "exact_records": list(unique.values()),
        "eligible_closures": eligible,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "material": "authored role-safe lexical and multiword constituent pools",
                       "selection": "prefix compatibility at observed exact-search residuals only"},
        "next_operator_if_no_novel": "Promote the witnessed multiword residual transitions into optional attachment slots on both plans, then search three-clause discourse states without freezing sentence endpoints.",
        "reader_next": "Novel mechanically eligible outputs require randomized blinded intact-prose versus shuffled-control ratings.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-budget", type=int, default=50_000)
    parser.add_argument("--replay-budget", type=int, default=250_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite output")
    result = run(trace_budget=args.trace_budget, replay_budget=args.replay_budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"trace_states": result["trace_states"],
                      "mining_witnesses": len(result["mining_witnesses"]),
                      "unique_exact_closures": result["unique_exact_closures"],
                      "eligible_closures": len(result["eligible_closures"])}, indent=2))


if __name__ == "__main__":
    main()
