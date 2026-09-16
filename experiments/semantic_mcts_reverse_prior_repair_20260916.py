"""Reverse-conditioned action-prior repair for semantic MCTS.

The base semantic-derivation MCTS sampled viable actions mostly from a lexical
frequency prior.  This repair leaves its grammar, lexical bank, and independent
left/right derivations unchanged, but scores each rollout action by the number
of newly exposed reflected characters and by the side needed to close the
current edge-length residual.  It is therefore a targeted search-policy repair,
not a new lexical bank or a replay of a static frame product.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/semantic-mcts-reverse-prior-repair-20260916.json"
EXPERIMENT_ID = "semantic-mcts-reverse-prior-repair-20260916"
SIGNATURE = (
    "semantic-mcts-derivation|uct-over-typed-grammar-actions|"
    "reverse-conditioned-action-prior|right-edge-residual-lookahead|"
    "independent-lexical-rollouts|exact-two-pointer-audit"
)
MIN_LETTERS = 39
RNG_SEED = 20260917
ROLLOUTS = 30_000

sys.path.insert(0, str(ROOT))
from experiments import semantic_mcts_derivation_20260916 as base


def _alignment(left_words: tuple[str, ...], right_edge_words: tuple[str, ...]) -> tuple[int, int, int]:
    left = base.normalize_letters(" ".join(left_words))
    right_rev = "".join(base.normalize_letters(w)[::-1] for w in right_edge_words)
    overlap = min(len(left), len(right_rev))
    matched = sum(a == b for a, b in zip(left[:overlap], right_rev[:overlap]))
    mismatched = overlap - matched
    # Positive residual means the left side has exposed more characters and
    # the next useful action is likely a right-edge action, and vice versa.
    return matched, mismatched, len(left) - len(right_rev)


def _prior(child: base.Node, side: str, word: str) -> float:
    matched, mismatched, residual = _alignment(child.left_words, child.right_edge_words)
    lexical = base._word_score(word)
    side_balance = 0.7 if (residual > 0 and side == "R") or (residual < 0 and side == "L") else 0.0
    # A short lookahead rewards a child that extends the currently constrained
    # edge without changing any already matched character.
    return 0.7 * matched - 2.5 * mismatched - 0.04 * abs(residual) + lexical + side_balance


def _rollout(node: base.Node, rng: random.Random, stats: Counter) -> tuple[float, base.Node]:
    current = node
    while not current.complete():
        actions = current.actions()
        if not actions:
            stats["grammar_exhaustion"] += 1
            break
        viable: list[tuple[tuple[str, str], base.Node, float]] = []
        for action in actions:
            child = current.child(action)
            stats["action_trials"] += 1
            if base._ledger_ok(child.left_words, child.right_edge_words):
                viable.append((action, child, _prior(child, action[0], action[1])))
            else:
                stats["ledger_pruned"] += 1
        if not viable:
            break
        # Stable softmax over the residual-aware prior.  It preserves stochastic
        # rollouts while directing effort toward actions that close the live
        # character ledger instead of merely frequent words.
        peak = max(score for _, _, score in viable)
        weights = [math.exp(min(8.0, score - peak)) for _, _, score in viable]
        _, current, _ = rng.choices(viable, weights=weights, k=1)[0]
        stats["action_steps"] += 1
    value = base._reward(current, stats)
    return value, current


def _mcts(plan_left: tuple[str, ...], plan_right: tuple[str, ...], rollouts: int, rng: random.Random, stats: Counter, probes: list[dict]) -> None:
    root = base.Node(plan_left, plan_right)
    for _ in range(rollouts):
        path = [root]
        current = root
        while not current.complete():
            if current.untried is None:
                current.untried = current.actions()
            if current.untried:
                # Expansion order is itself residual-aware, but every action is
                # retained until tried once so this remains an MCTS tree.
                best_idx = max(range(len(current.untried)), key=lambda i: _prior(current.child(current.untried[i]), *current.untried[i]))
                action = current.untried.pop(best_idx)
                child = current.child(action)
                if not base._ledger_ok(child.left_words, child.right_edge_words):
                    stats["ledger_pruned"] += 1
                    continue
                current.children[action] = child
                current = child
                path.append(current)
                break
            if not current.children:
                break
            log_parent = math.log(max(1, current.visits))
            action, current = max(
                current.children.items(),
                key=lambda item: item[1].value / max(1, item[1].visits) + 1.41 * math.sqrt(log_parent / max(1, item[1].visits)),
            )
            path.append(current)
        value, terminal = _rollout(current, rng, stats)
        text = base._terminal_text(terminal)
        if len(probes) < 120:
            tape = base.normalize_letters(text)
            probes.append({"text": text, "letters": len(tape), "exact": bool(tape) and tape == tape[::-1], "complete": terminal.complete(), "plan_left": plan_left, "plan_right": plan_right})
        for visited in path:
            visited.visits += 1
            visited.value += value
        stats["rollouts"] += 1


def run(*, rollouts: int = ROLLOUTS) -> dict:
    rng = random.Random(RNG_SEED)
    stats = Counter()
    probes: list[dict] = []
    per_plan = max(1, rollouts // len(base.PLANS))
    for _, left in base.PLANS:
        for _, right in base.PLANS:
            if stats["rollouts"] >= rollouts:
                break
            _mcts(left, right, min(per_plan, rollouts - stats["rollouts"]), rng, stats, probes)
        if stats["rollouts"] >= rollouts:
            break
    unique = {}
    for row in probes:
        unique.setdefault(row["text"], row)
    probes = sorted(unique.values(), key=lambda row: (-row["exact"], -row["letters"], row["text"]))
    longest = max((row["letters"] for row in probes), default=0)
    exact = [row for row in probes if row["exact"] and row["complete"]]
    admitted = []
    for row in exact:
        audit = base._audit(row["text"])
        row["audit"] = audit
        if audit["mechanically_admitted"] and base.normalize_letters(row["text"]) not in base.KNOWN:
            admitted.append(row)
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_semantic_mcts_reverse_prior_repair",
        "repair_of": "semantic-mcts-derivation-20260916",
        "method": "The semantic MCTS grammar and lexical bank are held fixed; a reverse-conditioned softmax prior scores each independent action by reflected-character lookahead and the side needed to close the edge residual.",
        "novelty_preflight": {"registry_entries_before_run": 92, "excluded_routes_before_run": 6, "status": "formal_preflight_before_execution", "signature_overlap": ["semantic-mcts-derivation-20260916"], "manual_review_required": True, "disposition": "concrete repair of the registered MCTS family"},
        "config": {"rollouts": rollouts, "random_seed": RNG_SEED, "plan_count": len(base.PLANS), "catalogue_text_imported": False, "word_order_only_generation": False, "reverse_conditioned_prior": True, "independent_lexical_rollouts": True},
        "stats": {**dict(stats), "rendered_candidates": len(probes), "exact_complete": len(exact), "longest_probe_letters": longest, "reader_eligible": 0, "mechanically_admitted": len(admitted)},
        "rendered_candidates_and_probes": probes,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "same hand-authored typed lexical alternatives as base family; no intact source sentences", "source_sentences_copied": False, "independent_audits": ["normalized-tape-reversal", "ASCII-two-pointer"]},
        "next_repair": "If this policy still closes only short branches, replace lexical actions with a nonterminal derivation prior that predicts complete semantic roles; do not widen the bank or replay the base random prior.",
        "reader_gate": "No row is reader evidence. Only a novel exact surface that passes mechanical admission may enter randomized blinded intact/shuffled reading.",
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUT}")
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": result["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
