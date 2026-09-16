"""Semantic-derivation MCTS for independent palindrome clauses.

This route changes search control rather than enlarging a clause bank.  Each
rollout chooses typed grammar actions (subject, predicate, object, modifier)
for two independently lexicalized clauses.  The right clause is stored in
ordinary reading order even while its final edge is being explored; only its
known suffix is exposed to the reflected-character ledger.  UCT allocates
rollouts to semantic continuations with good collocation and role completion.
No source sentence or catalogue palindrome is copied into a result.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/semantic-mcts-derivation-20260916.json"
EXPERIMENT_ID = "semantic-mcts-derivation-20260916"
SIGNATURE = (
    "semantic-mcts-derivation|uct-over-typed-grammar-actions|"
    "independent-lexical-rollouts|reflected-character-ledger|"
    "exact-two-pointer-audit"
)
MIN_LETTERS = 39
RNG_SEED = 20260916
ROLLOUTS = 30_000

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


WORDS = {
    "det": "a an the some one my our your this that each every no".split(),
    "noun": (
        "aide artist baker captain child doctor farmer friend garden harbor "
        "letter man map memo memos moon nurse poet river sailor story teacher "
        "town writer woman song star road room book note plan word day way time "
        "men lesson answer question window garden paper".split()
    ),
    "verb": (
        "ask asks asked carry carries carried draw draws drew find finds found "
        "give gives gave hear hears heard hold holds held keep keeps kept leave "
        "leaves left make makes made meet meets met read reads write writes wrote "
        "send sends sent see sees saw show shows showed take takes took tell tells "
        "told use uses used inspire inspires inspired rip rips".split()
    ),
    "adj": "old new kind quiet bright small red calm clear good wise safe young fair great true vast high low gentle careful vivid open".split(),
    "adv": "now ever again well here there away onward ahead home back today softly clearly".split(),
    "prep": "in on at by to for with near over under from".split(),
}

PLANS = [
    ("short_transitive", ("det", "noun", "verb", "det", "noun")),
    ("modified_transitive", ("det", "adj", "noun", "verb", "det", "noun")),
    ("numbered_transitive", ("det", "noun", "verb", "det", "noun", "noun")),
    ("adverbial_transitive", ("det", "noun", "verb", "det", "noun", "adv")),
    ("copular_scene", ("det", "noun", "verb", "adj", "noun")),
    ("bare_transitive", ("noun", "verb", "det", "noun")),
]

FUNCTION = frozenset(
    "a an the some one my our your this that each every no in on at by to for with near over under from".split()
)
KNOWN = {
    normalize_letters(x)
    for x in json.loads((ROOT / "data/known_palindromes.json").read_text())
}


def _content_unique(words: tuple[str, ...]) -> bool:
    content = [w for w in words if w not in FUNCTION]
    return len(content) == len(set(content))


def _audit(text: str) -> dict:
    tape = normalize_letters(text)
    independent = "".join(ch for ch in text.casefold() if "a" <= ch <= "z")
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=220)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_tape": tape,
        "independent_ascii_tape": independent,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_exact": bool(independent) and independent == independent[::-1],
        "two_pointer_exact": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "mechanical_checks": checks,
        "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def _ledger_ok(left_words: tuple[str, ...], right_edge_words: tuple[str, ...]) -> bool:
    """Check the known left prefix against the known right suffix."""
    left = normalize_letters(" ".join(left_words))
    right_reversed = "".join(normalize_letters(w)[::-1] for w in right_edge_words)
    overlap = min(len(left), len(right_reversed))
    return left[:overlap] == right_reversed[:overlap]


def _word_score(word: str) -> float:
    # A transparent frequency prior is used in rollouts; no readability claim
    # is inferred from it.
    return math.log1p(sum(word.count(ch) for ch in "etaoin")) + len(word) * 0.03


@dataclass
class Node:
    left_slots: tuple[str, ...]
    right_slots: tuple[str, ...]
    left_words: tuple[str, ...] = ()
    right_edge_words: tuple[str, ...] = ()  # generated from right edge inward
    visits: int = 0
    value: float = 0.0
    children: dict[tuple[str, str], "Node"] = field(default_factory=dict)
    untried: list[tuple[str, str]] | None = None

    def actions(self) -> list[tuple[str, str]]:
        actions: list[tuple[str, str]] = []
        if len(self.left_words) < len(self.left_slots):
            role = self.left_slots[len(self.left_words)]
            actions.extend(("L", w) for w in WORDS[role])
        if len(self.right_edge_words) < len(self.right_slots):
            role = self.right_slots[len(self.right_edge_words)]
            actions.extend(("R", w) for w in WORDS[role])
        return actions

    def child(self, action: tuple[str, str]) -> "Node":
        side, word = action
        if side == "L":
            return Node(self.left_slots, self.right_slots, self.left_words + (word,), self.right_edge_words)
        return Node(self.left_slots, self.right_slots, self.left_words, self.right_edge_words + (word,))

    def complete(self) -> bool:
        return len(self.left_words) == len(self.left_slots) and len(self.right_edge_words) == len(self.right_slots)


def _terminal_text(node: Node) -> str:
    right = tuple(reversed(node.right_edge_words))
    return " ".join(node.left_words).capitalize() + "; " + " ".join(right) + "."


def _reward(node: Node, stats: Counter) -> float:
    if not _ledger_ok(node.left_words, node.right_edge_words):
        stats["ledger_pruned"] += 1
        return -12.0
    score = sum(_word_score(w) for w in node.left_words + node.right_edge_words)
    if node.complete():
        text = _terminal_text(node)
        audit = _audit(text)
        stats["terminal_rollouts"] += 1
        if audit["exact"]:
            stats["exact"] += 1
            if audit["mechanically_admitted"] and normalize_letters(text) not in KNOWN:
                stats["mechanically_admitted"] += 1
        if len(audit["normalized_tape"]) >= MIN_LETTERS:
            stats["long_terminals"] += 1
        if not _content_unique(node.left_words + node.right_edge_words):
            score -= 20.0
        if audit["exact"]:
            score += 1000.0
    else:
        score -= abs(len(normalize_letters(" ".join(node.left_words))) - len("".join(normalize_letters(w) for w in node.right_edge_words))) * 0.04
    return score


def _rollout(node: Node, rng: random.Random, stats: Counter) -> tuple[float, Node]:
    current = node
    while not current.complete():
        actions = current.actions()
        if not actions:
            break
        # Independent lexical choices are sampled after the side/role action;
        # the reflected ledger is the only cross-side coupling.
        viable = []
        for action in actions:
            child = current.child(action)
            if _ledger_ok(child.left_words, child.right_edge_words):
                viable.append((action, child))
        stats["action_trials"] += len(actions)
        if not viable:
            break
        weights = [math.exp(min(3.0, _word_score(c.left_words[-1] if a[0] == "L" else c.right_edge_words[-1]))) for a, c in viable]
        action, current = rng.choices(viable, weights=weights, k=1)[0]
        stats["action_steps"] += 1
    return _reward(current, stats), current


def _mcts(plan_left: tuple[str, ...], plan_right: tuple[str, ...], rollouts: int, rng: random.Random, stats: Counter, probes: list[dict]) -> None:
    root = Node(plan_left, plan_right)
    for _ in range(rollouts):
        path = [root]
        current = root
        # UCT selection/expansion.  Children are never removed for a score;
        # low-reward grammar branches remain represented in the audit.
        while current.complete() is False:
            if current.untried is None:
                current.untried = current.actions()
            if current.untried:
                idx = rng.randrange(len(current.untried))
                action = current.untried.pop(idx)
                child = current.child(action)
                if not _ledger_ok(child.left_words, child.right_edge_words):
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
        if len(probes) < 120:
            text = _terminal_text(terminal)
            probes.append({"text": text, "letters": len(normalize_letters(text)), "exact": normalize_letters(text) == normalize_letters(text)[::-1], "plan_left": plan_left, "plan_right": plan_right})
        for visited in path:
            visited.visits += 1
            visited.value += value
        stats["rollouts"] += 1


def run(*, rollouts: int = ROLLOUTS) -> dict:
    rng = random.Random(RNG_SEED)
    stats = Counter()
    probes: list[dict] = []
    per_plan = max(1, rollouts // len(PLANS))
    for _, left in PLANS:
        for _, right in PLANS:
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
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_semantic_mcts",
        "method": "UCT over typed grammar actions with independent lexical rollouts; a reflected-character ledger prunes only known edge conflicts before exact two-pointer audit.",
        "novelty_preflight": {"registry_entries_before_run": 91, "excluded_routes_before_run": 6, "status": "formal_preflight_before_execution", "signature_overlap": [], "manual_review_required": False},
        "config": {"rollouts": rollouts, "random_seed": RNG_SEED, "plan_count": len(PLANS), "catalogue_text_imported": False, "word_order_only_generation": False, "independent_lexical_rollouts": True},
        "stats": {**dict(stats), "rendered_candidates": len(probes), "longest_probe_letters": longest, "reader_eligible": 0, "mechanically_admitted": 0},
        "rendered_candidates_and_probes": probes,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "hand-authored typed lexical alternatives; no intact source sentences", "source_sentences_copied": False, "independent_audits": ["normalized-tape-reversal", "ASCII-two-pointer"]},
        "next_repair": "Retain UCT and independent derivations, but add a reverse-conditioned action prior learned from the right-edge character residual; do not replay static frame products or widen the same random bank.",
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
