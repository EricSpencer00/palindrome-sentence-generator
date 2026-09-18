"""Dream-RSI-style replay controller for palindrome construction.

Dream-RSI improves the *exploration policy* by replaying recorded discovery
trees; it does not invent a candidate transition inside replay.  This adapter
turns the project's existing run artifacts into replay worlds, compares a
small deterministic policy frontier, and optionally deploys the winner on one
fresh two-region authoring round.  Exactness and readability remain separate
hard gates: replay scores can route work but cannot certify prose.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "dream-rsi-palindrome-20260917.json"
EXPERIMENT_ID = "dream-rsi-palindrome-20260917"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

SKIP_NAMES = {
    "dream-rsi-palindrome-20260917.json",
    "bank-free-sentence-revision-20260917.json",
    "bank-free-sentence-revision-rhythm-20260917.json",
}
EXPLICIT_HISTORY = (
    "runs/two-region-sentence-revision-20260917.json",
    "runs/two-region-sentence-revision-compass-20260917.json",
    "runs/two-region-sentence-revision-compass-anchored-20260917.json",
    "runs/dream-rsi-branching-two-region-20260917.json",
    "runs/dream-rsi-whole-reconstruction-20260917.json",
    "runs/brown-fsa-outer-product-20260917.json",
    "runs/luna-grammar-intersection-fix-20260917.json",
    "runs/derivational-imperative-expansion-20260917.json",
    "runs/agreement-coordinated-clause-search-20260917.json",
    "runs/seam-aware-joint-selection-20260917.json",
    "runs/reversible-word-composition-20260917.json",
    "runs/scene-inflection-lattice-20260917.json",
    "runs/luna-pair-clause-search-20260917.json",
    "runs/boundary-resegmentation-scene-20260917.json",
    "runs/cfg-character-intersection-20260917.json",
    "runs/attachment-valency-seam-csp-20260917.json",
    "runs/luna-semantic-slot-obligation-repair-20260917.json",
    "runs/attachment-valency-repair-20260917.json",
    "runs/earley-seam-lexical-20260917.json",
    "runs/coordinated-seam-chart-20260917.json",
    "runs/phrase-pair-bidirectional-cfg-20260917.json",
    "runs/attachment-equation-expansion-20260917.json",
    "runs/luna-centerout-common-grammar-20260917.json",
    "runs/typed-boundary-trie-20260917.json",
    "runs/semantic-phrase-chain-20260917.json",
    "runs/homophone-clause-seam-weaving-20260917.json",
    "runs/human-centerout-function-verb-noun-csp-20260917.json",
    "runs/cfg-reverse-tape-dp-20260917.json",
    "runs/dream-rsi-online-branching-two-region-20260917.json",
    "runs/lexicon-reverse-edge-chain-20260917.json",
    "runs/direct-joint-authoring-20260917.json",
    "runs/ordinary-phrase-pair-chain-20260917.json",
    "runs/character-cfg-earley-scene-20260917.json",
    "runs/typed-svo-phrase-edges-20260917.json",
    "runs/preclosure-lexical-seam-20260917.json",
    "runs/joint-boundary-first-mismatch-repair-20260917.json",
    "runs/typed-svo-seam-repair-20260917.json",
    "runs/live-role-seam-growth-20260917.json",
    "runs/seed-semantic-pairing-search-20260917.json",
    "runs/reverse-phrase-index-20260917.json",
    "runs/live-typed-chart-seam-solver-20260917.json",
    "runs/semantic-scene-live-equation-20260917.json",
    "runs/cegar-role-product-20260917.json",
    "runs/reverse-pos-phrase-lattice-20260917.json",
    "runs/typed-slot-reverse-trie-repair-20260917.json",
    "runs/multiclause-live-seam-chart-20260917.json",
    "runs/three-clause-center-crossing-20260917.json",
    "runs/dialogue-scene-semantic-palindrome-20260917.json",
    "runs/center-boundary-mirror-search-20260917.json",
    "runs/center-word-boundary-scene-20260917.json",
    "runs/dialogue-scene-lattice-crossing-20260917.json",
    "runs/typed-template-equation-search-20260917.json",
    "runs/imperative-vocative-seedless-20260917.json",
    "runs/seam-indexed-typed-repair-20260917.json",
    "runs/dream-rsi-masked-infilling-20260917.json",
    "runs/dream-rsi-palindrome-round39-20260917.json",
    "runs/two-sided-constituent-agreement-20260917.json",
    "runs/dream-rsi-palindrome-round40-20260917.json",
    "runs/joint-agent-object-masked-infill-20260917.json",
    "runs/dream-rsi-palindrome-round41-20260917.json",
    "runs/relative-clause-boundary-infill-20260917.json",
    "runs/three-region-discourse-anchor-20260917.json",
    "runs/semantic-frame-mirror-20260917.json",
    "runs/three-region-agreement-repair-20260917.json",
    "runs/relative-clause-seam-repair-20260917.json",
    "runs/relative-seam-agreement-20260917.json",
    "runs/two-sided-relative-balance-20260917.json",
    "runs/relative-head-verb-window-20260917.json",
    "runs/relative-seam-two-word-bridge-20260917.json",
    "runs/semantic-frame-seam-replacement-20260917.json",
    "runs/relative-seam-bridge-complement-20260917.json",
    "runs/whole-prose-repair-2026-09-13/pilot-01.json",
)
TEXT_KEYS = ("rendered", "text", "sentence", "surface")
ACTION_PATTERNS = (
    ("two_region", re.compile(r"two.?region|coordinated.*region", re.I)),
    ("whole_passage", re.compile(r"whole.?prose|bank.?free|authoring", re.I)),
    ("semantic", re.compile(r"semantic|valency|scene", re.I)),
    ("grammar", re.compile(r"grammar|cfg|earley|morpholog|agreement", re.I)),
    ("character_product", re.compile(r"character|graph|trie|residual", re.I)),
)


def normalize_letters(text: str) -> str:
    return "".join(char for char in text.casefold() if "a" <= char <= "z")


def independent_audit(text: str) -> dict[str, Any]:
    tape = normalize_letters(text)
    mismatches = sum(tape[i] != tape[-1 - i] for i in range(len(tape) // 2))
    return {
        "letters": len(tape),
        "exact": bool(tape) and tape == tape[::-1],
        "mismatch_count": mismatches,
        "mismatch_rate": mismatches / max(1, len(tape) // 2),
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def _action(source: str, experiment_id: str) -> str:
    material = f"{source} {experiment_id}"
    for label, pattern in ACTION_PATTERNS:
        if pattern.search(material):
            return label
    return "other"


def _text_and_audit(obj: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    text = next((obj.get(key) for key in TEXT_KEYS if isinstance(obj.get(key), str)), None)
    if not text:
        return None, None
    audit = obj.get("audit")
    if not isinstance(audit, dict):
        audit = {}
    if isinstance(obj.get("symmetry_diagnostics"), dict):
        diag = obj["symmetry_diagnostics"]
        audit = {**audit, **diag}
    if isinstance(obj.get("independent_exactness"), dict):
        exactness = obj["independent_exactness"]
        if "direct_symmetric_position_comparison" in exactness:
            audit = {**audit, "exact": exactness["direct_symmetric_position_comparison"]}
    if not any(key in audit for key in ("letters", "letter_count", "normalized_tape")):
        audit = independent_audit(text)
    return text, audit


def _row_is_candidate(obj: dict[str, Any]) -> bool:
    text, audit = _text_and_audit(obj)
    return bool(text and audit is not None)


@dataclass(frozen=True)
class Node:
    node_id: str
    world: str
    source_path: str
    source_location: str
    parent_sha256: str | None
    forward_sha256: str
    action: str
    rendered: str
    letters: int
    exact: bool
    mismatch_count: int
    mismatch_rate: float
    length_ok: bool
    shortcut_free: bool
    intact_surface: bool
    reader_certified: bool
    failure_signature: str


def _shortcut_free(obj: dict[str, Any]) -> bool:
    # Some old artifacts predate the explicit shortcut flags and contain a
    # repeated phrase disguised as a two-sentence "palindrome".  Reject
    # repeated contiguous content here so replay cannot promote catalogue-
    # style scaffolds merely because their metadata is incomplete.
    text, _ = _text_and_audit(obj)
    if text:
        words = [word.casefold() for word in re.findall(r"[A-Za-z]+", text)]
        # A chain of individually self-palindromic words is an exact tape but
        # not the requested construction.  Reject it even when a legacy
        # artifact omitted explicit anti-shortcut metadata.
        content_words = [word for word in words if word not in {
            "a", "an", "the", "i", "we", "you", "he", "she", "it",
            "they", "and", "or", "but", "if", "as", "of", "to", "in",
            "on", "at", "by", "for", "from", "with", "is", "are", "was",
            "were", "be", "been", "not", "no",
        }]
        if any(len(word) > 2 and word == word[::-1] for word in content_words):
            return False
        # Reuse the shared fail-closed lexical/structure gate for legacy rows
        # that lack explicit metadata.  These checks filter obvious fragments
        # and catalogue-like scaffolds; they do not certify human readability.
        try:
            from llm_palindrome.admission import mechanical_admission_checks
            checks = mechanical_admission_checks(
                text, min_letters=1, max_letters=max(10_000, len(normalize_letters(text)))
            )
            hard = (
                "word_form", "lexicon_words", "ordinary_short_words",
                "distinct_words", "no_self_palindromic_word",
                "no_repeated_nontrivial_unit", "no_self_palindromic_proper_multiword_span",
                "not_word_order_symmetry", "not_forbidden_catalogue_control",
                "not_catalogue_family_derivative", "not_forbidden_catalogue_endpoint_scaffold",
                "absent_from_local_catalogue",
            )
            if not all(checks.get(key, False) for key in hard):
                return False
        except (ImportError, OSError, ValueError):
            # Replay remains usable in minimal environments; the independent
            # checks above still reject the most dangerous legacy shortcuts.
            pass
        for width in (3, 4, 5):
            phrases = [tuple(words[index:index + width])
                       for index in range(len(words) - width + 1)]
            if len(phrases) != len(set(phrases)):
                return False
    provenance = obj.get("provenance")
    if isinstance(provenance, dict):
        if any(bool(provenance.get(key)) for key in (
            "catalogue_imported", "borrowed_catalogue_text", "finished_tape_reversed",
        )):
            return False
        # Do not scan the archived prompt under ``metadata``: prompts mention
        # forbidden shortcuts precisely to instruct the authoring model, and
        # that text is not evidence that the rendered row used one.
        provenance_text = " ".join(str(value) for key, value in provenance.items()
                                    if key not in {"catalogue_imported", "finished_tape_reversed", "metadata"}).casefold()
        if any(token in provenance_text for token in ("catalogue", "borrowed", "reversed tape")):
            return False
    flags = obj.get("shortcut_flags")
    if isinstance(flags, dict):
        rejected = {
            # Repeating an ordinary noun (for example, "map") can be
            # grammatical anaphora; contiguous repeated phrases are rejected
            # above.  Do not confuse that legitimate repetition with a copied
            # unit.
            "word_order_mirror", "word_order_only_symmetry", "word_order_only",
            "borrowed_catalogue_text",
            "reader_certified", "catalogue_family_derivative", "repeated_unit",
            "finished_tape_reversed", "fragment",
        }
        return not any(bool(flags.get(key)) for key in rejected)
    anti = obj.get("anti_shortcut")
    if isinstance(anti, dict):
        rejected = {"catalogue_source", "word_order_only", "repeated_unit", "fragment", "fixed_tape"}
        return not any(bool(anti.get(key)) for key in rejected)
    return True


def _failure_signature(obj: dict[str, Any], audit: dict[str, Any]) -> str:
    next_repair = obj.get("next_repair") or obj.get("failure_and_repair")
    if isinstance(next_repair, dict):
        next_repair = next_repair.get("failure") or next_repair.get("next_repair")
    if isinstance(next_repair, str) and next_repair:
        return next_repair[:120]
    mismatches = audit.get("first_mismatches") or audit.get("mismatches")
    if isinstance(mismatches, list) and mismatches:
        first = mismatches[0]
        if isinstance(first, (list, tuple)) and len(first) >= 4:
            if first[2] is not None and first[3] is not None:
                return f"edge:{first[2]}>{first[3]}"
        if isinstance(first, dict):
            left, right = first.get("left_letter"), first.get("right_letter")
            if left is not None and right is not None:
                return f"edge:{left}>{right}"
    return "exact" if bool(audit.get("exact")) else "unclassified"


def _iter_json_files() -> Iterable[Path]:
    # The checkout contains thousands of historical JSON files (including
    # very large raw traces).  Dream-RSI needs a replay world, not an
    # unbounded corpus-ingestion benchmark: use registered run artifacts plus
    # the current prose lineages, and cap individual files so parsing remains
    # a bounded offline phase.
    paths = {ROOT / rel for rel in EXPLICIT_HISTORY}
    try:
        registry = json.loads(REGISTRY.read_text())
        for entry in registry.get("entries", []):
            for rel in entry.get("run_artifacts", []):
                candidate = ROOT / rel
                if candidate.exists() and candidate.stat().st_size <= 3_000_000:
                    paths.add(candidate)
    except (OSError, json.JSONDecodeError):
        pass
    for path in sorted(paths):
        if not path.exists() or path.stat().st_size > 3_000_000:
            continue
        if path.name in SKIP_NAMES:
            continue
        yield path


def load_worlds() -> list[Node]:
    nodes: list[Node] = []
    seen: set[tuple[str, str]] = set()
    for path in _iter_json_files():
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        experiment_id = payload.get("experiment_id", path.stem) if isinstance(payload, dict) else path.stem
        root_repair = None
        if isinstance(payload, dict):
            repair = payload.get("failure_and_repair") or payload.get("next_repair")
            if isinstance(repair, dict):
                repair = repair.get("next_repair") or repair.get("failure")
            if isinstance(repair, str) and repair:
                root_repair = repair[:120]

        def visit(obj: Any, location: str) -> None:
            if isinstance(obj, dict):
                # Kernel fixtures validate the solver but are never replay
                # worlds or generated candidates.  Keep their provenance in
                # the artifact while excluding their rendered text here.
                is_withheld = bool(obj.get("withheld") or obj.get("not_a_generated_candidate"))
                if not is_withheld and _row_is_candidate(obj):
                    text, raw_audit = _text_and_audit(obj)
                    assert text is not None and raw_audit is not None
                    key = (str(path), text)
                    if key not in seen:
                        seen.add(key)
                        checked = independent_audit(text)
                        letters = int(raw_audit.get("letters") or raw_audit.get("letter_count") or len(normalize_letters(text)))
                        exact = bool(raw_audit.get("exact") or raw_audit.get("two_pointer_exact") or raw_audit.get("direct_symmetric_position_comparison"))
                        # Recompute the exact bit independently; an artifact's claim is never trusted.
                        exact = checked["exact"]
                        mismatch_count = checked["mismatch_count"]
                        mismatch_rate = checked["mismatch_rate"]
                        parent = obj.get("parent_sha256")
                        if not parent and isinstance(obj.get("provenance"), dict):
                            parent = obj["provenance"].get("parent_sha256")
                        forward = checked["sha256_forward"]
                        node_id = hashlib.sha256(f"{path}\0{location}\0{text}".encode()).hexdigest()[:20]
                        surface = obj.get("surface_diagnostic")
                        intact = not isinstance(surface, dict) or not bool(surface.get("fragment_markers") or surface.get("has_fragment_marker"))
                        reader = bool(obj.get("reader_certified")) or bool(obj.get("human_reader_study", {}).get("triggered")) if isinstance(obj.get("human_reader_study"), dict) else False
                        forced_control = any(token in f"{path} {experiment_id}".casefold() for token in (
                            "fixed-tape", "exact-tape", "near-survivor", "semantic-lexical-path",
                            "coupled-bidirectional-grammar", "catalogue", "topic-half",
                            "grammar-boundary", "lexical-admission-centerout",
                            "reversible-grammar",
                        ))
                        failure_signature = _failure_signature(obj, raw_audit)
                        # Many constructive lanes keep their repair instruction
                        # at the artifact root while storing candidate rows
                        # below it. Propagate that instruction into each replay
                        # row only when the row has no concrete failure of its
                        # own; this changes routing evidence, never admission.
                        if failure_signature == "unclassified" and root_repair:
                            failure_signature = root_repair
                        nodes.append(Node(
                            node_id=node_id,
                            world=path.stem,
                            source_path=str(path.relative_to(ROOT)),
                            source_location=location,
                            parent_sha256=str(parent) if parent else None,
                            forward_sha256=forward,
                            action=_action(str(path), str(experiment_id)),
                            rendered=text,
                            letters=letters,
                            exact=exact,
                            mismatch_count=mismatch_count,
                            mismatch_rate=mismatch_rate,
                            length_ok=100 <= letters <= 160,
                            shortcut_free=_shortcut_free(obj) and not (exact and forced_control),
                            intact_surface=intact,
                            reader_certified=reader,
                            failure_signature=failure_signature,
                        ))
                for key, value in obj.items():
                    visit(value, f"{location}.{key}")
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    visit(value, f"{location}[{index}]")

        visit(payload, "$" )
    return nodes


POLICIES: list[dict[str, Any]] = [
    {
        "name": "fixed_mismatch_first",
        "description": "prioritize the smallest mismatch rate, then depth",
        "lane_bonus": {}, "failure_penalty": 0.0, "diversity_bonus": 0.0,
        "length_bonus": 0.0,
    },
    {
        "name": "length_preserving",
        "description": "protect the 100-letter floor while reducing error",
        "lane_bonus": {}, "failure_penalty": 0.0, "diversity_bonus": 0.0,
        "length_bonus": 0.02,
    },
    {
        "name": "frontier_reset",
        "description": "penalize repeated failure signatures and seek new lanes",
        "lane_bonus": {}, "failure_penalty": 0.35, "diversity_bonus": 0.4,
        "length_bonus": 0.01,
    },
    {
        "name": "whole_passage_focus",
        "description": "route replay budget toward global prose reconstruction",
        "lane_bonus": {"whole_passage": 0.7, "two_region": 0.9},
        "failure_penalty": 0.15, "diversity_bonus": 0.2, "length_bonus": 0.02,
    },
    {
        "name": "orthogonal_round_robin",
        "description": "spread budget over materially different construction actions",
        "lane_bonus": {}, "failure_penalty": 0.1, "diversity_bonus": 1.0,
        "length_bonus": 0.01,
    },
    {
        "name": "exact_gate_first",
        "description": "prioritize exact nodes but never hide rejected provenance",
        "lane_bonus": {}, "failure_penalty": 0.0, "diversity_bonus": 0.1,
        "length_bonus": 0.0,
    },
    {
        "name": "failure_repair_first",
        "description": "route replay toward actionable seam failures and their distinct repairs",
        "lane_bonus": {}, "failure_penalty": 0.8, "diversity_bonus": 0.6,
        "length_bonus": 0.01, "failure_bonus": 0.9,
    },
]


def _tier(node: Node) -> int:
    if node.exact and node.length_ok and node.shortcut_free and node.intact_surface:
        return 4
    if node.exact:
        return 3
    if node.length_ok and node.shortcut_free and node.intact_surface:
        return 2
    return 1 if node.rendered else 0


def _priority(node: Node, policy: dict[str, Any], seen_failures: set[str], seen_actions: set[str], depth: int) -> tuple[float, ...]:
    novelty = policy["diversity_bonus"] if node.action not in seen_actions else 0.0
    repeated = policy["failure_penalty"] if node.failure_signature in seen_failures else 0.0
    exact_bonus = 10.0 if policy["name"] == "exact_gate_first" and node.exact else 0.0
    # A failure is useful only when it names a concrete seam/repair.  This
    # turns replay from passive ranking into failure-conditioned exploration:
    # the first visit to an actionable failure gets budget, while repeated
    # copies are explicitly discounted.  It never changes candidate gates.
    actionable = 1.0 if node.failure_signature not in {"exact", "unclassified"} else 0.0
    failure_bonus = policy.get("failure_bonus", 0.0) * actionable
    score = (
        _tier(node) * 10.0
        + exact_bonus
        - node.mismatch_rate
        + policy["lane_bonus"].get(node.action, 0.0)
        + novelty
        - repeated
        + failure_bonus
        + node.letters * policy["length_bonus"]
    )
    return (score, -depth, node.letters, node.node_id)


def failure_repair_queue(nodes: list[Node], limit: int = 20) -> list[dict[str, Any]]:
    """Return deduplicated, actionable failures for the next live generator.

    This is deliberately a construction queue, not a readability score: each
    item preserves provenance and asks the next authoring lane to branch on the
    observed seam.  No tape is invented or admitted by this diagnostic.
    """
    grouped: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        if node.failure_signature not in {"exact", "unclassified"}:
            grouped[node.failure_signature].append(node)
    rows = []
    for signature, members in grouped.items():
        exemplar = min(members, key=lambda n: (n.mismatch_rate, -n.letters))
        rows.append({"failure_signature": signature, "occurrences": len(members),
                     "action": exemplar.action, "source_path": exemplar.source_path,
                     "source_location": exemplar.source_location,
                     "next_repair": f"branch the {exemplar.action} operator at {signature}"})
    return sorted(rows, key=lambda row: (-row["occurrences"], row["failure_signature"]))[:limit]


def replay_world(nodes: list[Node], policy: dict[str, Any], budget: int) -> dict[str, Any]:
    by_parent: dict[tuple[str, str], list[Node]] = defaultdict(list)
    by_forward: dict[tuple[str, str], Node] = {}
    for node in nodes:
        by_forward[(node.world, node.forward_sha256)] = node
    roots: list[Node] = []
    for node in nodes:
        parent = by_forward.get((node.world, node.parent_sha256 or ""))
        if parent is None:
            roots.append(node)
        else:
            by_parent[(node.world, parent.node_id)].append(node)
    frontier: list[tuple[Node, int]] = [(node, 0) for node in roots]
    visited: list[Node] = []
    seen_nodes: set[str] = set()
    seen_failures: set[str] = set()
    seen_actions: set[str] = set()
    while frontier and len(visited) < budget:
        frontier.sort(key=lambda item: _priority(item[0], policy, seen_failures, seen_actions, item[1]), reverse=True)
        node, depth = frontier.pop(0)
        if node.node_id in seen_nodes:
            continue
        seen_nodes.add(node.node_id)
        visited.append(node)
        seen_failures.add(node.failure_signature)
        seen_actions.add(node.action)
        frontier.extend((child, depth + 1) for child in by_parent[(node.world, node.node_id)])
    admissible_exact = [node for node in visited if _tier(node) == 4]
    # Exact-but-rejected catalogue/shortcut controls are tracked separately;
    # they must not collapse the near-miss routing score to zero error.
    intact = [node for node in visited if not node.exact and node.length_ok and node.shortcut_free and node.intact_surface]
    best = min(intact, key=lambda node: (node.mismatch_rate, -node.letters), default=None)
    return {
        "visited": len(visited),
        "admissible_exact": len(admissible_exact),
        # Never report an admission count without the concrete rendered rows
        # that earned it.  This prevents an inherited exact/control signal
        # from looking like a new reader-facing candidate in replay summaries.
        "admissible_exact_rows": [
            {
                "node_id": node.node_id,
                "world": node.world,
                "source_path": node.source_path,
                "source_location": node.source_location,
                "rendered": node.rendered,
                "letters": node.letters,
                "exact": node.exact,
                "mismatch_count": node.mismatch_count,
                "forward_sha256": node.forward_sha256,
                "shortcut_free": node.shortcut_free,
                "intact_surface": node.intact_surface,
                "reader_certified": node.reader_certified,
            }
            for node in admissible_exact
        ],
        "exact_rejected": sum(node.exact and _tier(node) < 4 for node in visited),
        "new_actions": len(seen_actions),
        "best": asdict(best) if best else None,
        "failure_signatures": len(seen_failures),
        "visited_node_ids": [node.node_id for node in visited],
    }


def aggregate(nodes: list[Node], policy: dict[str, Any], budget: int) -> dict[str, Any]:
    worlds: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        worlds[node.world].append(node)
    reports = {world: replay_world(rows, policy, budget) for world, rows in sorted(worlds.items())}
    bests = [report["best"] for report in reports.values() if report["best"]]
    best = min(bests, key=lambda row: (row["mismatch_rate"], -row["letters"]), default=None)
    return {
        "policy": policy["name"],
        "world_count": len(reports),
        "world_reports": reports,
        "admissible_exact": sum(report["admissible_exact"] for report in reports.values()),
        "admissible_exact_rows": [
            row for report in reports.values() for row in report["admissible_exact_rows"]
        ],
        "exact_rejected": sum(report["exact_rejected"] for report in reports.values()),
        "new_actions": sum(report["new_actions"] for report in reports.values()),
        "best_mismatch_rate": min((row["mismatch_rate"] for row in bests), default=1.0),
        "best_letters": max((row["letters"] for row in bests), default=0),
        "best": best,
    }


def tree_shape(nodes: list[Node]) -> dict[str, Any]:
    """Describe whether the recorded history is actually a replayable tree.

    A large flat list can look like a search history while offering no policy
    choices.  This diagnostic keeps that distinction explicit so a replay
    win cannot be claimed when the data contains only single-child chains.
    """
    by_forward = {(node.world, node.forward_sha256): node for node in nodes}
    children: dict[str, int] = defaultdict(int)
    parent_edges = 0
    for node in nodes:
        parent = by_forward.get((node.world, node.parent_sha256 or ""))
        if parent is not None:
            parent_edges += 1
            children[parent.node_id] += 1
    branching = [count for count in children.values() if count > 1]
    node_by_id = {node.node_id: node for node in nodes}
    return {
        "nodes": len(nodes),
        "parent_edges": parent_edges,
        "root_nodes": len(nodes) - parent_edges,
        "branching_parents": len(branching),
        "max_children": max(children.values(), default=0),
        "worlds": len({node.world for node in nodes}),
        "worlds_with_branching": len({node_by_id[parent_id].world for parent_id, count in children.items()
                                      if count > 1 and parent_id in node_by_id}),
    }


def select_policy(reports: list[dict[str, Any]]) -> dict[str, Any]:
    # Exact admissible closures dominate; then rejected exact controls; then
    # lower residual error, while action diversity breaks ties.
    return max(reports, key=lambda report: (
        report["admissible_exact"],
        report["exact_rejected"],
        -report["best_mismatch_rate"],
        report["best_letters"],
        report["new_actions"],
        report["policy"],
    ))


def novelty_preflight(signature: str, artifact: str) -> dict[str, Any]:
    data = json.loads(REGISTRY.read_text())
    entries = data.get("entries", [])
    collisions = [row for row in entries if row.get("id") == EXPERIMENT_ID or row.get("signature") == signature or row.get("artifact") == artifact]
    return {
        "status": "passed" if not collisions else "blocked",
        "registry_entries_checked": len(entries),
        "signature_collision": any(row.get("signature") == signature for row in collisions),
        "artifact_collision": any(row.get("artifact") == artifact for row in collisions),
        "id_collision": any(row.get("id") == EXPERIMENT_ID for row in collisions),
    }


def run(out: Path, budget: int = 8, online: bool = False) -> dict[str, Any]:
    nodes = load_worlds()
    signature = "dream-rsi|historical-discovery-tree-replay|offline-policy-improvement|online-redeploy"
    artifact = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    preflight = novelty_preflight(signature, artifact)
    if preflight["status"] != "passed":
        raise RuntimeError(f"novelty_preflight_failed:{preflight}")
    digest = hashlib.sha256("\n".join(sorted(node.node_id for node in nodes)).encode()).hexdigest()
    train = [node for node in nodes if int(node.node_id[-1], 16) % 2 == 0]
    train_ids = {node.node_id for node in train}
    heldout = [node for node in nodes if node.node_id not in train_ids]
    train_reports = [aggregate(train, policy, budget) for policy in POLICIES]
    winner = select_policy(train_reports)
    heldout_report = aggregate(heldout, next(policy for policy in POLICIES if policy["name"] == winner["policy"]), budget)
    repair_queue = failure_repair_queue(train)
    policy_signatures = {
        (report["admissible_exact"], report["exact_rejected"],
         report["best_mismatch_rate"], report["best_letters"])
        for report in train_reports
    }
    online_run = None
    if online:
        online_out = ROOT / "runs" / "dream-rsi-online-branching-two-region-20260917.json"
        base_anchors = (
            "the theater archivist; a torn playbill; the locked drawer; carrying "
            "it to the reading table; marking the missing cast names before the house lights rose"
        )
        if repair_queue:
            base_anchors += "; Dream-RSI live repair priority: " + repair_queue[0]["next_repair"]
        command = [sys.executable, str(ROOT / "experiments" / "branching_two_region_authoring_20260917.py"),
                   "--experiment-id", "dream-rsi-online-branching-two-region-20260917",
                   "--out", str(online_out), "--branch-factor", "3", "--depth", "2",
                   "--initial",
                   "The theater archivist found a torn playbill in a locked drawer, carried it to the reading table, and marked the missing cast names before the house lights rose.",
                   "--anchors",
                   base_anchors]
        try:
            completed = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, check=False,
                timeout=240,
            )
            online_run = {
                "winner_policy": winner["policy"],
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": False,
                "artifact": str(online_out.relative_to(ROOT)) if online_out.exists() else None,
            }
        except subprocess.TimeoutExpired as exc:
            online_run = {
                "winner_policy": winner["policy"],
                "command": command,
                "returncode": None,
                "stdout": str(exc.stdout or ""),
                "stderr": str(exc.stderr or ""),
                "timed_out": True,
                "artifact": str(online_out.relative_to(ROOT)) if online_out.exists() else None,
            }
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": signature,
        "status": "completed_online_round" if online_run and online_run["returncode"] == 0 else "completed_replay_policy_selection",
        "novelty_preflight": preflight,
        "method": "historical discovery trees as replay simulators; deterministic policy frontier; held-out selection",
        "source_provenance": {
            "run_artifacts_scanned": sorted({node.source_path for node in nodes}),
            "node_count": len(nodes),
            "history_digest": digest,
            "unseen_candidates_in_replay": False,
        },
        "history_shape": tree_shape(nodes),
        "policy_separation": {
            "distinct_train_metric_signatures": len(policy_signatures),
            "policy_count": len(POLICIES),
            "interpretation": (
                "policies are indistinguishable on this history; collect sibling alternatives"
                if len(policy_signatures) == 1 else
                "policies produce distinct replay outcomes"
            ),
        },
        "split": {"train_nodes": len(train), "heldout_nodes": len(heldout), "split_rule": "last node-id hex nibble parity"},
        "replay_budget_per_world": budget,
        "policy_frontier": train_reports,
        "failure_repair_queue": repair_queue,
        "winner": winner,
        "heldout_winner": heldout_report,
        "online_deployment": online_run,
        "acceptance_order": [
            "exact + novel + intact prose",
            "exact but rejected provenance",
            "near miss with lower residual",
            "ordinary fluent non-palindrome",
        ],
        "reader_gate": "closed; replay and diagnostics cannot certify readability; any exact novel deployment must enter blinded intact/shuffled reader testing",
        "next_repair": "Use the selected policy to choose among sibling prose proposals; if a fresh tree still collapses to one child, change the live construction operator again rather than retuning the score.",
        "independent_audits": ["independent ASCII two-pointer scan on every replay node", "forward/reverse SHA-256"],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    result = run(args.out, budget=args.budget, online=args.online)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "nodes": result["source_provenance"]["node_count"],
        "winner": result["winner"]["policy"],
        "heldout_best_rate": result["heldout_winner"]["best_mismatch_rate"],
        "online": bool(result["online_deployment"]),
    }))
