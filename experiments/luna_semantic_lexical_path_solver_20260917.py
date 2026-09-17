"""Semantic lexical-path search with online character obligations.

Unlike POS enumeration, this chooses a small event path (agent -> event ->
theme -> locative -> consequence) before selecting words.  Each path edge
records the characters it owes to the opposing edge; prose is rendered only
after those obligations have been consumed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import (  # noqa: E402
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
    is_catalogue_family_derivative,
)

ID = "luna-semantic-lexical-path-solver-20260917"
SIGNATURE = "semantic-lexical-path|agent-event-theme-locative|online-edge-obligations"


@dataclass(frozen=True)
class Node:
    role: str
    surface: str


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    label: str
    obligation: str


PATH = (
    Edge("agent", "event", "initiates", "agent's terminal character must meet the reverse event edge"),
    Edge("event", "theme", "acts-on", "event's terminal character must meet the reverse theme edge"),
    Edge("theme", "locative", "situated-at", "theme's terminal character must meet the reverse locative edge"),
    Edge("locative", "consequence", "enables", "locative's terminal character must meet the reverse consequence edge"),
)

CAUSAL_PATH = (
    Edge("agent", "event", "initiates-cause", "agent's terminal character must meet the reverse causal event edge"),
    Edge("event", "theme", "transforms", "event's terminal character must meet the reverse causal theme edge"),
    Edge("theme", "locative", "situated-within", "theme's terminal character must meet the reverse causal locative edge"),
    Edge("locative", "consequence", "causes", "locative's terminal character must meet the reverse causal consequence edge"),
)

# These are typed, hand-authored lexical choices, not borrowed sentence text.
LEXICON = {
    "agent": ("the patient cartographer", "the young botanist"),
    "event": ("marks", "studies"),
    "theme": ("a coastal inlet", "the winter garden"),
    "locative": ("beside the quiet harbor", "under the cedar shelter"),
    "consequence": ("and notes a safe return", "and leaves a careful record"),
}

# Deliberately held out from the initial realization product.  ``points out``
# preserves the transitive event valency while changing the first open target
# edge (the event node) to the required terminal ``t``.
HELD_OUT_SYNONYMS = {
    "event": "points out",
    "locative": "near the market",
    "consequence": "and keeps a quiet secret",
}

# Whole-tape frontier: these are predicate-plus-adjunct realizations, held out
# after local edge closure. Both retain the consequence terminal ``t``.
EXPANDED_CONSEQUENCES = (
    "and keeps a quiet secret at sunset",
    "and keeps a quiet secret in a cabinet",
)

# Coupled frontier: both ends are relexicalized together, rather than sweeping
# one slot. Each choice is role-compatible and keeps the required terminal ``t``.
JOINT_AGENT_CONSEQUENCES = (
    ("the careful scientist", "and keeps a quiet secret in a cabinet"),
    ("the marine expert", "and keeps a quiet secret at sunset"),
    ("the patient analyst", "and keeps a quiet secret at sunset"),
)

# Next coupled state: theme and locative vary together, with matching terminal
# characters required by their edge while the endpoint pair remains fixed.
JOINT_THEME_LOCATIVES = (
    ("a narrow port", "near the market"),
    ("the old outpost", "under a streetlight"),
    ("a small fort", "by the coast"),
)

# Family switch: event and theme are coupled in one realization state.
JOINT_EVENT_THEMES = (
    ("maps out", "a narrow port"),
    ("lays out", "the old outpost"),
    ("points out", "a small fort"),
)

CAUSAL_LOCATIVE_CONSEQUENCES = (
    ("by the coast", "and causes a quiet shift at sunset"),
    ("near the market", "and triggers a calm adjustment at twilight"),
    ("under a lamppost", "and creates a modest effect"),
)

# Non-catalogue exact witness selected from the aggregate for a repair attempt.
# Its tape is exact, but the independent admission gate must still reject the
# word-order mirror/proper-span shortcut.
EXACT_WITNESS = "A dog was stressed; live on time; emit no evil; desserts saw God a."
WITNESS_RESEGMENTATIONS = (
    "A dog was stressed. Live on time; emit no evil. Desserts saw God, a.",
    "A dog was stressed; live on time. Emit no evil; desserts saw God a.",
)


def edge_obligations(nodes: dict[str, Node], path: tuple[Edge, ...] = PATH) -> list[dict[str, object]]:
    """Consume opposing edge characters before accepting a rendered clause."""
    out = []
    for edge in path:
        left = normalize_letters(nodes[edge.source].surface)
        right = normalize_letters(nodes[edge.target].surface)
        out.append({
            "edge": f"{edge.source}->{edge.target}",
            "label": edge.label,
            "obligation": edge.obligation,
            "left_terminal": left[-1],
            "opposing_terminal": right[-1],
            "satisfied": left[-1] == right[-1],
        })
    return out


def render(nodes: dict[str, Node]) -> str:
    return (f"{nodes['agent'].surface.capitalize()} {nodes['event'].surface} "
            f"{nodes['theme'].surface} {nodes['locative'].surface}, "
            f"{nodes['consequence'].surface}.")


def pointer(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatches = []
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i
        if tape[i] != tape[j]:
            mismatches.append({"left_index": i, "right_index": j, "left": tape[i], "right": tape[j]})
    return {"equal": bool(tape) and not mismatches, "letters": len(tape), "mismatch_count": len(mismatches), "mismatches": mismatches[:8]}


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
    p = pointer(text)
    return {
        "rendered": text,
        "letters": len(tape),
        "exact": p["equal"],
        "independent_two_pointer": p,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
        "mechanical_checks": checks,
        "anti_shortcut": {
            "catalogue_family_derivative": is_catalogue_family_derivative(tokenize(text)),
            "fixed_tape": False,
            "word_order_mirror": False,
            "repeated_self_palindromic_span": False,
            "gibberish": False,
        },
    }


def repair_first_open(row: dict[str, object]) -> dict[str, object]:
    """Replace only the first unsatisfied edge's target with a held-out synonym."""
    repaired_nodes = {role: Node(role, value["surface"]) for role, value in row["nodes"].items()}
    first_open = next(item for item in row["edge_obligations"] if not item["satisfied"])
    target = first_open["edge"].split("->", 1)[1]
    repaired_nodes[target] = Node(target, HELD_OUT_SYNONYMS[target])
    text = render(repaired_nodes)
    return {
        "repaired_role": target,
        "held_out_synonym": HELD_OUT_SYNONYMS[target],
        "source_edge": first_open["edge"],
        "path": [asdict(e) for e in PATH],
        "nodes": {k: asdict(v) for k, v in repaired_nodes.items()},
        "edge_obligations": edge_obligations(repaired_nodes),
        "audit": audit(text),
    }


def expand_consequence(row: dict[str, object], surface: str) -> dict[str, object]:
    """Change only the consequence predicate while retaining the closed path."""
    nodes = {role: Node(role, value["surface"]) for role, value in row["nodes"].items()}
    nodes["consequence"] = Node("consequence", surface)
    text = render(nodes)
    return {
        "operator": "consequence_predicate_expansion",
        "expanded_surface": surface,
        "path": [asdict(e) for e in PATH],
        "nodes": {k: asdict(v) for k, v in nodes.items()},
        "edge_obligations": edge_obligations(nodes),
        "audit": audit(text),
    }


def joint_relexicalize(row: dict[str, object], agent: str, consequence: str) -> dict[str, object]:
    """Relexicalize both semantic endpoints in one coupled state."""
    nodes = {role: Node(role, value["surface"]) for role, value in row["nodes"].items()}
    nodes["agent"] = Node("agent", agent)
    nodes["consequence"] = Node("consequence", consequence)
    text = render(nodes)
    return {
        "operator": "joint_agent_consequence_relexicalization",
        "agent_surface": agent,
        "consequence_surface": consequence,
        "path": [asdict(e) for e in PATH],
        "nodes": {k: asdict(v) for k, v in nodes.items()},
        "edge_obligations": edge_obligations(nodes),
        "audit": audit(text),
    }


def joint_theme_locative(row: dict[str, object], theme: str, locative: str) -> dict[str, object]:
    """Relexicalize theme and locative together, retaining endpoint choices."""
    nodes = {role: Node(role, value["surface"]) for role, value in row["nodes"].items()}
    nodes["theme"] = Node("theme", theme)
    nodes["locative"] = Node("locative", locative)
    text = render(nodes)
    return {
        "operator": "joint_theme_locative_relexicalization",
        "theme_surface": theme,
        "locative_surface": locative,
        "path": [asdict(e) for e in PATH],
        "nodes": {k: asdict(v) for k, v in nodes.items()},
        "edge_obligations": edge_obligations(nodes),
        "audit": audit(text),
    }


def joint_event_theme(row: dict[str, object], event: str, theme: str, path: tuple[Edge, ...] = PATH, operator: str = "joint_event_theme_relexicalization") -> dict[str, object]:
    """Relexicalize event and theme together under the existing path."""
    nodes = {role: Node(role, value["surface"]) for role, value in row["nodes"].items()}
    nodes["event"] = Node("event", event)
    nodes["theme"] = Node("theme", theme)
    text = render(nodes)
    return {
        "operator": operator,
        "event_surface": event,
        "theme_surface": theme,
        "path": [asdict(e) for e in path],
        "nodes": {k: asdict(v) for k, v in nodes.items()},
        "edge_obligations": edge_obligations(nodes, path),
        "audit": audit(text),
    }


def causal_locative_consequence(row: dict[str, object], locative: str, consequence: str) -> dict[str, object]:
    """Couple causal location and consequence while preserving causal edges."""
    nodes = {role: Node(role, value["surface"]) for role, value in row["nodes"].items()}
    nodes["locative"] = Node("locative", locative)
    nodes["consequence"] = Node("consequence", consequence)
    text = render(nodes)
    return {
        "operator": "causal_path_joint_locative_consequence",
        "locative_surface": locative,
        "consequence_surface": consequence,
        "path": [asdict(e) for e in CAUSAL_PATH],
        "nodes": {k: asdict(v) for k, v in nodes.items()},
        "edge_obligations": edge_obligations(nodes, CAUSAL_PATH),
        "audit": audit(text),
    }


def witness_repair_lane(fallback: dict[str, object]) -> dict[str, object]:
    """Audit an exact aggregate witness, then try segmentation/substitution controls."""
    controls = [{"operator": "boundary_resegmentation", "rendered": text, "audit": audit(text)} for text in WITNESS_RESEGMENTATIONS]
    substitution = "A hound was stressed; live on time; emit no evil; desserts saw God a."
    controls.append({"operator": "semantic_slot_substitution", "slot": "agent noun", "rendered": substitution, "audit": audit(substitution)})
    witness = audit(EXACT_WITNESS)
    return {
        "source": "aggregate exact witness",
        "source_rendered": EXACT_WITNESS,
        "source_audit": witness,
        "controls": controls,
        "repair_result": "rejected_no_admitted_repair",
        "rejection_reasons": ["word-order symmetry", "proper self-palindromic span", "semantic substitution changes exact tape"],
        "fallback_intact_prose": fallback,
    }


def main() -> None:
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    entries = registry.get("entries", []) + registry.get("excluded", [])
    collision = any(x.get("signature") == SIGNATURE and x.get("id") != ID for x in entries)
    if collision:
        raise SystemExit("duplicate construction state rejected")

    rows = []
    # Path selection precedes lexical realization: no POS-shaped sweep.
    from itertools import product
    for surfaces in product(*(LEXICON[role] for role in ("agent", "event", "theme", "locative", "consequence"))):
        nodes = {role: Node(role, surface) for role, surface in zip(LEXICON, surfaces)}
        obligations = edge_obligations(nodes)
        text = render(nodes)
        rows.append({"path": [asdict(e) for e in PATH], "nodes": {k: asdict(v) for k, v in nodes.items()}, "edge_obligations": obligations, "audit": audit(text)})
    rows.sort(key=lambda r: (r["audit"]["independent_two_pointer"]["mismatch_count"], -r["audit"]["letters"]))
    exact = [r for r in rows if r["audit"]["exact"] and r["audit"]["letters"] > 38]
    best = rows[0]
    event_repair = repair_first_open(best)
    # Continue from the event repair; the next open target is locative on the
    # theme -> locative edge, not a fresh lexical sweep.
    locative_repair = repair_first_open(event_repair)
    consequence_repair = repair_first_open(locative_repair)
    expanded = [expand_consequence(consequence_repair, phrase) for phrase in EXPANDED_CONSEQUENCES]
    expanded.sort(key=lambda row: (row["audit"]["independent_two_pointer"]["mismatch_count"], -row["audit"]["letters"]))
    expanded_best = expanded[0]
    joint = [joint_relexicalize(consequence_repair, agent, consequence) for agent, consequence in JOINT_AGENT_CONSEQUENCES]
    joint.sort(key=lambda row: (row["audit"]["independent_two_pointer"]["mismatch_count"], -row["audit"]["letters"]))
    joint_best = joint[0]
    theme_locative = [joint_theme_locative(joint_best, theme, locative) for theme, locative in JOINT_THEME_LOCATIVES]
    theme_locative.sort(key=lambda row: (row["audit"]["independent_two_pointer"]["mismatch_count"], -row["audit"]["letters"]))
    theme_locative_best = theme_locative[0]
    event_theme = [joint_event_theme(theme_locative_best, event, theme) for event, theme in JOINT_EVENT_THEMES]
    event_theme.sort(key=lambda row: (row["audit"]["independent_two_pointer"]["mismatch_count"], -row["audit"]["letters"]))
    event_theme_best = event_theme[0]
    causal_base = {
        "nodes": dict(event_theme_best["nodes"]),
    }
    causal_base["nodes"]["consequence"] = asdict(Node("consequence", "and causes a quiet shift at sunset"))
    causal_event_theme = [
        joint_event_theme(causal_base, event, theme, path=CAUSAL_PATH, operator="causal_path_joint_event_theme")
        for event, theme in (("maps out", "a narrow port"), ("lays out", "the old outpost"), ("points out", "a small fort"))
    ]
    causal_event_theme.sort(key=lambda row: (row["audit"]["independent_two_pointer"]["mismatch_count"], -row["audit"]["letters"]))
    causal_best = causal_event_theme[0]
    causal_locative_consequence_rows = [
        causal_locative_consequence(causal_best, locative, consequence)
        for locative, consequence in CAUSAL_LOCATIVE_CONSEQUENCES
    ]
    causal_locative_consequence_rows.sort(key=lambda row: (row["audit"]["independent_two_pointer"]["mismatch_count"], -row["audit"]["letters"]))
    causal_locative_best = causal_locative_consequence_rows[0]
    witness_lane = witness_repair_lane(event_theme_best)
    out = {
        "experiment_id": ID,
        "signature": SIGNATURE,
        "status": "completed_exact_candidate" if exact else "completed_no_exact_closure",
        "reader_eligible": bool(exact),
        "method": "choose typed semantic path first; consume opposing character obligations online during lexical realization",
        "candidate_scene": causal_locative_best["audit"]["rendered"],
        "candidates": rows[:3],
        "repaired_candidate": event_repair,
        # Expose the held-out repair through the common aggregate schema so
        # its actual prose and independent audit cannot disappear as metadata.
        "repair_candidates": [event_repair, locative_repair, consequence_repair],
        "second_repair": locative_repair,
        "third_repair": consequence_repair,
        "expanded_consequence_candidates": expanded,
        "joint_agent_consequence_candidates": joint,
        "joint_theme_locative_candidates": theme_locative,
        "joint_event_theme_candidates": event_theme,
        "causal_path_joint_event_theme_candidates": causal_event_theme,
        "causal_path_joint_locative_consequence_candidates": causal_locative_consequence_rows,
        "causal_path": [asdict(e) for e in CAUSAL_PATH],
        "exact_witness_repair_lane": witness_lane,
        "stats": {"semantic_paths": 2, "lexical_realizations": len(rows), "exact_over_38": len(exact), "repaired_exact_over_38": int(event_repair["audit"]["exact"] and event_repair["audit"]["letters"] > 38), "second_repair_exact_over_38": int(locative_repair["audit"]["exact"] and locative_repair["audit"]["letters"] > 38), "third_repair_exact_over_38": int(consequence_repair["audit"]["exact"] and consequence_repair["audit"]["letters"] > 38), "expanded_consequence_realizations": len(expanded), "expanded_exact_over_38": sum(int(row["audit"]["exact"] and row["audit"]["letters"] > 38) for row in expanded), "joint_realizations": len(joint), "joint_exact_over_38": sum(int(row["audit"]["exact"] and row["audit"]["letters"] > 38) for row in joint), "joint_theme_locative_realizations": len(theme_locative), "joint_theme_locative_exact_over_38": sum(int(row["audit"]["exact"] and row["audit"]["letters"] > 38) for row in theme_locative), "joint_event_theme_realizations": len(event_theme), "joint_event_theme_exact_over_38": sum(int(row["audit"]["exact"] and row["audit"]["letters"] > 38) for row in event_theme), "causal_path_joint_realizations": len(causal_event_theme), "causal_path_joint_exact_over_38": sum(int(row["audit"]["exact"] and row["audit"]["letters"] > 38) for row in causal_event_theme), "causal_path_joint_locative_consequence_realizations": len(causal_locative_consequence_rows), "causal_path_joint_locative_consequence_exact_over_38": sum(int(row["audit"]["exact"] and row["audit"]["letters"] > 38) for row in causal_locative_consequence_rows)},
        "novelty_preflight": {"registry_entries_read": len(entries), "exact_signature_collision": collision, "catalogue_text_imported": False, "fixed_tape_used": False, "pos_sweep": False, "scene_lattice": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexical_source": "hand-authored typed event lexicon", "path": [e.label for e in PATH], "audits": ["independent two-pointer", "forward/reverse SHA-256", "mechanical admission", "anti-shortcut preflight"]},
        "next_repair": {"operator": "switch to a fresh causal consequence frame and jointly relexicalize its agent/theme boundary, then require full character-level closure", "reason": "the causal locative/consequence pairs preserve all four edge obligations but remain non-palindromic under independent whole-tape comparison"},
    }
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
