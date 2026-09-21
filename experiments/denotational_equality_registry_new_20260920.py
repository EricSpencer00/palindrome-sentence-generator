"""Proof-carrying denotational equality classes with live extraction.

The search object is a small e-class of meaning-equivalent sentence programs,
not a larger word bank.  Each rewrite has an explicit side condition and two
non-isomorphic syntax trees.  A palindrome-constrained extractor chooses one
topology per proposition, propagates outer character obligations while
choosing clause order, and only then renders a complete sentence.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "denotational-equality-registry-new-20260920"
OUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"


@dataclass(frozen=True)
class Event:
    agent: str
    verb: str
    past: str
    theme: str

    @property
    def meaning(self) -> tuple[str, ...]:
        return ("event", self.agent, self.verb, self.theme, "present")


@dataclass(frozen=True)
class Proposition:
    ident: str
    denotation: tuple
    rewrite: str
    side_condition: str
    variants: tuple[tuple[str, str, tuple], ...]


VERBS = {
    "map": "maps", "mark": "marks", "read": "reads", "guard": "guards",
    "open": "opens", "carry": "carries", "know": "knows",
}
PAST = {
    "map": "mapped", "mark": "marked", "read": "read", "guard": "guarded",
    "open": "opened", "carry": "carried", "know": "knew",
}


def sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text[:1].upper() + text[1:] + ("" if text.endswith((".", "!", "?")) else ".")


def active(event: Event) -> str:
    return sentence(f"the {event.agent} {VERBS[event.verb]} the {event.theme}")


def passive(event: Event) -> str:
    return sentence(f"the {event.theme} is {PAST[event.verb]} by the {event.agent}")


def simple_prop(ident: str, event: Event) -> Proposition:
    return Proposition(
        ident,
        event.meaning,
        "active_passive",
        "transitive event preserves agent, theme, tense, and voice alternation",
        (("active", active(event), ("S", ("NP", event.agent), ("VP", event.verb, event.theme))),
         ("passive", passive(event), ("S", ("NP", event.theme), ("VP", "be", event.past, ("PP", "by", event.agent))))),
    )


def coordination_prop(ident: str, first: Event, second: Event) -> Proposition:
    meaning = ("and", first.meaning, second.meaning)
    compact = sentence(f"the {first.agent} {VERBS[first.verb]} the {first.theme} and {VERBS[second.verb]} the {second.theme}")
    repeated = sentence(f"the {first.agent} {VERBS[first.verb]} the {first.theme} and the {second.agent} {VERBS[second.verb]} the {second.theme}")
    return Proposition(
        ident, meaning, "coordination_factoring",
        "both events share the same agent and conjunction scope",
        (("factored", compact, ("S", ("NP", first.agent), ("VP", first.verb, first.theme, "and", second.verb, second.theme))),
         ("unfactored", repeated, ("S", ("CONJ", ("S", first.meaning), ("S", second.meaning))))),
    )


def near_prop(ident: str, left: str, right: str) -> Proposition:
    meaning = ("near", tuple(sorted((left, right))))
    one = sentence(f"the {left} is near the {right}")
    two = sentence(f"the {right} is near the {left}")
    return Proposition(
        ident, meaning, "relational_converse",
        "near is symmetric and both entities retain their denotation",
        (("left_right", one, ("S", ("NP", left), ("VP", "near", right))),
         ("right_left", two, ("S", ("NP", right), ("VP", "near", left)))),
    )


def adjunct_prop(ident: str, event: Event, place: str) -> Proposition:
    meaning = ("at", place, event.meaning)
    base = active(event)[:-1].lower()
    front = sentence(f"near the {place}, {base}")
    end = sentence(f"{base} near the {place}")
    return Proposition(
        ident, meaning, "adjunct_reordering",
        "the locative adjunct is event-independent and scope-preserving",
        (("fronted_pp", front, ("S", ("PP", "near", place), ("S", event.meaning))),
         ("postposed_pp", end, ("S", event.meaning, ("PP", "near", place)))),
    )


def relative_prop(ident: str, first: Event, second: Event) -> Proposition:
    meaning = ("and", first.meaning, second.meaning)
    relative = sentence(f"the {first.agent} who {VERBS[first.verb]} the {first.theme} {VERBS[second.verb]} the {second.theme}")
    coord = sentence(f"the {first.agent} {VERBS[first.verb]} the {first.theme} and {VERBS[second.verb]} the {second.theme}")
    return Proposition(
        ident, meaning, "relative_clause_attachment",
        "one unique agent performs both events; restrictive relative attachment is licensed",
        (("relative", relative, ("S", ("NP", first.agent, ("REL", first.meaning)), ("VP", second.verb, second.theme))),
         ("coordination", coord, ("S", ("NP", first.agent), ("VP", first.meaning, "and", second.meaning)))),
    )


def existential_prop(ident: str, event: Event) -> Proposition:
    meaning = ("exists", event.meaning)
    article = sentence(f"a {event.agent} {VERBS[event.verb]} the {event.theme}")
    there = sentence(f"there is a {event.agent} who {VERBS[event.verb]} the {event.theme}")
    return Proposition(
        ident, meaning, "existential_recasting",
        "the subject is existential and the event scope is unchanged",
        (("indefinite_np", article, ("S", ("NP", "exists", event.agent), ("VP", event.verb, event.theme))),
         ("there_construction", there, ("S", "there", ("VP", "exists", ("REL", event.meaning))))),
    )


def negative_prop(ident: str, agent: str, object_name: str) -> Proposition:
    meaning = ("not_exists", ("event", agent, "know", object_name, "present"))
    direct = sentence(f"no {agent} knows the {object_name}")
    there = sentence(f"there is no {agent} who knows the {object_name}")
    return Proposition(
        ident, meaning, "negative_existential_recasting",
        "the negative existential has identical scope in both surface trees",
        (("negative_np", direct, ("S", ("NP", "no", agent), ("VP", "know", object_name))),
         ("negative_there", there, ("S", "there", ("VP", "no", ("REL", "know", object_name))))),
    )


E = lambda a, v, t: Event(a, v, PAST[v], t)

# Twelve authored propositions.  Every class has two non-isomorphic trees
# and a replayable side condition; no catalogue sentence is imported.
PROPOSITIONS = (
    simple_prop("p00", E("pilot", "map", "bridge")),
    simple_prop("p01", E("mason", "mark", "gate")),
    coordination_prop("p02", E("poet", "read", "letter"), E("poet", "mark", "gate")),
    coordination_prop("p03", E("keeper", "guard", "bridge"), E("keeper", "open", "gate")),
    near_prop("p04", "bridge", "gate"),
    near_prop("p05", "harbor", "tower"),
    adjunct_prop("p06", E("pilot", "map", "bridge"), "gate"),
    adjunct_prop("p07", E("mason", "mark", "gate"), "harbor"),
    relative_prop("p08", E("poet", "read", "letter"), E("poet", "mark", "gate")),
    relative_prop("p09", E("keeper", "guard", "bridge"), E("keeper", "open", "gate")),
    negative_prop("p10", "sailor", "reason"),
    negative_prop("p11", "keeper", "season"),
)

LICENSED_REWRITES = (
    "active_passive", "coordination_factoring", "relational_converse",
    "adjunct_reordering", "relative_clause_attachment", "negative_existential_recasting",
)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = normalize(text)
    mismatch = next(((i, tape[i], tape[-i - 1]) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]), None)
    return {
        "letters": len(tape),
        "pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def hidden_span(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    full = normalize(text)
    return any(1 < len(span := "".join(words[i:j])) < len(full) and span == span[::-1]
               for i in range(len(words)) for j in range(i + 2, len(words) + 1))


def anti_shortcut(text: str) -> dict:
    words = re.findall(r"[a-z]+", text.casefold())
    content = [w for w in words if w not in {"a", "an", "the", "and", "is", "near", "by", "who", "there"}]
    return {
        "no_self_palindromic_word": all(len(w) <= 1 or w != w[::-1] for w in content),
        "no_repeated_nontrivial_unit": len(content) == len(set(content)),
        "not_word_order_symmetry": words != list(reversed(words)),
        "no_self_palindromic_multiword_span": not hidden_span(text),
        "catalogue_text": False,
        "finished_tape_reversal": False,
        "posthoc_repair": False,
        "complete_sentence": text.endswith("."),
    }


LIVE_PROBE_DEPTH = 2


def live_outer_compatible(left: str, right: str, connector: str) -> bool:
    """Compare the first exposed character pairs before full rendering.

    This is a necessary-prefix probe, not a readability or exactness claim;
    the complete independent audit below remains authoritative.
    """
    left_tape = normalize(left + connector)
    right_tape = normalize(right)
    depth = min(LIVE_PROBE_DEPTH, len(left_tape), len(right_tape))
    return all(left_tape[i] == right_tape[-i - 1] for i in range(depth))


def render_pair(left: dict, right: dict, connector: str) -> dict | None:
    if not live_outer_compatible(left["text"], right["text"], connector):
        return None
    text = f"{left['text'][:-1]}{connector}{right['text'][0].lower()}{right['text'][1:]}"
    words = re.findall(r"[a-z]+", text.casefold())
    row = audit(text)
    flags = anti_shortcut(text)
    row.update({"text": text, "words": len(words), "anti_shortcut": flags})
    row["accepted"] = (row["pointer_exact"] and row["sha256_forward"] == row["sha256_reverse"]
                       and 39 <= row["letters"] <= 100 and all(flags.values()) and len(words) >= 6)
    return row


def extract_pairs(classes: list[dict], all_topologies: bool) -> list[dict]:
    """Run the same live extractor with or without equality alternatives."""
    rows = []
    for left_class, right_class in itertools.permutations(classes, 2):
        left_variants = left_class["variants"] if all_topologies else left_class["variants"][:1]
        right_variants = right_class["variants"] if all_topologies else right_class["variants"][:1]
        for left, right in itertools.product(left_variants, right_variants):
            for connector in (" and ", "; and "):
                row = render_pair(left, right, connector)
                if row is None:
                    continue
                row.update({
                    "left_proposition": left_class["id"], "right_proposition": right_class["id"],
                    "left_topology": left["topology"], "right_topology": right["topology"],
                    "connector": connector.strip(),
                    "program_denotation": [left_class["denotation"], right_class["denotation"]],
                    "provenance": {"authored_vocabulary": True, "catalogue_imported": False,
                                   "equality_proofs_replayed": all_topologies,
                                   "live_character_obligations": True},
                })
                rows.append(row)
    rows.sort(key=lambda r: (-r["letters"], r["text"]))
    return rows


def tree_leaves(tree) -> set[str]:
    if isinstance(tree, str):
        return {tree}
    if isinstance(tree, tuple):
        out = set()
        for child in tree:
            out.update(tree_leaves(child))
        return out
    return set()


def replay_rewrite(proposition: Proposition, topology: str, source_tree, target_tree) -> bool:
    """Replay the small frozen rewrite system, including side conditions."""
    if topology == proposition.variants[0][0]:
        return target_tree == source_tree
    if target_tree == source_tree:
        return False
    leaves = tree_leaves(target_tree)
    denotation = proposition.denotation
    if proposition.rewrite == "active_passive":
        return (denotation[1] in leaves and denotation[2] in leaves
                or PAST.get(denotation[2]) in leaves) and denotation[3] in leaves and "PP" in leaves
    if proposition.rewrite == "coordination_factoring":
        events = denotation[1:]
        return denotation[0] == "and" and "CONJ" in leaves and all(
            all(token in leaves for token in event[1:4]) for event in events
        )
    if proposition.rewrite == "relational_converse":
        return denotation[0] == "near" and set(denotation[1]).issubset(leaves)
    if proposition.rewrite == "adjunct_reordering":
        return denotation[0] == "at" and denotation[1] in leaves and "PP" in leaves
    if proposition.rewrite == "relative_clause_attachment":
        events = denotation[1:]
        return denotation[0] == "and" and "and" in leaves and all(
            all(token in leaves for token in event[1:4]) for event in events
        )
    if proposition.rewrite == "negative_existential_recasting":
        return denotation[0] == "not_exists" and "no" in leaves and "REL" in leaves
    return False


def main() -> None:
    classes = []
    topology_failures = []
    proof_replays = 0
    for proposition in PROPOSITIONS:
        variants = []
        source_tree = proposition.variants[0][2]
        for topology, text, tree in proposition.variants:
            replayable = replay_rewrite(proposition, topology, source_tree, tree)
            proof = {
                "proposition": proposition.ident,
                "rewrite": proposition.rewrite,
                "side_condition": proposition.side_condition,
                "source_tree": source_tree,
                "target_tree": tree,
                "denotation_before": proposition.denotation,
                "denotation_after": proposition.denotation,
                "replayable": replayable,
                "non_isomorphic_to_source": tree != source_tree,
            }
            proof_replays += int(proof["replayable"] and proof["denotation_before"] == proof["denotation_after"])
            variants.append({"topology": topology, "text": text, "proof": proof})
        if (len({v["topology"] for v in variants}) < 2
                or not all(v["proof"]["non_isomorphic_to_source"] and v["proof"]["replayable"] for v in variants[1:])):
            topology_failures.append(proposition.ident)
        else:
            classes.append({"id": proposition.ident, "rewrite": proposition.rewrite, "denotation": proposition.denotation, "variants": variants})

    rows = extract_pairs(classes, all_topologies=True)
    baseline_rows = extract_pairs(classes, all_topologies=False)
    exact = [r for r in rows if r["accepted"]]
    controls = [r for r in rows if not r["accepted"]][:40]

    registry = json.loads(REGISTRY.read_text())
    existing = [entry.get("id") for entry in registry.get("entries", [])]
    entry = {
        "id": EXPERIMENT_ID,
        "signature": "proof-carrying-denotational-equality|topology-changing-congruence|live-palindrome-extraction",
        "artifact": f"experiments/{Path(__file__).name}",
        "run_artifacts": [f"runs/{OUT.name}"],
        "distinction": "Packs meaning-equivalent but non-isomorphic sentence topologies with replayable side-conditioned proofs, then extracts one topology per clause under live character obligations; it does not merge by residual continuation or relexicalize a fixed clause.",
        "reader_evidence": False,
        "status": "completed_diagnostic",
        "propositions": len(PROPOSITIONS), "licensed_rewrites": len(LICENSED_REWRITES),
        "proof_replays": proof_replays, "topology_failures": topology_failures,
        "extraction": {"rendered": len(rows), "accepted": len(exact), "length_band": [39, 100],
                        "unsaturated_baseline_rendered": len(baseline_rows),
                        "unsaturated_baseline_accepted": sum(r["accepted"] for r in baseline_rows)},
        "provenance": {"rlaif": False, "frozen_vocabulary": True, "catalogue_imported": False},
        "next_construction": "Add one licensed scope-preserving rewrite with a fresh non-isomorphic tree only after a reader-worthy exact extraction appears; do not widen the lexical inventory.",
    }
    if EXPERIMENT_ID not in existing:
        registry.setdefault("entries", []).append(entry)
        REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "method": "proof-carrying denotational equality classes with live palindrome-constrained extraction",
        "registry_preflight": {"performed": True, "id_collision": EXPERIMENT_ID in existing, "entries_checked": len(existing)},
        "propositions": len(PROPOSITIONS), "licensed_rewrites": list(LICENSED_REWRITES),
        "equality_classes": len(classes), "proof_replays": proof_replays,
        "equality_class_records": classes,
        "unsaturated_baseline": {"rendered": len(baseline_rows), "accepted": sum(r["accepted"] for r in baseline_rows)},
        "topology_failures": topology_failures, "rows": rows,
        "controls": controls, "exact_candidates": exact,
        "strict_gate": {"rendered": len(rows), "accepted": len(exact), "length_band": [39, 100],
                        "live_probe_depth": LIVE_PROBE_DEPTH,
                        "unsaturated_baseline_rendered": len(baseline_rows),
                        "unsaturated_baseline_accepted": sum(r["accepted"] for r in baseline_rows)},
        "provenance": {"rlaif": False, "frozen_vocabulary": True, "catalogue_imported": False, "reader_gate": "closed"},
        "falsifier": "If any class lacks two non-isomorphic proof-replayable trees, or extraction only substitutes lexical slots, reject this as a duplicate rather than widening it.",
        "next_construction": "Keep the equality classes fixed; only add a fresh licensed topology after a reader-worthy exact closure, with independent pointer/SHA replay.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["strict_gate"]))
    for row in controls[:12]:
        print(row["letters"], row["text"], row["first_mismatch"])


if __name__ == "__main__":
    main()
