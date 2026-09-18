"""Cross-attachment residual repair with a seeded free-centre product.

The fixed imperative action/object/location core can select a different
semantically licensed purpose attachment from the same frame family.  The
selected surface is compiled as one connected grammar, seeded only with its
actual matched outer characters, and then expanded through the remaining
character product.  The inside product accepts an even centre or a one-letter
centre anywhere and applies the online proper-island prune at every state.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import compile_slots
from experiments.progressive_phrase_residual_product_20260913 import online_span_loss
from experiments.residual_aware_endpoint_expansion_20260913 import (
    CORES, Core, OBJECT_ADJECTIVES, PLACE_ADJECTIVES, PLACES, PURPOSE_OBJECT_ADJECTIVES,
    PURPOSE_OPTIONS, ACTION_OBJECTS, article,
)
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 30, 160


def frame_family_key(core: Core) -> tuple[str, str, str, str, str]:
    return (core.action, core.object, core.object_adj, core.place, core.place_adj)


def attachment_pairs(cores: tuple[Core, ...] = CORES) -> tuple[dict, ...]:
    """Join distinct purpose attachments only within a typed core family."""
    families = defaultdict(list)
    for core in cores: families[frame_family_key(core)].append(core)
    rows = []
    for family, options in families.items():
        for source in options:
            opening = "".join(source.words[:3])
            for selected in options:
                if selected.identifier == source.identifier: continue
                terminal = "".join(selected.words[-4:])
                depth = 0
                for left, right in zip(opening, terminal[::-1]):
                    if left != right: break
                    depth += 1
                if depth < 3: continue
                rows.append({"source_core": source.identifier, "selected_core": selected.identifier,
                             "family": list(family), "opening_words": list(source.words[:3]),
                             "selected_terminal_words": list(selected.words[-4:]),
                             "matched_pairs": depth, "opening_letters": opening,
                             "reversed_terminal_letters": terminal[::-1],
                             "attachment_changed": True, "semantic_roles": ["agent_action", "purpose_object"]})
    return tuple(rows)


def path_for_words(grammar, words):
    cursor = grammar.start; path = []
    for slot, word in enumerate(words):
        if word not in grammar.slots[slot]: raise AssertionError("word not in compiled slot")
        for offset, char in enumerate(word):
            options = [edge for edge in grammar.edges if edge.source == cursor and edge.char == char]
            edge = next((edge for edge in options if offset < len(word)-1 or edge.completed_word == word), None)
            if edge is None: raise AssertionError("surface cannot replay through grammar")
            path.append(edge); cursor = edge.target
    if cursor != grammar.end: raise AssertionError("path did not end")
    return tuple(path)


def inside_product(grammar, words, matched_pairs, max_states=100_000):
    path = path_for_words(grammar, words)
    prefix = path[:matched_pairs]; suffix = path[-matched_pairs:]
    left = prefix[-1].target; right = suffix[0].source
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in grammar.edges: outgoing[edge.source].append(edge); incoming[edge.target].append(edge)
    reachable_cache = {}
    def reachable(node):
        if node not in reachable_cache:
            reachable_cache[node] = {node}
            for edge in outgoing[node]: reachable_cache[node].update(reachable(edge.target))
        return reachable_cache[node]
    reachable(grammar.start)
    stack = [(left, right, prefix, suffix, tuple((i + 1, path[i].char) for i in range(matched_pairs)))]
    seen, records, ledgers = set(), [], []
    states = 0; pruned = 0
    while stack and states < max_states:
        left, right, prefix, suffix, pairs = stack.pop()
        key = (left, right, tuple((e.source, e.target, e.char) for e in prefix), tuple((e.source, e.target, e.char) for e in suffix))
        if key in seen: continue
        seen.add(key); states += 1
        ledger = {"matched_pairs": len(pairs), "left_node": left, "right_node": right,
                  "pair_trace": [[i, c] for i, c in pairs], "remaining_edges": max(0, len(path) - len(prefix) - len(suffix))}
        ledgers.append(ledger)
        prune = online_span_loss(grammar, left, right, initial=False)
        if prune is not None:
            pruned += 1; ledger["online_span_prune"] = prune
            continue
        middles = [()] if left == right else []
        middles += [(edge,) for edge in outgoing[left] if edge.target == right]
        for middle in middles:
            candidate = prefix + middle + suffix
            if len(candidate) != len(path): continue
            surface = "".join(edge.char for edge in candidate)
            if surface == surface[::-1]:
                records.append({"words": list(words), "letters": len(surface), "exact": True,
                                "center_characters": len(middle), "midpoint_letter_offset": len(prefix),
                                "inside_ledger": ledger})
        for first in outgoing[left]:
            for last in incoming[right]:
                if first.char == last.char and last.source in reachable(first.target):
                    stack.append((first.target, last.source, prefix + (first,), (last,) + suffix,
                                  pairs + ((len(pairs) + 1, first.char),)))
    return {"states": states, "pending_states": len(stack), "states_exhausted": not stack,
            "truncated": bool(stack), "seed_pairs": matched_pairs,
            "deepest_pairs": max((len(row["pair_trace"]) for row in ledgers), default=matched_pairs),
            "online_span_pruned": pruned, "ledgers": ledgers, "records": records}


INDEPENDENT_ACTION_OBJECTS = {"draft": {"letter", "report"}, "edit": {"letter", "report"},
                              "file": {"letter", "report"}, "make": {"model", "canvas"}}
INDEPENDENT_PURPOSE_OBJECTS = {
    ("draft", "letter", "make"): {"card", "note"}, ("draft", "letter", "write"): {"guide"},
    ("draft", "report", "share"): {"guide", "note"}, ("draft", "report", "file"): {"report"},
    ("edit", "letter", "send"): {"note"}, ("edit", "letter", "write"): {"guide"},
    ("edit", "report", "write"): {"guide"}, ("edit", "report", "share"): {"note"},
    ("edit", "report", "track"): {"tide"}, ("file", "letter", "send"): {"note"},
    ("file", "report", "share"): {"guide"}, ("make", "model", "show"): {"model"},
    ("make", "canvas", "show"): {"canvas"},
}
INDEPENDENT_OBJECTS = frozenset({"letter", "report", "model", "canvas", "card", "note", "guide", "tide"})
INDEPENDENT_PLACES = frozenset(PLACES)


def independent_parse(text):
    tokens = tuple(__import__("re").findall(r"[a-z]+", text.lower()))
    if len(tokens) != 13: return {"ok": False, "reason": "wrong_role_arity"}
    action, od, oa, obj, prep, pd, pa, place, to, purpose, pod, poa, pobj = tokens
    article_ok = od == article(oa) and pd == article(pa) and pod == article(poa)
    role_ok = (action in INDEPENDENT_ACTION_OBJECTS and obj in INDEPENDENT_ACTION_OBJECTS[action]
               and obj in INDEPENDENT_OBJECTS and oa in set(OBJECT_ADJECTIVES)
               and prep == "in" and pa in set(PLACE_ADJECTIVES) and place in INDEPENDENT_PLACES
               and to == "to" and poa in set(PURPOSE_OBJECT_ADJECTIVES) and pobj in INDEPENDENT_OBJECTS
               and pobj in INDEPENDENT_PURPOSE_OBJECTS.get((action, obj, purpose), set()))
    return {"ok": article_ok and role_ok, "agreement_ok": article_ok, "valency_ok": role_ok,
            "tokens": list(tokens), "independent_inventory": True}


def audit(core, join, product):
    text = render(core.words); ledger = outside_in(text); parsed = independent_parse(text)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [k for k, v in central.items() if not v]
    if not ledger["exact"]: codes.append("outside_in_ledger_mismatch")
    if not parsed["ok"]: codes.append("independent_complete_reparse_failed")
    return {"record_kind": "cross_attachment_inside_product", "source_core": join["source_core"],
            "selected_core": core.identifier, "rendered": text, "endpoint_join": join,
            "outside_in_ledger": ledger, "inside_product": product, "independent_parse": parsed,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def outside_in(text):
    tape = normalize_letters(text); events = []
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i; event = {"pair": i + 1, "left": tape[i], "right": tape[j], "equal": tape[i] == tape[j]}; events.append(event)
        if not event["equal"]: return {"exact": False, "letters": len(tape), "events": events, "first_mismatch": event, "normalized_sha256": sha256(tape.encode()).hexdigest()}
    return {"exact": bool(tape), "letters": len(tape), "events": events, "first_mismatch": None, "normalized_sha256": sha256(tape.encode()).hexdigest()}


def render(words):
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def run(max_states=100_000):
    pairs = attachment_pairs(); by_id = {core.identifier: core for core in CORES}; records = []; residual_diagnostics = []; inside_runs = []; inner_rejections = []
    channels = defaultdict(lambda: {"eligible": 0, "scheduled": 0, "truncated": False, "deepest": 0})
    for join in pairs:
        channel = join["opening_prefix"] if "opening_prefix" in join else join["opening_letters"][:3]
        channels[channel]["eligible"] += 1
        if join["matched_pairs"] < 6:
            if len(residual_diagnostics) < 200:
                selected = by_id[join["selected_core"]]
                residual_diagnostics.append({"record_kind": "cross_attachment_residual_below_inner_threshold",
                                             "source_core": join["source_core"], "selected_core": join["selected_core"],
                                             "rendered": render(selected.words), "endpoint_join": join,
                                             "outside_in_ledger": outside_in(render(selected.words)),
                                             "rejection": "inner_product_requires_six_outer_pairs"})
            continue
        if channels[channel]["scheduled"] >= max_states:
            channels[channel]["truncated"] = True; continue
        selected = by_id[join["selected_core"]]; grammar = compile_slots(tuple((word,) for word in selected.words))
        product_result = inside_product(grammar, selected.words, join["matched_pairs"], max_states=max_states)
        channels[channel]["scheduled"] += 1; channels[channel]["deepest"] = max(channels[channel]["deepest"], product_result["deepest_pairs"])
        inside_runs.append({"source_core": join["source_core"], "selected_core": selected.identifier,
                            "endpoint_pairs": join["matched_pairs"],
                            "product": {key: value for key, value in product_result.items() if key != "records"}})
        if not product_result["records"] and len(inner_rejections) < 100:
            inner_rejections.append({"source_core": join["source_core"], "selected_core": selected.identifier,
                                     "rendered": render(selected.words), "endpoint_pairs": join["matched_pairs"],
                                     "outside_in_ledger": outside_in(render(selected.words)),
                                     "product_rejection": {key: value for key, value in product_result.items() if key != "records"}})
        for rec in product_result["records"]: records.append(audit(selected, join, rec))
    admitted = [row for row in records if row["mechanically_admitted"]]
    return {"status": "cross_attachment_inside_product", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
        "minimum_endpoint_pairs": 3, "minimum_pairs_before_inside_product": 6, "cross_core_attachment_selection": True,
        "partial_word_residual_index": True, "free_even_odd_center": True, "online_span_prune": True,
        "independent_complete_reparse": True, "corpus_generation": False, "human_readability_required_after_admission": True},
        "authored_core_count": len(CORES), "attachment_pair_count": len(pairs), "channel_coverage": dict(channels),
        "inner_product_runs": inside_runs, "inner_rejection_ledgers": inner_rejections,
        "residual_diagnostics": residual_diagnostics,
        "records": records, "exact_candidates": records, "admitted_candidates": admitted,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "material": "authored modular purpose attachments; no catalogue or fixed output",
                       "known_catalogue_check": "central admission gates applied"},
        "next_construction_operator": "Add a semantically licensed attachment family whose indexed residual extends a distinct core beyond the current deepest pair, preserving cross-core provenance.",
        "reader_facing_next_test": "Only an admitted exact surface may enter randomized blinded intact-prose versus shuffled-control reading; programmatic checks do not certify readability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", required=True, type=Path); parser.add_argument("--max-states", type=int, default=100_000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True); result = run(args.max_states); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "attachments": result["attachment_pair_count"], "exact": len(result["exact_candidates"]), "admitted": len(result["admitted_candidates"])}, indent=2))


if __name__ == "__main__": main()
