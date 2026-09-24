"""Exhaustively compare two exact clause-pair operators on one frozen tape.

This program is self-contained for a sanitized remote run. It reads only the
frozen parent and historical relation-count snapshot, both JSON inputs, and
writes a new run record. The comparison is structural, not a readability
evaluation.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
NOVELTY_PATH = ROOT / "runs" / "incumbent-672-global-novelty-snapshot-20260922.json"
OUT = ROOT / "runs" / "comparison-568-online-residual-vs-offline-reverse-index-20260924.json.gz"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
ENTITIES = ("Nora", "Leon", "Aron", "Noel", "Mara", "Aram", "Nadia", "Aidan", "Liam", "Ira")
PREDICATES = ("stops", "spots", "sees")
DEPTH = 4
LEFT_CUT, RIGHT_CUT = 48, 520


def normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text).lower()


def raw_after_letters(text: str, count: int) -> int:
    found = 0
    for i, ch in enumerate(text):
        if ch.isascii() and ch.isalpha():
            found += 1
            if found == count:
                return i + 1
    raise ValueError(count)


def independent_audit(text: str) -> dict[str, object]:
    # Independent implementation: scan raw characters in place, without
    # calling normalize(), then compare the normalized tape in the opposite
    # direction and hash both views.
    tape = "".join(ch.lower() for ch in text if ch.isascii() and ch.isalpha())
    mismatch = None
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i
        if tape[i] != tape[j]:
            mismatch = {"left_offset": i, "right_offset": j,
                        "left": tape[i], "right": tape[j]}
            break
    fwd = hashlib.sha256(tape.encode("ascii")).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    normalized = normalize(text)
    return {
        "letters": len(tape), "outside_in_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch, "normalizer_reverse_equal": normalized == normalized[::-1],
        "sha256_forward": fwd, "sha256_reverse": rev, "sha_equal": fwd == rev,
    }


def load_frozen_inputs() -> tuple[str, dict[str, int], dict[str, object]]:
    parent = json.loads(PARENT_PATH.read_text())
    row = next(x for x in parent["rows"] if x["id"] == PARENT_ID)
    base = str(row["rendered"])
    parent_tape = normalize(base)
    if len(parent_tape) != 568 or hashlib.sha256(parent_tape.encode()).hexdigest() != PARENT_SHA:
        raise AssertionError("frozen parent digest/length mismatch")
    if parent_tape != parent_tape[::-1]:
        raise AssertionError("frozen parent is not exact")
    novelty = json.loads(NOVELTY_PATH.read_text())
    return base, dict(novelty["relation_counts"]), novelty


def relation(s: str, p: str, o: str) -> str:
    return f"{s.lower()} {p} {o.lower()}"


def frame(s: str, p: str, o: str) -> str:
    return f"{s.lower()}|{p}|{o.lower()}"


def clause_tape(s: str, p: str, o: str) -> str:
    return normalize(f"{s} {p} {o}")


def build_clauses() -> list[dict[str, str]]:
    out = []
    for s in ENTITIES:
        for p in PREDICATES:
            for o in ENTITIES:
                tape = clause_tape(s, p, o)
                out.append({"subject": s, "predicate": p, "object": o,
                            "relation": relation(s, p, o), "frame": frame(s, p, o),
                            "surface": f"{s} {p} {o}.", "tape": tape})
    return out


def online_matches(left: dict[str, str], object_constraint: str | None,
                   clauses: list[dict[str, str]], counters: Counter) -> list[dict[str, str]]:
    frontier = [c for c in clauses if object_constraint is None or c["object"] == object_constraint]
    for offset, emitted in enumerate(left["tape"]):
        counters["character_steps"] += len(frontier)
        frontier = [c for c in frontier
                    if offset < len(c["tape"]) and c["tape"][::-1][offset] == emitted]
        counters["frontier_empty_events"] += not bool(frontier)
        if not frontier:
            return []
    return [c for c in frontier if len(c["tape"]) == len(left["tape"])]


def candidate_key(left_chain: list[dict[str, str]], right_reverse: list[dict[str, str]]) -> str:
    return "||".join([*(c["frame"] for c in left_chain), "--",
                      *(c["frame"] for c in right_reverse)])


def enumerate_arm(arm: str, base: str, clauses: list[dict[str, str]],
                  prior: dict[str, int], novelty: dict[str, object]) -> dict[str, object]:
    counters: Counter = Counter()
    eligible = [c for c in clauses if c["subject"] != c["object"]
                and c["tape"] != c["tape"][::-1]]
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for c in eligible:
        by_subject[c["subject"]].append(c)
    match_cache: dict[tuple[str, str | None], list[dict[str, str]]] = {}
    if arm == "offline_reverse_pair_index":
        # Index by exact reversed letter tape; object continuity is applied at
        # lookup time. This stores all complete clause pairs before any chain.
        index: dict[str, list[dict[str, str]]] = defaultdict(list)
        for c in clauses:
            index[c["tape"][::-1]].append(c)
        counters["indexed_clause_rows"] = sum(len(v) for v in index.values())
        counters["exact_reverse_pair_edges"] = sum(
            len(index.get(c["tape"], [])) for c in eligible
        )
        def matches(left: dict[str, str], object_constraint: str | None) -> list[dict[str, str]]:
            cache_key = (left["frame"], object_constraint)
            if cache_key in match_cache:
                return match_cache[cache_key]
            counters["pair_index_lookups"] += 1
            result = [c for c in index.get(left["tape"], [])
                      if object_constraint is None or c["object"] == object_constraint]
            match_cache[cache_key] = result
            return result
    elif arm == "online_character_residual":
        def matches(left: dict[str, str], object_constraint: str | None) -> list[dict[str, str]]:
            cache_key = (left["frame"], object_constraint)
            if cache_key in match_cache:
                return match_cache[cache_key]
            counters["residual_lookups"] += 1
            candidates = [c for c in eligible
                          if object_constraint is None or c["object"] == object_constraint]
            result = online_matches(left, object_constraint, candidates, counters)
            match_cache[cache_key] = result
            return result
    else:
        raise ValueError(arm)

    records: dict[str, dict[str, object]] = {}
    rejected: Counter = Counter()
    # Constrained inventory and all admission gates are shared across arms.
    def visit(left_chain: list[dict[str, str]], right_reverse: list[dict[str, str]]) -> None:
        if len(left_chain) == DEPTH:
            right_chain = list(reversed(right_reverse))
            if any(right_chain[i]["object"] != right_chain[i + 1]["subject"]
                   for i in range(DEPTH - 1)):
                rejected["right_connectivity"] += 1
                return
            all_rel = [c["relation"] for c in left_chain + right_chain]
            if len(set(all_rel)) != 2 * DEPTH:
                rejected["distinct_event_gate"] += 1
                return
            if any(prior.get(r, 0) for r in all_rel):
                rejected["historical_novelty_gate"] += 1
                return
            if len({c["predicate"] for c in left_chain}) < 2:
                rejected["predicate_diversity_gate"] += 1
                return
            key = candidate_key(left_chain, right_reverse)
            if key in records:
                rejected["duplicate_candidate"] += 1
                return
            left_text = " ".join(c["surface"] for c in left_chain)
            right_text = " ".join(c["surface"] for c in right_chain)
            l_tape, r_tape = normalize(left_text), normalize(right_text)
            if l_tape != r_tape[::-1]:
                raise AssertionError("paired clause equation failed before rendering")
            left_raw = raw_after_letters(base, LEFT_CUT) + 1
            right_raw = raw_after_letters(base, RIGHT_CUT) + 1
            rendered = base[:left_raw] + " " + left_text + base[left_raw:right_raw] + " " + right_text + base[right_raw:]
            audit = independent_audit(rendered)
            if not (audit["outside_in_exact"] and audit["normalizer_reverse_equal"] and audit["sha_equal"]):
                raise AssertionError(f"independent full candidate audit failed: {key}")
            toks = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", rendered.lower())
            trigrams = [tuple(toks[i:i + 3]) for i in range(max(0, len(toks) - 2))]
            tri_counts = Counter(trigrams)
            tri_excess = sum(n - 1 for n in tri_counts.values() if n > 1)
            sentence_surfaces = [c["surface"].lower() for c in left_chain + right_chain]
            records[key] = {
                "rendered": rendered,
                "audit": audit,
                "provenance": {
                    "parent_artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
                    "parent_id": PARENT_ID, "parent_sha256": PARENT_SHA,
                    "method_arm": arm, "seam_normalized_cuts": [LEFT_CUT, RIGHT_CUT],
                    "left_chain": [c["frame"] for c in left_chain],
                    "right_chain_rendered_order": [c["frame"] for c in right_chain],
                    "right_reverse_pairing_order": [c["frame"] for c in right_reverse],
                    "novelty_snapshot_id": novelty["snapshot_id"],
                    "novelty_snapshot_commit": novelty["snapshot_commit"],
                    "novelty_scope_note": "membership against the frozen historical snapshot; not a claim of global originality",
                },
                "structural_metrics": {
                    "letters": audit["letters"], "growth_over_parent": audit["letters"] - 568,
                    "word_count": len(toks), "unique_word_count": len(set(toks)),
                    "unique_inserted_relations": len(set(all_rel)),
                    "inserted_relation_count": len(all_rel),
                    "left_predicate_count": len({c["predicate"] for c in left_chain}),
                    "repeated_trigram_excess": tri_excess,
                    "trigram_occurrences": len(trigrams),
                    "repeated_trigram_rate": tri_excess / len(trigrams) if trigrams else 0.0,
                    "duplicate_inserted_clause_count": len(sentence_surfaces) - len(set(sentence_surfaces)),
                    "proper_palindromic_tokens_present": any(t == t[::-1] for t in toks),
                },
            }
            return
        left_subject = left_chain[-1]["object"] if left_chain else None
        object_constraint = right_reverse[-1]["subject"] if right_reverse else None
        left_options = eligible if left_subject is None else by_subject[left_subject]
        for left in left_options:
            counters["left_clause_expansions"] += 1
            if left_subject is not None and left["subject"] != left_subject:
                continue
            if any(left["relation"] == old["relation"] for old in left_chain):
                continue
            counterparts = matches(left, object_constraint)
            counters["matched_counterpart_rows"] += len(counterparts)
            for right in counterparts:
                if right["subject"] == right["object"] or right["tape"] == right["tape"][::-1]:
                    continue
                if any(right["relation"] == old["relation"] for old in right_reverse):
                    continue
                visit(left_chain + [left], right_reverse + [right])

    visit([], [])
    sorted_rows = [records[k] for k in sorted(records)]
    serial_keys = sorted(records)
    digest = hashlib.sha256("\n".join(serial_keys).encode()).hexdigest()
    return {
        "arm": arm, "exhaustive": True, "depth": DEPTH,
        "stats": {"accepted_candidates": len(sorted_rows), "rejection_counts": dict(sorted(rejected.items())),
                  "operator_counters": dict(counters), "candidate_key_sha256": digest},
        "candidates": sorted_rows,
    }


def main() -> None:
    base, prior, novelty = load_frozen_inputs()
    clauses = build_clauses()
    eligible = [c for c in clauses if c["subject"] != c["object"] and c["tape"] != c["tape"][::-1]]
    online = enumerate_arm("online_character_residual", base, clauses, prior, novelty)
    offline = enumerate_arm("offline_reverse_pair_index", base, clauses, prior, novelty)
    online_keys = sorted(x["provenance"]["left_chain"] and (
        "||".join(x["provenance"]["left_chain"] + ["--"] + x["provenance"]["right_reverse_pairing_order"])
    ) for x in online["candidates"])
    offline_keys = sorted(x["provenance"]["left_chain"] and (
        "||".join(x["provenance"]["left_chain"] + ["--"] + x["provenance"]["right_reverse_pairing_order"])
    ) for x in offline["candidates"])
    if online_keys != offline_keys:
        raise AssertionError("candidate sets differ between exhaustive arms")
    online_summary = {k: v for k, v in online.items() if k != "candidates"}
    offline_summary = {k: v for k, v in offline.items() if k != "candidates"}
    candidate_rows = online["candidates"]
    for candidate in candidate_rows:
        candidate["provenance"]["method_arms"] = ["online_character_residual", "offline_reverse_pair_index"]
        candidate["provenance"].pop("method_arm", None)
    payload = {
        "experiment_id": "comparison-568-online-residual-vs-offline-reverse-index-20260924",
        "scope": "structural construction comparison only; no readability conclusion",
        "parent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
                   "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA},
        "grammar": {"entities": list(ENTITIES), "predicates": list(PREDICATES),
                    "clauses_before_gates": len(clauses), "clauses_after_subject_object_and_self_palindrome_gates": len(eligible)},
        "shared_protocol": {"seam_normalized_cuts": [LEFT_CUT, RIGHT_CUT], "depth": DEPTH,
            "left_and_right_object_subject_continuity": True,
            "historical_novelty_gate": "all eight relation strings absent in frozen snapshot counts",
            "distinct_event_gate": "eight distinct subject-predicate-object relation triples",
            "other_gates": ["subject != object", "reject self-palindromic clause tapes", "at least two predicates on left chain", "complete SVO clauses"],
            "enumeration": "exhaustive DFS of every connected left chain and every exact reverse-clause counterpart path"},
        "novelty_snapshot": {"snapshot_id": novelty["snapshot_id"], "snapshot_commit": novelty["snapshot_commit"],
            "manifest_sha256": novelty["manifest_sha256"], "chronology": novelty.get("note", "frozen historical relation counts")},
        "candidate_sets_match": True,
        "arms": {"online": online_summary, "offline": offline_summary},
        "accepted_candidates": candidate_rows,
    }
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"artifact": str(OUT.relative_to(ROOT)),
        "clauses": len(clauses), "eligible_clauses": len(eligible),
        "online_candidates": len(online["candidates"]), "offline_candidates": len(offline["candidates"]),
        "candidate_sets_match": True,
        "online_stats": online["stats"], "offline_stats": offline["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
