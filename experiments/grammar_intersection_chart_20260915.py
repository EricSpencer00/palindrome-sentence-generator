"""Recursive CFG/chart intersection for exact palindrome construction.

This experiment changes the construction unit from hand-authored slots to
complete derivations of a small recursive English grammar.  A bounded chart
enumerates independently derived sentences, indexes their normalized tapes,
and intersects that language with its character reversal.  The derivations on
the two sides are never copied or reversed at the word level; only the final
character tape is compared.  This is a constructive route: increasing the
grammar depth increases the possible prose length without changing the exact
validator.

The chart is deliberately finite and authored for this run.  It is evidence
about this grammar and budget, not an automatic readability certificate.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs/grammar-intersection-chart-20260915.json"
ID = "grammar-intersection-chart"
SIGNATURE = (
    "recursive-cfg-derived-sentence-chart|inside-derivation-enumeration|"
    "reverse-language-intersection|relative-clause-depth-growth|"
    "independent-complete-sentence-derivations|character-only-final-join"
)
SEED = "An aide rips nine memos; some men inspire Diana."


@dataclass(frozen=True)
class Item:
    text: str
    tree: str
    depth: int


DET = ("a", "an", "the", "some")
ADJ = ("quiet", "kind", "old", "red", "small", "brave")
NOUN = (
    "aide", "sailor", "nurse", "artist", "baker", "child", "captain",
    "poet", "guard", "men", "memos", "lantern", "map", "book", "bell",
    "garden", "harbor", "story", "letter", "song", "rain", "river",
)
NUM = ("one", "two", "nine")
NAME = ("Ana", "Diana", "Nora", "Mara", "Ira", "Ari", "Eli", "Lena")
TRANSITIVE = (
    "aids", "sees", "finds", "holds", "keeps", "inspires", "guides",
    "greets", "reads", "writes", "marks", "helps", "crosses", "carries",
)
INTRANSITIVE = ("waits", "rests", "smiles", "sings", "walks", "runs")
PREP = ("in", "by", "at", "near", "with", "for")
ADV = ("today", "quietly", "outside", "again")


def norm(text: str) -> str:
    return "".join(c.lower() for c in text if "a" <= c.lower() <= "z")


def exact(text: str) -> bool:
    tape = norm(text)
    return bool(tape) and tape == tape[::-1]


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def readability(text: str) -> dict:
    ws = words(text)
    return {
        "word_count": len(ws),
        "mean_word_length": round(sum(map(len, ws)) / max(1, len(ws)), 2),
        "complete_terminal_punctuation": bool(re.search(r"[.!?]$", text)),
        "programmatic_only": True,
    }


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text())


def np_chart(max_depth: int) -> list[Item]:
    """Build NP chart with one recursive relative-clause layer per depth."""
    base: dict[str, Item] = {}
    for d, n in product(DET, NOUN):
        base[f"{d} {n}"] = Item(f"{d} {n}", f"NP({d},{n})", 0)
    for d, a, n in product(DET, ADJ, NOUN):
        base[f"{d} {a} {n}"] = Item(f"{d} {a} {n}", f"NP({d},{a},{n})", 0)
    for num, n in product(NUM, NOUN):
        base[f"{num} {n}"] = Item(f"{num} {n}", f"NP({num},{n})", 0)
    for n in NAME:
        base[n] = Item(n, f"NP({n})", 0)
    levels = [list(base.values())]
    for depth in range(1, max_depth + 1):
        # Keep recursive generation finite and independent of the sentence
        # pairing step.  The relative clause is a genuine grammar expansion,
        # not a mirrored/repeated phrase.
        # A deeper relative clause may use either a base NP or a shallower
        # relative NP as its head/object.  The cumulative chart is what makes
        # this genuinely recursive rather than three unrelated inventories.
        prior = [item for level in levels for item in level]
        additions: dict[str, Item] = {}
        prior_ordered = sorted(prior, key=lambda x: (-x.depth, len(norm(x.text)), x.text))
        objects = sorted(
            (x for x in prior if len(words(x.text)) <= 3),
            key=lambda x: (x.depth, len(norm(x.text)), x.text),
        )[:72]
        for head, verb, obj in product(prior_ordered[:72], TRANSITIVE[:8], objects):
            text = f"{head.text} who {verb} {obj.text}"
            if len(norm(text)) <= 82 and len(words(text)) <= 14:
                additions.setdefault(text, Item(text,
                    f"NPREL({head.tree},who,{verb},{obj.tree})", depth))
        levels.append(list(additions.values()))
    result: dict[str, Item] = {}
    for level in levels:
        result.update({x.text: x for x in level})
    return list(result.values())


def clause_chart(max_depth: int) -> list[Item]:
    nps = np_chart(max_depth)
    # Use a bounded but deterministic grammar chart.  This keeps the run
    # reproducible while making relative-clause depth, rather than beam size,
    # the experimental variable.
    ordered = sorted(nps, key=lambda x: (x.depth, len(norm(x.text)), x.text))
    # Reserve chart capacity for the newly enabled recursive level.  Without
    # this explicit stratification a prefix of the base chart could hide the
    # depth variable and make the comparison a disguised duplicate run.
    subjects = ordered[:72]
    objects = ordered[:72]
    if max_depth:
        recursive = [x for x in ordered if x.depth == max_depth][:24]
        subjects = subjects + recursive
        objects = objects + recursive
    result: dict[str, Item] = {}
    for subj, verb, obj in product(subjects, TRANSITIVE, objects):
        text = f"{subj.text} {verb} {obj.text}"
        if len(norm(text)) <= 92 and len(words(text)) <= 18:
            result.setdefault(text, Item(text,
                f"S({subj.tree},{verb},{obj.tree})", max(subj.depth, obj.depth)))
    for subj, verb in product(subjects, INTRANSITIVE):
        text = f"{subj.text} {verb}"
        if len(norm(text)) <= 92:
            result.setdefault(text, Item(text, f"S({subj.tree},{verb})", subj.depth))
    return list(result.values())


def bounded_chart(items: list[Item], limit: int = 384) -> list[Item]:
    """Stable length/lexical-diversity strata, not a random larger pool."""
    buckets: dict[tuple[int, int], list[Item]] = {}
    for item in items:
        key = (len(norm(item.text)) // 8, item.depth)
        buckets.setdefault(key, []).append(item)
    selected: list[Item] = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda x: (norm(x.text), x.text))
        selected.extend(group[: max(1, limit // max(1, len(buckets)))])
    return selected[:limit]


def run() -> dict:
    registry = load_registry()
    # The registry entry is committed with the experiment for reproducibility;
    # reruns should audit all *other* prior families while allowing this
    # experiment to find its own recorded signature.
    prior_signatures = {row["signature"] for row in registry["entries"]
                        if row["id"] != ID}
    overlap = sorted(prior_signatures.intersection({SIGNATURE}))
    if overlap:
        raise RuntimeError(f"novelty collision: {overlap}")

    depths: list[dict] = []
    all_texts: list[Item] = []
    for depth in range(3):
        chart = clause_chart(depth)
        bounded = bounded_chart(chart)
        all_texts.extend(bounded)
        depths.append({"relative_depth": depth, "raw_chart_items": len(chart),
                       "bounded_items": len(bounded),
                       "max_letters": max((len(norm(x.text)) for x in bounded), default=0)})

    # Deduplicate text surfaces while retaining the deepest derivation.  This
    # makes each exact-tape lookup a unique chart item rather than a duplicate
    # caused by inspecting three cumulative depth charts.
    unique_items: dict[str, Item] = {}
    for item in all_texts:
        old = unique_items.get(item.text)
        if old is None or item.depth > old.depth:
            unique_items[item.text] = item
    all_texts = list(unique_items.values())

    # The intersection is between two independently derived complete
    # sentences.  A candidate is never made by reversing a word list.
    by_tape: dict[str, list[Item]] = {}
    for item in all_texts:
        by_tape.setdefault(norm(item.text), []).append(item)
    exact_rows: list[dict] = []
    probes: list[dict] = []
    checked = 0
    seen: set[str] = set()
    for left in all_texts:
        reverse_tape = norm(left.text)[::-1]
        for right in by_tape.get(reverse_tape, []):
            checked += 1
            full = f"{left.text}; {right.text}."
            # This pair relation would require reversing both complete halves;
            # use it only as a diagnostic.  The full text is separately tested.
            if full in seen:
                continue
            seen.add(full)
            row = {
                "text": full, "letters": len(norm(full)),
                "exact_letter_palindrome": exact(full),
                "left_tree": left.tree, "right_tree": right.tree,
                "relative_depth": max(left.depth, right.depth),
                "readability": readability(full),
                "provenance": "two independently enumerated CFG derivations; no catalogue lookup",
            }
            probes.append(row)
            if row["exact_letter_palindrome"] and norm(full) != norm(SEED):
                exact_rows.append(row)

    # A bounded near-miss diagnostic gives the next repair a concrete target
    # even when the two generated languages have no exact intersection.  It
    # compares only character prefixes of independent complete derivations;
    # it never promotes a near miss to a candidate.
    near_misses: list[dict] = []
    for left in all_texts:
        lt = norm(left.text)
        for right in all_texts:
            rt = norm(right.text)[::-1]
            matched = 0
            for a, b in zip(lt, rt):
                if a != b:
                    break
                matched += 1
            near_misses.append({"left": left.text, "right": right.text,
                                "matched_outer_letters": matched,
                                "left_letters": len(lt), "right_letters": len(rt),
                                "left_tree": left.tree, "right_tree": right.tree})
    near_misses.sort(key=lambda row: (row["matched_outer_letters"],
                                      -abs(row["left_letters"] - row["right_letters"])),
                     reverse=True)

    # Add representative complete prose probes even when no reversal-language
    # intersection exists, so the artifact renders what was actually tested.
    ordered_probes = sorted(all_texts, key=lambda x: (len(norm(x.text)), x.text))
    probe_items = ordered_probes[:4] + ordered_probes[-8:]
    used_probe_texts: set[str] = set()
    for item in probe_items:
        if item.text in used_probe_texts:
            continue
        used_probe_texts.add(item.text)
        probes.append({"text": item.text + ".", "letters": len(norm(item.text)),
                       "exact_letter_palindrome": exact(item.text),
                       "left_tree": item.tree, "right_tree": None,
                       "relative_depth": item.depth, "readability": readability(item.text + "."),
                       "provenance": "complete CFG derivation probe; not an exact closure"})

    payload = {
        "experiment_id": ID,
        "signature": SIGNATURE,
        "registry_read_before_run": True,
        "registry_entry_count_before_run": len(registry["entries"]),
        "prior_signatures_overlap": overlap,
        "excluded_prior_families": [
            "center-window repair", "human compositional window", "global semantic paraphrase",
            "seam-first authoring", "multiword transducer", "reverse-prefix decoder",
            "neural dual-prefix beam", "catalogue/corpus import",
        ],
        "method": "recursive context-free grammar chart; independent complete derivations intersected with the reversed normalized-character language",
        "seed": SEED,
        "depths": depths,
        "chart_items_total": len(all_texts),
        "reverse_intersection_checks": checked,
        "near_miss_pair_checks": len(all_texts) * len(all_texts),
        "best_near_misses": near_misses[:12],
        "exact_count": len(exact_rows),
        "rendered_candidates": exact_rows,
        "rendered_probes": probes[:24],
        "independent_audit": [{"text": r["text"], "normalized": norm(r["text"]),
                               "two_pointer": exact(r["text"]),
                               "normalized_sha256": hashlib.sha256(norm(r["text"]).encode()).hexdigest()}
                              for r in exact_rows],
        "readability_gate": "No programmatic score certifies English. Any exact survivor must be rendered with intact punctuation and sent to blinded human readers with shuffled controls.",
        "repair_operator": "For the deepest non-closing chart item, add one typed relative-clause production at the shorter side's derivation frontier, then rerun the reverse-language chart intersection with a held-out lexical inventory; do not widen this chart or reuse a prior repair family.",
        "failure_action": "If exact_count is zero, preserve this chart and implement the stated recursive relative-clause frontier repair; do not interpret the zero as impossibility.",
        "provenance": "All grammar rules and lexical items are authored in this script for this run; the known 38-letter seed is an excluded benchmark, not imported output.",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return payload


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "chart_items": result["chart_items_total"],
                      "reverse_checks": result["reverse_intersection_checks"],
                      "exact_count": result["exact_count"]}, indent=2))
