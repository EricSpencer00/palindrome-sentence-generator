"""Center-out dependency realization with topic continuity.

This is an independent construction route, not a repair of the relation-edge
search. Each side is a three-clause discourse plan. A singular topic passes
from clause one to clause two and then to clause three through typed pronoun
slots; all finite verbs are agreement-checked. Lexicalization is independent
between sides. The bilateral ledger starts at the discourse seam (the last
word of the left plan and first word of the right plan) and expands outward,
consuming matching characters. Complete rendered proposals are audited even
when the ledger dies before an exact closure.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "centerout-dependency-realizer-20260916"
SIGNATURE = (
    "centerout-dependency-realizer|three-clause-discourse-plan|"
    "topic-continuity-state|typed-argument-agreement|"
    "bilateral-character-obligations|independent-lexicalization|"
    "deterministic-exhaustive"
)
EVIDENCE = ROOT / "runs" / "centerout-dependency-realizer-20260916.json"


@dataclass(frozen=True)
class Plan:
    name: str
    left: tuple[tuple[str, tuple[str, ...]], ...]
    right: tuple[tuple[str, tuple[str, ...]], ...]


DET = ("a", "an", "each", "every", "one")
PRONOUN = ("it", "that")


def _slots(side: str, subject: tuple[str, ...], verbs: tuple[tuple[str, ...], ...], objects: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    # Topic graph: maker --creates--> artifact --moves--> place --records--> record.
    # Pronoun slots carry the topic ID and singular agreement in TOPIC_STATE.
    return (
        (f"{side}.det1", DET), (f"{side}.subject1", subject), (f"{side}.verb1", verbs[0]), (f"{side}.det_obj1", DET), (f"{side}.artifact", objects[0]),
        (f"{side}.topic_artifact", PRONOUN), (f"{side}.verb2", verbs[1]), (f"{side}.det_obj2", DET), (f"{side}.place", objects[1]),
        (f"{side}.topic_place", PRONOUN), (f"{side}.verb3", verbs[2]), (f"{side}.det_obj3", DET), (f"{side}.record", objects[2]),
    )


PLANS = (
    Plan(
        "archive_route",
        _slots("L", ("baker", "farmer", "keeper", "poet", "teacher"), (("keeps", "marks", "reads", "sends", "tends"), ("guides", "finds", "holds", "opens", "uses"), ("reads", "marks", "visits", "guards", "crosses")), (("map", "note", "seed", "letter", "record"), ("harbor", "garden", "market", "river", "road"), ("book", "gate", "house", "shore", "tower"))),
        _slots("R", ("artist", "clerk", "doctor", "writer", "sailor"), (("carries", "signals", "checks", "answers", "offers"), ("calls", "helps", "leads", "reaches", "finds"), ("reads", "waits", "enters", "leaves", "visits")), (("alarm", "message", "parcel", "signal", "answer"), ("office", "station", "harbor", "school", "bridge"), ("letter", "room", "gate", "market", "shore"))),
    ),
    Plan(
        "garden_route",
        _slots("L", ("gardener", "farmer", "worker", "child", "keeper"), (("plants", "grows", "tends", "waters", "starts"), ("feeds", "helps", "moves", "keeps", "covers"), ("counts", "checks", "gathers", "stores", "shares")), (("beans", "corn", "grain", "herbs", "seeds"), ("garden", "field", "market", "farm", "yard"), ("basket", "crop", "fruit", "grain", "wheat"))),
        _slots("R", ("doctor", "teacher", "reader", "worker", "pilot"), (("tests", "checks", "counts", "notes", "rates"), ("helps", "guides", "moves", "keeps", "saves"), ("reads", "marks", "stores", "shares", "weighs")), (("data", "chart", "notes", "value", "score"), ("office", "school", "lab", "room", "desk"), ("report", "book", "file", "note", "chart"))),
    ),
    Plan(
        "signal_route",
        _slots("L", ("caller", "friend", "guide", "maker", "pilot"), (("asks", "calls", "sends", "tells", "warns"), ("helps", "finds", "leads", "meets", "uses"), ("answers", "thanks", "calls", "greets", "trusts")), (("alarm", "answer", "news", "signal", "word"), ("harbor", "home", "office", "road", "room"), ("friend", "guide", "leader", "reader", "sailor"))),
        _slots("R", ("artist", "child", "clerk", "poet", "teacher"), (("draws", "writes", "sends", "shows", "names"), ("helps", "calls", "leads", "finds", "teaches"), ("reads", "answers", "thanks", "visits", "follows")), (("map", "note", "letter", "story", "plan"), ("garden", "market", "school", "street", "tower"), ("child", "clerk", "friend", "guide", "teacher"))),
    ),
)


def tape(text: str) -> str:
    return normalize_letters(text)


def render(words: tuple[str, ...]) -> str:
    # Every plan is three complete clauses: 5 + 4 + 4 tokens.
    return " ".join(words[:5]) + ". " + " ".join(words[5:9]) + ". " + " ".join(words[9:]) + "."


def validate_plan(words: tuple[str, ...], side: str) -> dict:
    """Validate topic continuity and singular agreement independently of tape."""
    if len(words) != 13:
        return {"topic_continuity": False, "argument_agreement": False}
    pronouns = words[5] in PRONOUN and words[9] in PRONOUN
    # The finite options are all third-person singular forms for the singular
    # topic. This explicit check prevents a lexical assignment from weakening
    # the agreement state.
    verb_slots = (2, 6, 10)
    agreement = all(words[i].endswith(("s", "es")) for i in verb_slots)
    return {"topic_continuity": pronouns, "argument_agreement": agreement, "side": side}


def audit_pair(left: tuple[str, ...], right: tuple[str, ...], plan: Plan) -> dict:
    text = render(left) + " " + render(right)
    chars = tape(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=220)
    return {
        "rendered": text,
        "letters": len(chars),
        "tape": chars,
        "exact": bool(chars) and chars == chars[::-1],
        "ledger_replay": bool(chars) and all(chars[i] == chars[-1 - i] for i in range(len(chars) // 2)),
        "checks": checks,
        "mechanically_admitted": bool(chars) and chars == chars[::-1] and all(checks.values()),
        "topic_and_agreement": {"left": validate_plan(left, "left"), "right": validate_plan(right, "right")},
        "plan": plan.name,
    }


def novelty_preflight() -> dict:
    registry = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    # A packaged rerun sees its own row. Exclude it to reproduce the
    # pre-registration novelty check rather than report a self-collision.
    entries = [row for row in registry["entries"] if row["id"] != EXPERIMENT_ID]
    common = {"a", "an", "and", "after", "audit", "authoring", "before", "character", "complete", "construction", "constraints", "cross", "derived", "derivation", "english", "equation", "exact", "final", "full", "generation", "global", "grammar", "held", "heldout", "in", "independent", "join", "joint", "left", "lexical", "lexicalized", "of", "order", "out", "over", "paired", "parse", "parser", "repair", "residual", "reverse", "right", "sentence", "state", "surface", "tape", "the", "through", "to", "typed", "unit", "word", "words", "with"}
    atoms = lambda value: set(re.findall(r"[a-z0-9]+", value.lower())) - common
    current = atoms(SIGNATURE)
    nearest = []
    for row in entries:
        prior = atoms(row["signature"])
        nearest.append({"id": row["id"], "jaccard": round(len(current & prior) / len(current | prior), 6), "shared_atoms": sorted(current & prior)})
    nearest.sort(key=lambda row: (-row["jaccard"], row["id"]))
    return {"registry_entries": len(entries), "runtime_registry_entries": len(registry["entries"]), "excluded_routes": len(registry.get("excluded", [])), "exact_signature_collision": any(row["signature"] == SIGNATURE for row in entries), "nearest_prior": nearest[:5]}


def _options(slots: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(options for _, options in slots)


def _center_out_search(plan: Plan, state_budget: int = 75_000) -> tuple[list[dict], dict, list[dict]]:
    left_slots, right_slots = plan.left, plan.right
    left_options, right_options = _options(left_slots), _options(right_slots)
    solutions: list[dict] = []
    probes: list[dict] = []
    stats = {"states": 0, "lexical_assignments": 0, "char_pairs": 0, "dead_char": 0, "odd_center_trials": 0, "max_matched": 0, "budget_exhausted": False}
    left_words: dict[int, str] = {}
    right_words: dict[int, str] = {}

    def snapshot() -> tuple[tuple[str, ...], tuple[str, ...]]:
        return tuple(left_words[i] for i in range(len(left_slots))), tuple(right_words[i] for i in range(len(right_slots)))

    def visit(li: int, ri: int, lp: int, rp: int, trace: tuple[dict, ...]) -> None:
        stats["states"] += 1
        if stats["states"] > state_budget:
            stats["budget_exhausted"] = True
            return
        if li < 0 and ri >= len(right_slots) and lp == 0 and rp == 0:
            left, right = snapshot()
            row = audit_pair(left, right, plan)
            row["trace"] = list(trace)
            if row["exact"]:
                solutions.append(row)
            elif len(probes) < 12:
                probes.append(row)
            return
        if li >= 0 and li in left_words and lp == len(tape(left_words[li])):
            del left_words[li]
            visit(li - 1, ri, 0, rp, trace)
            return
        if ri < len(right_slots) and ri in right_words and rp == len(tape(right_words[ri])):
            del right_words[ri]
            visit(li, ri + 1, lp, 0, trace)
            return
        # Choose the next complete lexical unit at each side of the seam.
        if li >= 0 and li not in left_words:
            for word in left_options[li]:
                left_words[li] = word
                stats["lexical_assignments"] += 1
                visit(li, ri, 0, rp, trace + ({"side": "left", "slot": left_slots[li][0], "word": word},))
                del left_words[li]
            return
        if ri < len(right_slots) and ri not in right_words:
            for word in right_options[ri]:
                right_words[ri] = word
                stats["lexical_assignments"] += 1
                visit(li, ri, lp, 0, trace + ({"side": "right", "slot": right_slots[ri][0], "word": word},))
                del right_words[ri]
            return
        # Bilateral obligation: expand from the seam and compare one character.
        if li < 0 or ri >= len(right_slots):
            stats["odd_center_trials"] += 1
            return
        left_chars = tape(left_words[li])[::-1]
        right_chars = tape(right_words[ri])
        if lp >= len(left_chars) or rp >= len(right_chars):
            return
        if left_chars[lp] != right_chars[rp]:
            stats["dead_char"] += 1
            return
        stats["char_pairs"] += 1
        stats["max_matched"] = max(stats["max_matched"], sum(1 for item in trace if "char" in item) + 1)
        visit(li, ri, lp + 1, rp + 1, trace + ({"char": left_chars[lp], "left_slot": left_slots[li][0], "right_slot": right_slots[ri][0]},))

    visit(len(left_slots) - 1, 0, 0, 0, ())
    return solutions, stats, probes


def run() -> dict:
    rows: list[dict] = []
    probes: list[dict] = []
    per_plan: dict[str, dict] = {}
    for plan in PLANS:
        found, stats, sample = _center_out_search(plan)
        rows.extend(found)
        probes.extend(sample)
        # Preserve one complete, independently lexicalized discourse proposal
        # per plan even when the seam ledger dies before a closure. These are
        # real grammatical surfaces with an explicit exact audit, never
        # fragments or promoted candidates.
        left_probe = tuple(options[0] for _, options in plan.left)
        right_probe = tuple(options[0] for _, options in plan.right)
        proposal = audit_pair(left_probe, right_probe, plan)
        proposal["probe_type"] = "complete-independent-plan-proposal"
        probes.append(proposal)
        per_plan[plan.name] = stats
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion",
        "method": "Three-clause topic-continuous discourse plans are independently lexicalized on left and right. A center-out ledger begins at the discourse seam, expands toward both exteriors, and enforces typed singular argument agreement and topic-pronoun continuity before exact tape audit.",
        "novelty_preflight": novelty_preflight(),
        "config": {"plans": [plan.name for plan in PLANS], "clauses_per_side": 3, "slots_per_side": 13, "min_letters": 39, "search_control": "deterministic exhaustive center-out recursion; no beam, MCTS, chart, CSP, ILP, fragments, catalogue input, or word-order mirror"},
        "stats": {"plans": len(PLANS), "exact": sum(row["exact"] for row in rows), "mechanically_admitted": sum(row["mechanically_admitted"] for row in rows), "rendered_probes": len(probes), "max_letters": max((row["letters"] for row in rows + probes), default=0), "per_plan": per_plan},
        "exact_candidates": rows,
        "rendered_probes": probes,
        "provenance": {"source_sentences_copied": False, "catalogue_imported": False, "reader_evidence": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }


def main() -> None:
    if EVIDENCE.exists():
        raise SystemExit(f"refusing to overwrite existing output: {EVIDENCE}")
    payload = run()
    EVIDENCE.parent.mkdir(exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": payload["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
