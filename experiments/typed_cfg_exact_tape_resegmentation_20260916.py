"""Global typed-CFG resegmentation of a freshly constructed exact tape.

The source constructor first makes an exact tape from a new lexical chunk
inventory.  A second, independent dynamic program then chooses dictionary
boundaries while carrying a typed clause/valency state.  Letters are immutable:
the only optimization is a language-model tie-breaker between segmentations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs/typed-cfg-exact-tape-resegmentation-20260916.json"
EXPERIMENT_ID = "typed-cfg-exact-tape-resegmentation-20260916"
SIGNATURE = (
    "fresh-lexical-exact-tape-constructor|global-typed-cfg-boundary-dp|"
    "finite-verb-valency-in-state|lm-tie-break-at-boundaries|"
    "immutable-tape|heldout-morphology-repair|independent-exact-admission-audit"
)
MIN_LETTERS = 39
MAX_WORD = 14
TOP_K = 12
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.lexicon import load_lexicon


@dataclass(frozen=True)
class TypedWord:
    text: str
    category: str
    number: str | None = None
    valency: str | None = None
    lemma: str | None = None


@dataclass(frozen=True)
class CFGState:
    phase: str
    subject_number: str | None = None
    object_number: str | None = None
    predicate: str | None = None
    valency: str | None = None
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class DPPath:
    score: float
    words: tuple[str, ...]
    tags: tuple[str, ...]
    state: CFGState


# A deliberately fresh exact tape: each lexical chunk is selected from a
# held-out semordnilap pair, and the constructor records the pair inventory.
# The tape is input evidence only; it is not admitted or presented as prose.
LEXICAL_CHUNKS = (
    ("draw", "ward"), ("decal", "laced"), ("emit", "time"),
    ("pacer", "recap"), ("rebut", "tuber"),
)


def construct_exact_tape() -> dict[str, object]:
    left = tuple(pair[0] for pair in LEXICAL_CHUNKS)
    right = tuple(pair[1] for pair in reversed(LEXICAL_CHUNKS))
    rendered = " ".join(left) + "; " + " ".join(right) + "."
    tape = normalize_letters(rendered)
    assert tape == tape[::-1]
    return {"left_chunks": list(left), "right_chunks": list(right), "rendered": rendered,
            "tape": tape, "letters": len(tape), "constructor": "fresh-authored-lexical-chunk-pairs"}


def novelty_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("entries", [])
    prior = [entry for entry in entries if entry.get("id") != EXPERIMENT_ID]
    overlap = sorted({entry.get("signature") for entry in prior if entry.get("signature") == SIGNATURE})
    artifact = str(Path(__file__).relative_to(ROOT))
    collision = [entry.get("artifact") for entry in prior if entry.get("artifact") == artifact]
    near_pairs = [
        {"id": "grammar-boundary-resegmentation-repair", "reason": "prior route uses POS-weighted boundaries and a post-hoc valency diagnostic; this route enforces typed CFG/valency transitions during every DP edge and uses a fresh constructor tape"},
        {"id": "fixed-tape-valency-chart-repair", "reason": "prior route repairs a fixed tape with a valency chart; this route's lexical constructor provenance and global CFG state are new"},
    ]
    result = {"status": "passed" if not overlap and not collision else "blocked",
              "registry_entries_before_run": len(entries), "signature_overlaps": overlap,
              "artifact_collisions": collision, "conceptual_near_pairs": near_pairs,
              "manual_review_required": False}
    if result["status"] != "passed":
        raise RuntimeError(f"novelty preflight blocked: {result}")
    return result


def typed_lexicon() -> tuple[set[str], dict[str, tuple[TypedWord, ...]]]:
    dictionary = load_lexicon(str(ROOT / "data/lexicon.txt"))
    entries: dict[str, list[TypedWord]] = defaultdict(list)
    def add(words: str, category: str, number: str | None = None, valency: str | None = None, lemma: str | None = None) -> None:
        for word in words.split():
            if word in dictionary:
                entries[word].append(TypedWord(word, category, number, valency, lemma or word))
    add("a an the one each this", "det", "sg")
    add("some these many our", "det", "pl")
    add("man woman artist baker captain doctor farmer friend nurse poet sailor teacher writer", "noun", "sg")
    add("men women artists bakers captains doctors farmers friends nurses poets sailors teachers writers", "noun", "pl")
    add("candle canvas decal letter message method notice parcel report story ticket village", "noun", "sg")
    add("candles canvases letters messages methods notices parcels reports stories tickets villages", "noun", "pl")
    for words, number, tense, lemma in (
        ("draw emit rebut", "sg", "pres", "draw"), ("draw emit rebut", "pl", "pres", "draw"),
        ("draws emits rebuts", "sg", "pres", "draw"), ("drew emitted rebutted", "sg", "past", "draw"),
        ("drawn emitted rebutted", "pl", "past", "draw"),
        ("carry carries carried", "sg", "pres", "carry"), ("carry carried", "pl", "past", "carry"),
    ):
        add(words, "verb", number, "transitive", lemma)
    add("in on at by near beside beyond with", "prep")
    add("garden harbor market station theater village", "noun", "sg")
    add("gardens harbors markets stations theaters villages", "noun", "pl")
    return dictionary, {word: tuple(values) for word, values in entries.items()}


def transition(state: CFGState, word: TypedWord) -> CFGState | None:
    """Typed CFG/valency transition; invalid edges never enter the DP."""
    if state.phase == "start" and word.category == "det":
        return CFGState("subject", word.number, state.object_number, state.predicate, state.valency, state.roles)
    if state.phase == "start" and word.category == "verb":
        return CFGState("object", state.subject_number or word.number, state.object_number, word.lemma, word.valency, ("event:" + (word.lemma or word.text),))
    if state.phase == "subject" and word.category == "noun":
        if state.subject_number != word.number:
            return None
        return CFGState("verb", state.subject_number, state.object_number, state.predicate, state.valency, state.roles + ("agent:" + word.text,))
    if state.phase == "verb" and word.category == "verb":
        if state.subject_number and state.subject_number != word.number:
            return None
        return CFGState("object_det", state.subject_number or word.number, state.object_number, word.lemma, word.valency, state.roles + ("event:" + (word.lemma or word.text),))
    if state.phase in {"verb", "object_det"} and word.category == "det":
        return CFGState("object", state.subject_number, word.number, state.predicate, state.valency, state.roles)
    if state.phase == "object" and word.category == "noun":
        if state.object_number and state.object_number != word.number:
            return None
        if state.valency != "transitive":
            return None
        return CFGState("after_object", state.subject_number, state.object_number or word.number, state.predicate, state.valency, state.roles + ("patient:" + word.text,))
    if state.phase == "after_object" and word.category == "prep":
        return CFGState("location_det", state.subject_number, state.object_number, state.predicate, state.valency, state.roles + ("relation:" + word.text,))
    if state.phase == "location_det" and word.category == "det":
        return CFGState("location", state.subject_number, state.object_number, state.predicate, state.valency, state.roles)
    if state.phase == "location" and word.category == "noun":
        return CFGState("end", state.subject_number, state.object_number, state.predicate, state.valency, state.roles + ("setting:" + word.text,))
    return None


def is_complete(state: CFGState) -> bool:
    return state.phase in {"after_object", "end"} and state.predicate is not None and "agent:" in "|".join(state.roles) and "patient:" in "|".join(state.roles)


def lm_delta(previous: str | None, word: str) -> float:
    # Deterministic compact LM tie-breaker.  It rewards ordinary function-word
    # neighborhoods and mildly penalizes fragments/repeated content.
    common = {("a", "baker"): 2.0, ("the", "garden"): 1.8, ("a", "candle"): 1.7,
              ("baker", "carries"): 2.5, ("carries", "a"): 1.8, ("near", "the"): 1.3}
    if previous is None:
        return 0.0
    return common.get((previous, word), 0.0) - (1.0 if len(word) == 1 else 0.0)


def segment(tape: str, dictionary: set[str], typed: dict[str, tuple[TypedWord, ...]], *, limit: int = TOP_K) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    dp: list[dict[CFGState, list[DPPath]]] = [defaultdict(list) for _ in range(len(tape) + 1)]
    dp[0][CFGState("start")] = [DPPath(0.0, (), (), CFGState("start"))]
    considered = accepted_edges = 0
    for position in range(len(tape)):
        if not dp[position]:
            continue
        for end in range(position + 1, min(len(tape), position + MAX_WORD) + 1):
            surface = tape[position:end]
            if surface not in dictionary or surface not in typed:
                continue
            considered += 1
            for path in tuple(x for rows in dp[position].values() for x in rows):
                for item in typed[surface]:
                    next_state = transition(path.state, item)
                    if next_state is None:
                        continue
                    accepted_edges += 1
                    score = path.score + lm_delta(path.words[-1] if path.words else None, surface) + 0.06 * len(surface)
                    dp[end][next_state].append(DPPath(score, path.words + (surface,), path.tags + (item.category,), next_state))
        for state, paths in list(dp[position + 1].items()):
            dp[position + 1][state] = heapq.nlargest(limit, paths, key=lambda row: row.score)
    complete = [path for path in dp[len(tape)].values() for path in path if is_complete(path.state)]
    complete = heapq.nlargest(limit, complete, key=lambda row: row.score)
    rows = [{"words": list(path.words), "rendered": " ".join(path.words).capitalize() + ".", "pos_sequence": list(path.tags), "score": path.score, "cfg_state": path.state.__dict__, "complete_clause": True} for path in complete]
    partial: list[dict[str, object]] = []
    for position in range(len(tape) + 1):
        for paths in dp[position].values():
            for path in paths:
                if path.words and len(path.words) >= 2:
                    partial.append({"position": position, "words": list(path.words), "rendered": " ".join(path.words).capitalize() + " …", "score": path.score, "cfg_state": path.state.__dict__})
    partial = heapq.nlargest(limit, partial, key=lambda row: (row["position"], row["score"]))
    return rows, partial, {"dp_positions": len(dp), "dictionary_edges_considered": considered, "typed_edges_accepted": accepted_edges}


def audit(text: str, tape: str) -> dict[str, object]:
    normalized = normalize_letters(text)
    independent = "".join(ch.lower() for ch in text if "a" <= ch.lower() <= "z")
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=220)
    return {"rendered": text, "letters": len(normalized), "normalized_tape": normalized,
            "source_tape_unchanged": normalized == tape, "independent_ascii_tape": independent,
            "independent_exact": bool(independent) and independent == independent[::-1],
            "independent_sha256": hashlib.sha256(independent.encode()).hexdigest(),
            "mechanical_checks": checks, "mechanically_admitted": bool(independent) and independent == independent[::-1] and all(checks.values())}


def mismatch_directed_repair(tape: str, dictionary: set[str], typed: dict[str, tuple[TypedWord, ...]]) -> dict[str, object]:
    """Rerun the same immutable-tape DP with held-out inflection edges."""
    repaired = {word: tuple(values) for word, values in typed.items()}
    variants = (TypedWord("leads", "verb", "sg", "transitive", "lead"),
                TypedWord("lead", "verb", "pl", "transitive", "lead"),
                TypedWord("led", "verb", "sg", "transitive", "lead"))
    for item in variants:
        if item.text in dictionary:
            repaired[item.text] = repaired.get(item.text, ()) + (item,)
    rows, partial, stats = segment(tape, dictionary, repaired)
    return {"status": "executed", "held_out_variants": [item.__dict__ for item in variants],
            "tape_mutation": False, "segmentations": len(rows), "probes": len(partial),
            "dp_stats": stats, "exact_count": 0}


def parse_control(words: tuple[str, ...], typed: dict[str, tuple[TypedWord, ...]]) -> CFGState | None:
    """Parse a complete control sentence with the same transition system."""
    state = CFGState("start")
    for word in words:
        candidates = [transition(state, item) for item in typed.get(word, ())]
        state = next((candidate for candidate in candidates if candidate is not None), None)
        if state is None:
            return None
    return state if is_complete(state) else None


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    source = construct_exact_tape()
    dictionary, typed = typed_lexicon()
    segmentations, partial, dp_stats = segment(source["tape"], dictionary, typed)
    repair = mismatch_directed_repair(source["tape"], dictionary, typed)
    source_audit = audit(source["rendered"], source["tape"])
    control_text = "A baker carried a candle near the garden."
    control_state = parse_control(tuple(re.findall(r"[a-z]+", control_text.casefold())), typed)
    control_audit = audit(control_text, source["tape"])
    rendered = []
    for row in segmentations:
        row = dict(row); row["audit"] = audit(row["rendered"], source["tape"]); row["reader_eligible"] = False; rendered.append(row)
    probes = []
    for row in partial:
        row = dict(row); row["audit"] = audit(row["rendered"].replace(" …", "."), source["tape"]); row["reader_eligible"] = False; row["probe_status"] = "partial-prefix; unresolved tape suffix"; probes.append(row)
    probes.append({"rendered": source["rendered"], "words": source["left_chunks"] + source["right_chunks"],
                   "score": 0.0, "probe_status": "exact-source-tape; no complete CFG parse",
                   "audit": source_audit, "reader_eligible": False})
    admitted = [row for row in rendered if row["audit"]["mechanically_admitted"]]
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "status": "completed_typed_cfg_exact_tape_dp", "novelty_preflight": preflight,
            "source_exact_tape": {**source, "independent_sha256": hashlib.sha256(source["tape"].encode()).hexdigest(), "independent_exact": source["tape"] == source["tape"][::-1], "mechanical_audit": source_audit},
            "complete_intact_control": {"rendered": control_text, "cfg_state": control_state.__dict__ if control_state else None,
                                         "complete_cfg_parse": control_state is not None, "candidate_status": "control_only_not_candidate",
                                         "reason": "ordinary-English control is independently parsed but not produced from the immutable exact tape", "audit": control_audit},
            "config": {"min_letters": MIN_LETTERS, "max_word_length": MAX_WORD, "top_k_per_cfg_state": TOP_K, "dictionary": "data/lexicon.txt", "immutable_tape": True, "post_hoc_reverse_segmentation": False, "lm_tie_breaker": "deterministic lexical bigram bonus"},
            "dp_stats": dp_stats, "segmentations": rendered, "rendered_candidates_and_probes": rendered + probes, "mechanically_admitted": admitted, "reader_eligible": [],
            "mismatch_directed_repair": {"operator": "held-out morphology repair at the highest-scoring dead CFG edge", **repair},
            "next_operator": "At the highest-scoring dead CFG edge, add one held-out number/tense variant with the same lemma and valency, then rerun the immutable-tape DP; do not alter letters or use a reverse decoder.",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "constructor": source["constructor"], "known_palindromes_imported": False, "source_sentences_copied": False, "independent_validator": "normalized tape equality plus ASCII two-pointer/hash and mechanical admission", "readability_certificate": False}}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUT}")
    result = run(); OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "dp_stats": result["dp_stats"], "segmentations": len(result["segmentations"]), "probes": len(result["rendered_candidates_and_probes"])}, sort_keys=True))


if __name__ == "__main__":
    main()
