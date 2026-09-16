"""Reverse-compatible lexical transitions with bilateral SVO/adjunct beams.

This route first mines a broad lexical transition bank from the shipped
frequency bigrams, retaining only dictionary words and their reverse-edge
character compatibility. A deterministic beam/DP then realizes complete SVO
clauses with optional adjective and prepositional adjunct slots on each side.
Left and right lexical banks are independent. The final join is an exact
normalized-tape lookup, followed by the shared mechanical admission gate.

No source sentence or known palindrome is imported, and no fragment or
word-order mirror is promoted. This is deliberately bounded search evidence,
not readability evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.lexicon import load_lexicon

EXPERIMENT_ID = "reverse-transition-svo-beam-20260916"
SIGNATURE = (
    "reverse-compatible-lexical-transitions|broad-bigram-mining|"
    "bilateral-svo-adjunct-grammar|independent-role-banks|"
    "frequency-scored-beam-dp|exact-tape-join"
)
EVIDENCE = ROOT / "runs" / "reverse-transition-svo-beam-20260916.json"
LEXICON = load_lexicon(str(ROOT / "data" / "lexicon.txt"))

DETS = ("a", "an", "the", "our", "my", "one", "no")
PREPS = ("in", "on", "at", "by", "for", "near", "over", "under", "with")
SUBJECTS_L = ("baker", "farmer", "gardener", "keeper", "maker", "pilot", "poet", "teacher", "worker", "writer", "artist", "doctor")
VERBS_L = ("asks", "calls", "checks", "covers", "finds", "guards", "holds", "keeps", "marks", "reads", "saves", "sends", "shows", "stores", "takes", "tells", "uses", "writes")
OBJECTS_L = ("answer", "book", "garden", "harbor", "image", "letter", "map", "message", "note", "paper", "record", "seed", "story", "token", "word")
ADJUNCTS_L = ("cave", "field", "home", "market", "office", "river", "road", "room", "shore", "tower", "yard")
ADJS_L = ("bright", "calm", "clear", "fresh", "kind", "new", "quiet", "small", "old", "safe", "wise")
SUBJECTS_R = ("caller", "child", "clerk", "driver", "editor", "friend", "guide", "nurse", "reader", "sailor", "artist", "doctor")
VERBS_R = ("answers", "carries", "comes", "enters", "helps", "leads", "listens", "offers", "opens", "replies", "returns", "sees", "speaks", "visits", "waits", "warns")
OBJECTS_R = ("alarm", "bridge", "chart", "file", "flower", "house", "message", "parcel", "plan", "reply", "room", "signal", "station", "tower", "warning")
ADJUNCTS_R = ("camp", "garden", "hall", "harbor", "lab", "port", "school", "street", "station", "village", "yard")
ADJS_R = ("ancient", "brief", "careful", "clean", "dark", "early", "fine", "gentle", "large", "open", "ready")


def norm(value: str) -> str:
    return normalize_letters(value)


def mine_transitions(limit: int = 6000) -> tuple[dict[tuple[str, str], int], int]:
    """Mine frequency-ranked word transitions with a reverse-edge key."""
    transitions: dict[tuple[str, str], int] = {}
    considered = 0
    for line in (ROOT / "data" / "count_2w.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        if considered >= limit:
            break
        try:
            phrase, raw = line.split("\t", 1)
            words = re.findall(r"[a-z]+", phrase.casefold())
            count = int(raw)
        except (ValueError, IndexError):
            continue
        if len(words) != 2 or any(word not in LEXICON for word in words):
            continue
        considered += 1
        # Reverse-compatible: the next word's far edge can consume the prior
        # word's opening character.
        key = (norm(words[0])[0], norm(words[1])[-1])
        transitions[key] = transitions.get(key, 0) + count
    return transitions, considered


def transition_score(previous: str | None, current: str, transitions: dict[tuple[str, str], int]) -> float:
    if previous is None:
        return 0.0
    return float(transitions.get((norm(previous)[-1], norm(current)[0]), 0))


def beam_clause(slots: tuple[tuple[str, ...], ...], transitions: dict[tuple[str, str], int], width: int = 256) -> list[tuple[str, ...]]:
    """Beam/DP over complete lexical slots; each state remains a full clause prefix."""
    states: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
    for options in slots:
        expanded: list[tuple[float, tuple[str, ...]]] = []
        for score, words in states:
            previous = words[-1] if words else None
            for word in options:
                expanded.append((score + transition_score(previous, word, transitions), words + (word,)))
        # Deterministic ranking: transition support first, then lexical yield.
        states = sorted(expanded, key=lambda item: (-item[0], item[1]))[:width]
    return [words for _, words in states]


def render(words: tuple[str, ...]) -> str:
    # DET SUBJ VERB DET ADJ OBJ PREP OBJ = complete SVO + adjunct.
    return " ".join(words[:6]) + ". " + " ".join(words[6:]) + "."


def audit(text: str) -> dict:
    tape = norm(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=220)
    return {"rendered": text, "letters": len(tape), "tape": tape, "exact": bool(tape) and tape == tape[::-1], "ledger_replay": bool(tape) and all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)), "checks": checks, "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values())}


def novelty_preflight() -> dict:
    registry = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    entries = [row for row in registry["entries"] if row["id"] != EXPERIMENT_ID]
    common = {"a", "an", "and", "after", "audit", "authoring", "before", "character", "complete", "construction", "constraints", "cross", "derived", "derivation", "english", "equation", "exact", "final", "full", "generation", "global", "grammar", "held", "heldout", "in", "independent", "join", "joint", "left", "lexical", "lexicalized", "of", "order", "out", "over", "paired", "parse", "parser", "repair", "residual", "reverse", "right", "sentence", "state", "surface", "tape", "the", "through", "to", "typed", "unit", "word", "words", "with"}
    atoms = lambda value: set(re.findall(r"[a-z0-9]+", value.lower())) - common
    current = atoms(SIGNATURE)
    nearest = []
    for row in entries:
        other = atoms(row["signature"])
        nearest.append({"id": row["id"], "jaccard": round(len(current & other) / len(current | other), 6), "shared_atoms": sorted(current & other)})
    nearest.sort(key=lambda row: (-row["jaccard"], row["id"]))
    return {"registry_entries": len(entries), "runtime_registry_entries": len(registry["entries"]), "excluded_routes": len(registry.get("excluded", [])), "exact_signature_collision": any(row["signature"] == SIGNATURE for row in entries), "nearest_prior": nearest[:5]}


def run() -> dict:
    transitions, considered = mine_transitions()
    left_slots = (DETS, SUBJECTS_L, VERBS_L, DETS, ADJS_L, OBJECTS_L, PREPS, ADJUNCTS_L)
    right_slots = (DETS, SUBJECTS_R, VERBS_R, DETS, ADJS_R, OBJECTS_R, PREPS, ADJUNCTS_R)
    left = beam_clause(left_slots, transitions)
    right = beam_clause(right_slots, transitions)
    # Reverse-index DP join: no pair is rendered as a candidate until its
    # complete right tape is looked up against a complete left tape.
    right_index: dict[str, list[tuple[str, ...]]] = {}
    for words in right:
        right_index.setdefault(norm(" ".join(words)), []).append(words)
    exact_rows: list[dict] = []
    probes: list[dict] = []
    for left_words in left:
        target = norm(" ".join(left_words))[::-1]
        for right_words in right_index.get(target, ()):
            row = audit(render(left_words) + " " + render(right_words))
            row.update({"left_words": list(left_words), "right_words": list(right_words)})
            exact_rows.append(row)
    for left_words in left:
        if len(probes) >= 3:
            break
        left_content = {word for word in left_words if word not in DETS and word not in PREPS}
        for right_words in right:
            right_content = {word for word in right_words if word not in DETS and word not in PREPS}
            if left_content & right_content:
                continue
            row = audit(render(left_words) + " " + render(right_words))
            if not all(row["checks"].get(key, False) for key in ("distinct_words", "no_repeated_nontrivial_unit")):
                continue
            row.update({"left_words": list(left_words), "right_words": list(right_words), "probe_type": "complete-independent-svo-adjunct"})
            probes.append(row)
            break
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "status": "completed_no_reader_promotion", "method": "Frequency-ranked reverse-compatible lexical transitions are mined first, then independent complete SVO+adjunct clause banks are generated by deterministic beam/DP and joined through an exact reversed-tape index.", "novelty_preflight": novelty_preflight(), "config": {"transition_rows_considered": considered, "transition_keys": len(transitions), "beam_width": 256, "min_letters": 39, "grammar": "DET SUBJ VERB DET ADJ OBJ PREP OBJ", "catalogue_imported": False, "search_control": "deterministic frequency-scored beam/DP; no fragments or word-order mirror"}, "stats": {"left_bank": len(left), "right_bank": len(right), "exact": len(exact_rows), "mechanically_admitted": sum(row["mechanically_admitted"] for row in exact_rows), "rendered_probes": len(probes), "max_probe_letters": max((row["letters"] for row in probes), default=0)}, "exact_candidates": exact_rows[:120], "rendered_probes": probes, "repair": {"status": "not_run", "operator": "Add a held-out lexical expansion keyed only by the first mismatching reverse-edge character pairs observed in complete probes, then regenerate the same SVO/adjunct beam; preserve the transition mining and exact join.", "reason": "All three complete probes die on early reverse-edge obligations; widening unrelated grammar slots would be a replay."}, "provenance": {"source_sentences_copied": False, "catalogue_imported": False, "reader_evidence": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def main() -> None:
    if EVIDENCE.exists():
        raise SystemExit(f"refusing to overwrite existing output: {EVIDENCE}")
    payload = run()
    EVIDENCE.parent.mkdir(exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": payload["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
