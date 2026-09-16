"""Held-out determiner-aware adjunct repair for reverse-transition SVO beams.

The base route emitted ``PREP NOUN`` adjuncts and its complete probes all
failed first at the outer character pair ``a``/``b``.  This repair changes the
adjunct to the complete ``PREP DET NOUN`` constituent and adds only held-out
terminal nouns whose final character addresses that observed mismatch.  The
same independent SVO beam and exact reversed-tape join are retained.
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

EXPERIMENT_ID = "reverse-transition-svo-beam-adjunct-repair-20260916"
SIGNATURE = (
    "reverse-compatible-lexical-transitions|broad-bigram-mining|"
    "bilateral-svo-adjunct-grammar|independent-role-banks|"
    "frequency-scored-beam-dp|exact-tape-join|"
    "determiner-licensed-adjunct-repair|mismatch-keyed-terminal-expansion"
)
BASE_EVIDENCE = ROOT / "runs" / "reverse-transition-svo-beam-20260916.json"
EVIDENCE = ROOT / "runs" / "reverse-transition-svo-beam-adjunct-repair-20260916.json"
LEXICON = load_lexicon(str(ROOT / "data" / "lexicon.txt"))

DETS = ("a", "an", "the", "our", "my", "one", "no")
PREPS = ("in", "on", "at", "by", "for", "near", "over", "under", "with")
SUBJECTS_L = ("baker", "farmer", "gardener", "keeper", "maker", "pilot", "poet", "teacher", "worker", "writer", "artist", "doctor")
VERBS_L = ("asks", "calls", "checks", "covers", "finds", "guards", "holds", "keeps", "marks", "reads", "saves", "sends", "shows", "stores", "takes", "tells", "uses", "writes")
OBJECTS_L = ("answer", "book", "garden", "harbor", "image", "letter", "map", "message", "note", "paper", "record", "seed", "story", "token", "word")
SUBJECTS_R = ("caller", "child", "clerk", "driver", "editor", "friend", "guide", "nurse", "reader", "sailor", "artist", "doctor")
VERBS_R = ("answers", "carries", "comes", "enters", "helps", "leads", "listens", "offers", "opens", "replies", "returns", "sees", "speaks", "visits", "waits", "warns")
OBJECTS_R = ("alarm", "bridge", "chart", "file", "flower", "house", "message", "parcel", "plan", "reply", "room", "signal", "station", "tower", "warning")
ADJS_L = ("bright", "calm", "clear", "fresh", "kind", "new", "quiet", "small", "old", "safe", "wise")
ADJS_R = ("ancient", "brief", "careful", "clean", "dark", "early", "fine", "gentle", "large", "open", "ready")
BASE_TERMINALS_R = ("camp", "garden", "hall", "harbor", "lab", "port", "school", "street", "station", "village", "yard")

# These are held out from the base terminal bank.  They are admitted only
# when a measured first mismatch has left character ``a`` and right character
# ``b``; each candidate ends in ``a`` and is a place noun after ``the``.
HELD_OUT_TERMINALS_BY_EDGE = {
    "ab": ("area", "cinema", "opera", "plaza", "villa"),
}


def norm(value: str) -> str:
    return normalize_letters(value)


def mine_transitions(limit: int = 6000) -> tuple[dict[tuple[str, str], int], int]:
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
        key = (norm(words[0])[0], norm(words[1])[-1])
        transitions[key] = transitions.get(key, 0) + count
    return transitions, considered


def transition_score(previous: str | None, current: str, transitions: dict[tuple[str, str], int]) -> float:
    if previous is None:
        return 0.0
    return float(transitions.get((norm(previous)[-1], norm(current)[0]), 0))


def article_agrees(determiner: str, following: str) -> bool:
    initial = norm(following)[0]
    if determiner == "a":
        return initial not in "aeiou"
    if determiner == "an":
        return initial in "aeiou"
    return True


def grammatical_prefix(words: tuple[str, ...]) -> bool:
    """Keep every complete slot prefix locally grammatical and agreement-safe."""
    if len(words) >= 2 and not article_agrees(words[0], words[1]):
        return False
    if len(words) >= 5 and not article_agrees(words[3], words[4]):
        return False
    if len(words) >= 9 and not article_agrees(words[7], words[8]):
        return False
    # All role banks in this repair are singular; retain third-person singular
    # lexicalizations so SVO agreement is explicit, not assumed.
    if len(words) >= 3 and not words[2].endswith("s"):
        return False
    return True


def beam_clause(slots: tuple[tuple[str, ...], ...], transitions: dict[tuple[str, str], int], width: int = 256, protected_final: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    states: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
    for slot_index, options in enumerate(slots):
        expanded: list[tuple[float, tuple[str, ...]]] = []
        for score, words in states:
            previous = words[-1] if words else None
            for word in options:
                candidate = words + (word,)
                if grammatical_prefix(candidate):
                    expanded.append((score + transition_score(previous, word, transitions), candidate))
        ranked = sorted(expanded, key=lambda item: (-item[0], item[1]))
        if slot_index == len(slots) - 1 and protected_final:
            protected_by_word: dict[str, tuple[float, tuple[str, ...]]] = {}
            for item in ranked:
                if item[1][-1] in protected_final:
                    protected_by_word.setdefault(item[1][-1], item)
            protected = [protected_by_word[word] for word in protected_final if word in protected_by_word]
            ordinary = [item for item in ranked if item[1][-1] not in protected_by_word]
            states = ordinary[: max(0, width - len(protected))] + protected
        else:
            states = ranked[:width]
    return [words for _, words in states]


def first_mismatch_keys() -> tuple[str, ...]:
    payload = json.loads(BASE_EVIDENCE.read_text())
    keys: Counter[str] = Counter()
    for row in payload["rendered_probes"]:
        tape = row["tape"]
        for index in range(len(tape) // 2):
            if tape[index] != tape[-1 - index]:
                keys[tape[index] + tape[-1 - index]] += 1
                break
    return tuple(sorted(keys))


def held_out_terminals(keys: tuple[str, ...]) -> tuple[str, ...]:
    words: list[str] = []
    for key in keys:
        words.extend(HELD_OUT_TERMINALS_BY_EDGE.get(key, ()))
    # Keep the repair lexical and fail closed: every held-out unit must be a
    # dictionary word, and only words ending in the observed left edge remain.
    # A base mismatch ``a``/``b`` means the left opening ``a`` must equal the
    # right terminal's closing character; the held-out terminal therefore ends
    # in the *left* member of each observed pair.
    allowed_endings = {key[0] for key in keys}
    return tuple(sorted({word for word in words if word in LEXICON and word[-1] in allowed_endings}))


def render(words: tuple[str, ...]) -> str:
    # DET SUBJ VERB DET ADJ OBJ PREP DET NOUN: complete SVO + adjunct.
    return " ".join(words[:6]) + ". " + " ".join(words[6:]) + "."


def audit(text: str) -> dict:
    tape = norm(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=220)
    return {
        "rendered": text,
        "letters": len(tape),
        "tape": tape,
        "exact": bool(tape) and tape == tape[::-1],
        "ledger_replay": bool(tape) and all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "checks": checks,
        "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
    }


def novelty_preflight() -> dict:
    entries = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())["entries"]
    common = {"a", "an", "and", "after", "audit", "authoring", "before", "character", "complete", "construction", "constraints", "cross", "derived", "derivation", "english", "equation", "exact", "final", "full", "generation", "global", "grammar", "held", "heldout", "in", "independent", "join", "joint", "left", "lexical", "lexicalized", "of", "order", "out", "over", "paired", "parse", "parser", "repair", "residual", "reverse", "right", "sentence", "state", "surface", "tape", "the", "through", "to", "typed", "unit", "word", "words", "with"}
    atoms = lambda value: set(re.findall(r"[a-z0-9]+", value.lower())) - common
    current = atoms(SIGNATURE)
    nearest = []
    for row in entries:
        other = atoms(row["signature"])
        nearest.append({"id": row["id"], "jaccard": round(len(current & other) / len(current | other), 6), "shared_atoms": sorted(current & other)})
    nearest.sort(key=lambda row: (-row["jaccard"], row["id"]))
    return {"registry_entries": len(entries), "exact_signature_collision": any(row["signature"] == SIGNATURE for row in entries), "nearest_prior": nearest[:5]}


def run() -> dict:
    keys = first_mismatch_keys()
    held_out = held_out_terminals(keys)
    transitions, considered = mine_transitions()
    left_slots = (DETS, SUBJECTS_L, VERBS_L, DETS, ADJS_L, OBJECTS_L, PREPS, DETS, ("cave", "field", "home", "market", "office", "river", "road", "room", "shore", "tower", "yard"))
    right_slots = (DETS, SUBJECTS_R, VERBS_R, DETS, ADJS_R, OBJECTS_R, PREPS, DETS, tuple(BASE_TERMINALS_R) + held_out)
    left = beam_clause(left_slots, transitions)
    right = beam_clause(right_slots, transitions, protected_final=held_out)
    right_index: dict[str, list[tuple[str, ...]]] = {}
    for words in right:
        right_index.setdefault(norm(" ".join(words)), []).append(words)
    exact_rows: list[dict] = []
    for left_words in left:
        target = norm(" ".join(left_words))[::-1]
        for right_words in right_index.get(target, ()):
            row = audit(render(left_words) + " " + render(right_words))
            row.update({"left_words": list(left_words), "right_words": list(right_words), "probe_type": "complete-independent-svo-determiner-adjunct"})
            exact_rows.append(row)
    probes: list[dict] = []
    for left_words in left:
        if len(probes) >= 3:
            break
        left_content = {word for word in left_words if word not in DETS and word not in PREPS}
        # Put the newly admitted edge-keyed terminal alternatives first in the
        # rendered audit set so the repair evidence visibly exercises them.
        right_probe_order = sorted(right, key=lambda words: (words[-1] not in held_out, words))
        for right_words in right_probe_order:
            right_content = {word for word in right_words if word not in DETS and word not in PREPS}
            if left_content & right_content:
                continue
            row = audit(render(left_words) + " " + render(right_words))
            if not all(row["checks"].get(key, False) for key in ("distinct_words", "no_repeated_nontrivial_unit")):
                continue
            row.update({"left_words": list(left_words), "right_words": list(right_words), "probe_type": "complete-independent-svo-determiner-adjunct"})
            probes.append(row)
            break
    return {
        "experiment_id": EXPERIMENT_ID,
        "repair_of": "reverse-transition-svo-beam-20260916",
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion",
        "method": "Held-out terminal lexical alternatives are selected only from first reverse-edge mismatches in the frozen base probes; both sides then receive a determiner-aware SVO+adjunct beam and exact reversed-tape join.",
        "novelty_preflight": novelty_preflight(),
        "config": {"transition_rows_considered": considered, "transition_keys": len(transitions), "beam_width": 256, "min_letters": 39, "grammar": "DET SUBJ VERB DET ADJ OBJ PREP DET NOUN", "mismatch_keys": list(keys), "held_out_terminals": list(held_out), "catalogue_imported": False, "search_control": "deterministic frequency-scored beam/DP; no fragments or word-order mirror"},
        "stats": {"left_bank": len(left), "right_bank": len(right), "exact": len(exact_rows), "mechanically_admitted": sum(row["mechanically_admitted"] for row in exact_rows), "rendered_probes": len(probes), "max_probe_letters": max((row["letters"] for row in probes), default=0)},
        "exact_candidates": exact_rows[:120],
        "rendered_probes": probes,
        "provenance": {"base_evidence": str(BASE_EVIDENCE.relative_to(ROOT)), "source_sentences_copied": False, "catalogue_imported": False, "reader_evidence": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
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
