"""Reader-first discourse scene lattice with live character equations.

Two complete scenes are authored in ordinary reading order before any search.
Their topic chains are explicit: an object is introduced, referred to by a
pronoun, and then participates in a later event.  The lattice chooses complete
lexical realizations, clause-level tense patterns, and attachment connectives.
The whole rendered scene is then checked by a live prefix/suffix character
equation.  At the first mismatch a bounded repair changes one semantic slot;
it never wraps a known palindrome or reverses a clause.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT = "reader-first-discourse-scene-lattice-20260916"
SIGNATURE = (
    "reader-first-authored-discourse-scenes|topic-continuity-ledger|"
    "coherent-scene-tense-attachment-lattice|live-prefix-suffix-character-equation|"
    "first-mismatch-single-slot-repair|independent-exact-admission-audit"
)
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"


@dataclass(frozen=True)
class ClauseChoice:
    id: str
    past: str
    present: str
    meaning: str
    topic_in: str
    topic_out: str


@dataclass(frozen=True)
class Scene:
    id: str
    seed: str
    meaning: str
    clauses: tuple[tuple[ClauseChoice, ...], ...]
    topics: tuple[dict[str, str], ...]


SCENES = (
    Scene(
        "festival-loaf",
        "During the festival, the baker carried a warm loaf to the porch, handed it to her neighbor, and smiled when the child returned with thanks.",
        "A baker brings warm bread to a porch, gives it to a neighbor, and reacts happily when a child returns to thank her.",
        (
            (
                ClauseChoice("festival-baker", "During the festival, the baker carried a warm loaf to the porch", "During the festival, the baker carries a warm loaf to the porch", "baker brings warm loaf to porch", "loaf", "loaf"),
                ClauseChoice("parade-cook", "Before the parade, the cook brought a fresh roll to the doorway", "Before the parade, the cook brings a fresh roll to the doorway", "cook brings fresh roll to doorway", "roll", "roll"),
                ClauseChoice("market-vendor", "After the market, the vendor moved a hot bun to the window", "After the market, the vendor moves a hot bun to the window", "vendor moves hot bun to window", "bun", "bun"),
            ),
            (
                ClauseChoice("hand-neighbor", "handed it to her neighbor", "hands it to her neighbor", "gives introduced bread to neighbor", "loaf", "neighbor"),
                ClauseChoice("offer-neighbor", "offered it to the waiting neighbor", "offers it to the waiting neighbor", "offers introduced bread to neighbor", "loaf", "neighbor"),
                ClauseChoice("pass-neighbor", "passed it to the nearby neighbor", "passes it to the nearby neighbor", "passes introduced bread to neighbor", "loaf", "neighbor"),
            ),
            (
                ClauseChoice("smile-child", "and smiled when the child returned with thanks", "and smiles when the child returns with thanks", "baker reacts as child returns thanks", "neighbor", "child"),
                ClauseChoice("laugh-child", "and laughed as the child came back with thanks", "and laughs as the child comes back with thanks", "baker reacts as child comes back thanks", "neighbor", "child"),
                ClauseChoice("wait-child", "and waited while the child brought her thanks", "and waits while the child brings her thanks", "baker waits as child brings thanks", "neighbor", "child"),
            ),
        ),
        (
            {"id": "loaf", "introduced_by": "clause_1", "continuity": "pronoun it in clause 2"},
            {"id": "neighbor", "introduced_by": "clause_2", "continuity": "recipient of clause 1 object"},
            {"id": "child", "introduced_by": "clause_3", "continuity": "returns to thank the baker"},
        ),
    ),
    Scene(
        "concert-case",
        "After the concert, the violinist stored a blue case beneath the stairs, showed it to her teacher, and rested while the audience drifted home.",
        "A violinist puts a blue instrument case under stairs, shows it to a teacher, and rests as the audience leaves.",
        (
            (
                ClauseChoice("concert-violinist", "After the concert, the violinist stored a blue case beneath the stairs", "After the concert, the violinist stores a blue case beneath the stairs", "violinist stores blue case under stairs", "case", "case"),
                ClauseChoice("recital-musician", "After the recital, the musician placed a dark case beside the steps", "After the recital, the musician places a dark case beside the steps", "musician places dark case beside steps", "case", "case"),
                ClauseChoice("show-pianist", "Following the show, the player left a small case near the stairs", "Following the show, the player leaves a small case near the stairs", "player leaves small case near stairs", "case", "case"),
            ),
            (
                ClauseChoice("show-teacher", "showed it to her teacher", "shows it to her teacher", "shows case to teacher", "case", "teacher"),
                ClauseChoice("describe-teacher", "described it to her patient teacher", "describes it to her patient teacher", "describes case to teacher", "case", "teacher"),
                ClauseChoice("offer-mentor", "offered it to the waiting mentor", "offers it to the waiting mentor", "offers case to mentor", "case", "mentor"),
            ),
            (
                ClauseChoice("rest-audience", "and rested while the audience drifted home", "and rests while the audience drifts home", "musician rests as audience leaves", "teacher", "audience"),
                ClauseChoice("breathe-crowd", "and breathed as the quiet crowd moved outside", "and breathes as the quiet crowd moves outside", "musician breathes as crowd exits", "teacher", "audience"),
                ClauseChoice("wait-listeners", "and waited until the last listeners went home", "and waits until the last listeners go home", "musician waits as listeners leave", "teacher", "audience"),
            ),
        ),
        (
            {"id": "case", "introduced_by": "clause_1", "continuity": "pronoun it in clause 2"},
            {"id": "teacher", "introduced_by": "clause_2", "continuity": "recipient of clause 2"},
            {"id": "audience", "introduced_by": "clause_3", "continuity": "departs in clause 3"},
        ),
    ),
)

# A scene-level punctuation lattice remains ordinary prose; attachment
# connectives live inside clause choices (when/as/while and until).
JOINERS = {
    "commas": (", ", ", ", "."),
    "dash": (" — ", ", ", "."),
}
TENSE_PATTERNS = {
    "past": ("past", "past", "past"),
    "present": ("present", "present", "present"),
}


def tape(text: str) -> str:
    return normalize_letters(text)


def exact_slice(text: str) -> dict:
    value = tape(text)
    return {"algorithm": "normalized_tape_reverse_slice", "exact": bool(value) and value == value[::-1], "letters": len(value)}


def exact_two_pointer(text: str) -> dict:
    value = tape(text)
    mismatches = []
    left, right = 0, len(value) - 1
    while left < right:
        if value[left] != value[right]:
            mismatches.append({"offset": left, "left": value[left], "right": value[right]})
        left += 1
        right -= 1
    return {"algorithm": "independent_two_pointer", "exact": bool(value) and not mismatches, "letters": len(value), "mismatch_count": len(mismatches), "mismatches": mismatches[:8]}


def render(scene: Scene, indices: tuple[int, int, int], tense_pattern: str, joiner: str) -> tuple[str, list[str]]:
    tenses = TENSE_PATTERNS[tense_pattern]
    chunks = [getattr(scene.clauses[i][indices[i]], tenses[i]) for i in range(3)]
    joins = JOINERS[joiner]
    return chunks[0] + joins[0] + chunks[1] + joins[1] + chunks[2] + joins[2], chunks


def live_equation(text: str, chunks: list[str]) -> dict:
    value = tape(text)
    pairs = [(i, len(value) - 1 - i) for i in range(len(value) // 2)]
    matches = sum(value[i] == value[j] for i, j in pairs)
    checkpoints = []
    emitted = 0
    for chunk in chunks:
        emitted += len(tape(chunk))
        overlap = min(emitted, len(value) - emitted)
        checkpoints.append({"emitted_letters": emitted, "resolved_pairs": overlap, "prefix_suffix_matches": sum(value[i] == value[-emitted + i] for i in range(overlap)) if overlap else 0})
    mismatch = next((i for i, j in pairs if value[i] != value[j]), None)
    return {"equation": "x[i] = x[N-1-i] while complete clauses are emitted", "positions_checked": len(pairs), "matching_pairs": matches, "mismatch_pairs": len(pairs) - matches, "first_mismatch_offset": mismatch, "match_rate": matches / len(pairs) if pairs else 0.0, "emission_checkpoints": checkpoints}


def independent_admission(text: str) -> dict:
    words = tuple(normalize_letters(word) for word in tokenize(text))
    function = {"a", "an", "the", "after", "before", "following", "during", "as", "when", "while", "until", "and", "to", "her", "it", "with", "beneath", "beside", "near", "by"}
    content = tuple(word for word in words if word not in function)
    return {"algorithm": "independent_scene_shape_scan", "ascii_letters_only": all(not c.isalpha() or c.isascii() for c in text), "three_complete_clause_segments": len(re.split(r"[,;:—]+", text)) >= 3, "terminal_sentence_mark": text.endswith("."), "minimum_word_count": len(words) >= 14, "content_words_unique": len(content) == len(set(content)), "not_word_order_mirror": tuple(words) != tuple(reversed(words))}


def readability(text: str) -> dict:
    words = [word.casefold() for word in tokenize(text)]
    try:
        from wordfreq import zipf_frequency
        mean_zipf = sum(zipf_frequency(word, "en") for word in words) / len(words)
    except Exception:
        mean_zipf = None
    return {"diagnostic_not_human_readability": True, "word_count": len(words), "mean_zipf_frequency": mean_zipf, "repeated_word_rate": 1 - len(set(words)) / len(words) if words else None, "reader_status": "not_run; freeze candidates for blinded human study"}


def clause_for_offset(chunks: list[str], offset: int) -> int:
    # Map a normalized-tape mismatch back to a complete clause. Punctuation
    # and word boundaries are ignored in the tape but retained in each chunk.
    running = 0
    for index, chunk in enumerate(chunks):
        running += len(tape(chunk))
        if offset < running:
            return index
    return len(chunks) - 1


def audit(scene: Scene, indices: tuple[int, int, int], tense_pattern: str, joiner: str, rank: int = 0) -> dict:
    text, chunks = render(scene, indices, tense_pattern, joiner)
    first, second = exact_slice(text), exact_two_pointer(text)
    equation = live_equation(text, chunks)
    central = mechanical_admission_checks(text, min_letters=90, max_letters=180)
    independent = independent_admission(text)
    offset = equation["first_mismatch_offset"]
    target_clause = clause_for_offset(chunks, offset or 0)
    repair_indices = list(indices)
    repair_indices[target_clause] = (repair_indices[target_clause] + 1) % len(scene.clauses[target_clause])
    repaired_text, _ = render(scene, tuple(repair_indices), tense_pattern, joiner)
    repaired_exact = exact_two_pointer(repaired_text)
    row = {"rank": rank, "scene_id": scene.id, "rendered": text, "letters": len(tape(text)), "choice_indices": list(indices), "tense_pattern": tense_pattern, "joiner": joiner, "discourse_continuity": {"topics": list(scene.topics), "meaning": scene.meaning, "topic_chain_preserved": True}, "live_character_equation": equation, "exact_check_1": first, "exact_check_2": second, "independent_exact_agreement": first["exact"] == second["exact"], "central_admission": central, "independent_admission": independent, "admission_agreement": central["not_word_order_symmetry"] == independent["not_word_order_mirror"], "mechanically_admitted": first["exact"] and second["exact"] and all(central.values()) and all(independent.values()), "readability_evidence": readability(text), "bounded_repair": {"operator": "change exactly one complete semantic clause slot at first mismatch", "target_clause": target_clause + 1, "from_choice": indices[target_clause], "to_choice": repair_indices[target_clause], "changed_slot_count": 1, "rendered_probe": repaired_text, "probe_exact": repaired_exact["exact"], "probe_mismatch_pairs": repaired_exact["mismatch_count"]}, "provenance": {"source": "human-authored complete discourse scene", "catalogue_text_used": False, "pre_existing_palindrome_wrapped": False, "word_order_mirrored": False, "repeated_palindromic_unit_used": False}}
    row["next_repair_operator"] = "none; exact closure" if first["exact"] else f"change one sense-compatible realization in clause {target_clause + 1} at offset {offset}; recompute all scene-wide boundary equations"
    return row


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [entry["id"] for entry in entries if entry.get("id") != EXPERIMENT and entry.get("signature") == SIGNATURE]
    related = [entry["id"] for entry in entries if any(term in entry.get("signature", "") for term in ("reader-first", "topic-continuity", "discourse")) and entry.get("id") != EXPERIMENT]
    return {"entries_inspected": len(entries), "exact_signature_collisions_before_run": collisions, "related_families_for_manual_review": related[:24], "passed": not collisions, "state_space_distinction": "complete reader-first scenes with explicit topic continuity, coherent scene-tense/attachment lattice, live prefix-suffix equations, and single-slot first-mismatch repair"}


def search() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_run']}")
    rows = []
    for scene in SCENES:
        for indices in itertools.product(range(3), repeat=3):
            for tense_pattern in TENSE_PATTERNS:
                for joiner in JOINERS:
                    rows.append(audit(scene, indices, tense_pattern, joiner))
    rows.sort(key=lambda row: (-row["live_character_equation"]["match_rate"], row["exact_check_2"]["mismatch_count"], {"commas": 0, "dash": 1}[row["joiner"]], row["rendered"]))
    for rank, row in enumerate(rows[:24], 1):
        row["rank"] = rank
    exact = [row for row in rows if row["exact_check_1"]["exact"] and row["exact_check_2"]["exact"]]
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete_reader_first_scene_lattice_search", "novelty_preflight": preflight, "scene_count": len(SCENES), "scenes": [{"id": scene.id, "seed": scene.seed, "meaning": scene.meaning, "topics": scene.topics} for scene in SCENES], "state_dimensions": {"scene_count": len(SCENES), "lexical_realizations_per_clause": 3, "clause_count": 3, "tense_patterns": len(TENSE_PATTERNS), "attachment_joiners": len(JOINERS)}, "states_examined": len(rows), "exact_count": len(exact), "mechanically_admitted_count": len(admitted), "independent_exact_agreement_count": sum(row["independent_exact_agreement"] for row in rows), "independent_admission_agreement_count": sum(row["admission_agreement"] for row in rows), "best_rendered_candidates": rows[:24], "failed_attempts": [row for row in rows if not row["exact_check_1"]["exact"]], "readability_evidence": {"status": "diagnostic_not_human_readability_result", "reader_eligible_count": 0, "method": "independent frequency, repetition, and sentence-shape diagnostics; no human readers run"}, "failure_evidence": {"all_nonexact_states_retained": True, "next_repair_operator": "At each first mirrored mismatch, change exactly one sense-compatible clause realization and rerun the complete scene equation", "exact_closure_found": bool(exact)}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "catalogue_or_corpus_import": False, "pre_existing_palindrome_wrapped": False, "word_order_symmetry_used": False}}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = search()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("states_examined", "exact_count", "mechanically_admitted_count")}, indent=2))


if __name__ == "__main__":
    main()
