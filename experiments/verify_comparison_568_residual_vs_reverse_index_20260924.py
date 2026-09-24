"""Independent streaming audit of the 2026-09-24 compressed comparison run."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "comparison-568-online-residual-vs-offline-reverse-index-20260924.json.gz"
OUT = ROOT / "runs" / "comparison-568-online-residual-vs-offline-reverse-index-20260924.audit.json"
PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
NOVELTY = ROOT / "runs" / "incumbent-672-global-novelty-snapshot-20260922.json"
EXPECTED_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
ENTITIES = ("Nora", "Leon", "Aron", "Noel", "Mara", "Aram", "Nadia", "Aidan", "Liam", "Ira")
PREDICATES = ("stops", "spots", "sees")


def letters(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text).casefold()


def outside_in(text: str) -> tuple[bool, int | None, str, str]:
    tape = [c.casefold() for c in text if c.isascii() and c.isalpha()]
    lo, hi = 0, len(tape) - 1
    mismatch = None
    while lo < hi:
        if tape[lo] != tape[hi]:
            mismatch = lo
            break
        lo += 1
        hi -= 1
    norm = letters(text)
    return mismatch is None and bool(tape), mismatch, norm, hashlib.sha256(norm.encode()).hexdigest()


def sentence(frame: str) -> str:
    subj, pred, obj = frame.split("|")
    entity = {name.casefold(): name for name in ENTITIES}
    if subj not in entity or obj not in entity or pred not in PREDICATES:
        raise AssertionError(f"unknown grammar frame {frame}")
    return f"{entity[subj]} {pred} {entity[obj]}."


def expected_key(row: dict[str, object]) -> str:
    prov = row["provenance"]
    return "||".join(prov["left_chain"] + ["--"] + prov["right_reverse_pairing_order"])


def audit_candidate(row: dict[str, object], base: str, prior: dict[str, int],
                    base_left: int, base_right: int) -> tuple[str, dict[str, object]]:
    prov = row["provenance"]
    if prov["parent_sha256"] != EXPECTED_SHA or prov["parent_id"] != "outer-causal-scene-568-working-incumbent":
        raise AssertionError("candidate parent provenance mismatch")
    if prov["method_arms"] != ["online_character_residual", "offline_reverse_pair_index"]:
        raise AssertionError("candidate arm provenance mismatch")
    left_frames = list(prov["left_chain"])
    right_reverse = list(prov["right_reverse_pairing_order"])
    if len(left_frames) != 4 or len(right_reverse) != 4:
        raise AssertionError("depth mismatch")
    left_sentences = [sentence(x) for x in left_frames]
    right_sentences = [sentence(x) for x in reversed(right_reverse)]
    if any(left_frames[i].split("|")[2] != left_frames[i + 1].split("|")[0] for i in range(3)):
        raise AssertionError("left chain disconnected")
    if any(right_sentences[i].split()[2].rstrip(".").lower() != right_sentences[i + 1].split()[0].lower()
           for i in range(3)):
        raise AssertionError("right chain disconnected")
    if len(set(left_frames + right_reverse)) != 8:
        raise AssertionError("distinct event gate mismatch")
    all_frames = left_frames + list(reversed(right_reverse))
    relations = [f.replace("|", " ") for f in all_frames]
    if any(prior.get(r, 0) for r in relations):
        raise AssertionError("historical novelty gate mismatch")
    if len({f.split("|")[1] for f in left_frames}) < 2:
        raise AssertionError("predicate diversity gate mismatch")
    left_tape, right_tape = letters(" ".join(left_sentences)), letters(" ".join(right_sentences))
    if left_tape != right_tape[::-1]:
        raise AssertionError("independent inserted-block equation failed")
    expected = base[:base_left] + " " + " ".join(left_sentences) + base[base_left:base_right] + " " + " ".join(right_sentences) + base[base_right:]
    text = str(row["rendered"])
    if text != expected:
        raise AssertionError("candidate text does not replay from parent and frames")
    exact, mismatch, tape, digest = outside_in(text)
    if not exact or tape != tape[::-1] or digest != hashlib.sha256(tape[::-1].encode()).hexdigest():
        raise AssertionError(f"whole-text palindrome check failed at {mismatch}")
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", text.lower())
    tris = [tuple(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2))]
    tri_counts = Counter(tris)
    excess = sum(n - 1 for n in tri_counts.values() if n > 1)
    computed = {
        "letters": len(tape), "growth_over_parent": len(tape) - 568,
        "word_count": len(tokens), "unique_word_count": len(set(tokens)),
        "unique_inserted_relations": len(set(relations)), "inserted_relation_count": len(relations),
        "left_predicate_count": len({f.split("|")[1] for f in left_frames}),
        "repeated_trigram_excess": excess, "trigram_occurrences": len(tris),
        "repeated_trigram_rate": excess / len(tris) if tris else 0.0,
        "duplicate_inserted_clause_count": 8 - len(set(left_sentences + right_sentences)),
        "proper_palindromic_tokens_present": any(t == t[::-1] for t in tokens),
    }
    if computed != row["structural_metrics"]:
        raise AssertionError("stored structural metric differs from independent recomputation")
    saved_audit = row["audit"]
    if saved_audit["letters"] != len(tape) or saved_audit["sha256_forward"] != digest:
        raise AssertionError("stored generator audit differs")
    return expected_key(row), computed


def object_rows(stream):
    """Decode just accepted_candidates incrementally from the top-level JSON."""
    decoder = json.JSONDecoder()
    text = stream.read()
    marker = '"accepted_candidates":['
    pos = text.index(marker) + len(marker)
    while True:
        while text[pos].isspace() or text[pos] == ",":
            pos += 1
        if text[pos] == "]":
            break
        row, pos = decoder.raw_decode(text, pos)
        yield row


def main() -> None:
    parent = json.loads(PARENT.read_text())
    base = next(r["rendered"] for r in parent["rows"] if r["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(base)
    if len(parent_tape) != 568 or hashlib.sha256(parent_tape.encode()).hexdigest() != EXPECTED_SHA:
        raise AssertionError("input parent digest mismatch")
    novelty = json.loads(NOVELTY.read_text())
    prior = novelty["relation_counts"]
    def raw_after(n):
        seen = 0
        for i, ch in enumerate(base):
            if ch.isascii() and ch.isalpha():
                seen += 1
                if seen == n:
                    return i + 1
        raise ValueError(n)
    base_left, base_right = raw_after(48) + 1, raw_after(520) + 1
    if (base_left, base_right) != (62, 722):
        raise AssertionError("seam source positions changed")

    with gzip.open(RUN, "rt", encoding="utf-8") as stream:
        # For cross-arm summary checks, independently decode the compact metadata
        # portion before streaming the candidate array.
        document = json.load(stream)
    if not document["candidate_sets_match"]:
        raise AssertionError("arms reported a candidate mismatch")
    arm_stats = [document["arms"][arm]["stats"] for arm in ("online", "offline")]
    if arm_stats[0]["candidate_key_sha256"] != arm_stats[1]["candidate_key_sha256"]:
        raise AssertionError("arm digests differ")
    keys_hash = hashlib.sha256()
    lengths, unique_words, trigram_rates, trigram_excesses = [], [], [], []
    seen = set()
    count = 0
    for row in document["accepted_candidates"]:
        key, metrics = audit_candidate(row, base, prior, base_left, base_right)
        if key in seen:
            raise AssertionError("duplicate candidate key")
        seen.add(key)
        if count:
            keys_hash.update(b"\n")
        keys_hash.update(key.encode())
        lengths.append(metrics["letters"])
        unique_words.append(metrics["unique_word_count"])
        trigram_rates.append(metrics["repeated_trigram_rate"])
        trigram_excesses.append(metrics["repeated_trigram_excess"])
        count += 1
    if count != arm_stats[0]["accepted_candidates"] or count != arm_stats[1]["accepted_candidates"]:
        raise AssertionError("candidate count mismatch")
    if keys_hash.hexdigest() != arm_stats[0]["candidate_key_sha256"]:
        raise AssertionError("candidate key digest mismatch")
    if arm_stats[0]["candidate_key_sha256"] != "e5e77c7d3fc4b05983c3c32f94835bb6cdc33a3ca0acb55d1d3a8012598feb87":
        raise AssertionError("unexpected candidate set digest")
    result = {
        "audit_id": "comparison-568-online-residual-vs-offline-reverse-index-20260924-independent-audit",
        "source_artifact": RUN.name, "independent_candidate_rows_checked": count,
        "all_candidates_replayed_from_parent": True,
        "all_candidates_pass_two_independent_exactness_checks": True,
        "all_candidate_provenance_and_novelty_gates_recomputed": True,
        "all_structural_metrics_recomputed": True,
        "candidate_key_sha256": keys_hash.hexdigest(),
        "length_summary": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
        "unique_word_summary": {"min": min(unique_words), "median": statistics.median(unique_words), "max": max(unique_words)},
        "repeated_trigram_rate_summary": {"min": min(trigram_rates), "mean": statistics.fmean(trigram_rates),
                                           "median": statistics.median(trigram_rates), "max": max(trigram_rates)},
        "repeated_trigram_excess_summary": {"min": min(trigram_excesses), "mean": statistics.fmean(trigram_excesses),
                                            "median": statistics.median(trigram_excesses), "max": max(trigram_excesses)},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
