"""Independently audit every hard gate in the rejected possessive ablation."""
from __future__ import annotations

import argparse
from itertools import product
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN_CATALOGUE_TAPES = frozenset({"margeletsnorahseesharonstelegram"})
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
SUBJECTS = ("marge", "nora", "marie", "sarah")
RECIPIENTS = ("hara", "aino", "norah", "ari", "ane")
OWNERS = ("sarah", "sonia", "sharon", "sari", "sena")
OBJECTS = ("telegram",)


def normalize_independently(text: str) -> str:
    if any(character.isalpha() and not character.isascii() for character in text):
        raise ValueError("unsupported non-ASCII alphabetic character")
    return "".join(re.findall("[a-z]", text.casefold()))


def words_independently(text: str) -> tuple[str, ...]:
    return tuple(WORD.findall(text.casefold()))


def catalogue_family_independently(units: tuple[str, ...]) -> bool:
    return any(
        window[1] == "lets" and window[3] == "see"
        and normalize_independently(window[4]).endswith("s")
        and all(normalize_independently(unit) for unit in window)
        for start in range(len(units) - 5)
        for window in (units[start:start + 6],)
    )


def boundary_aligned_word_mirror_independently(units: tuple[str, ...]) -> bool:
    normalized = tuple(normalize_independently(unit) for unit in units)
    return normalized == tuple(unit[::-1] for unit in reversed(normalized))


def render_independently(subject: str, recipient: str, owner: str, object_word: str) -> str:
    return f"{subject.capitalize()} lets {recipient.capitalize()} see {owner.capitalize()}'s {object_word}."


def independently_enumerated_closures() -> list[dict]:
    """Re-enumerate the pinned ablation language without importing its generator."""
    closures = []
    for subject, recipient, owner, object_word in product(SUBJECTS, RECIPIENTS, OWNERS, OBJECTS):
        rendered = render_independently(subject, recipient, owner, object_word)
        tape = normalize_independently(rendered)
        if tape == tape[::-1]:
            closures.append({
                "rendered": rendered,
                "slots": {
                    "subject": subject, "finite_verb": "lets", "recipient": recipient,
                    "infinitive": "see", "possessor": owner, "object": object_word,
                },
            })
    return closures


def hard_gates_independently(text: str, local_catalogue: set[str]) -> dict[str, bool]:
    try:
        tape, units = normalize_independently(text), words_independently(text)
        supported_ascii_letters = True
    except ValueError:
        tape, units, supported_ascii_letters = "", (), False
    normalized_units = tuple(normalize_independently(unit) for unit in units)
    from llm_palindrome.lexicon import is_real_word, load_lexicon
    lexicon = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    normalized_catalogue = {normalize_independently(item) for item in local_catalogue}
    return {
        "supported_ascii_letters": supported_ascii_letters,
        "nonempty": bool(tape and units),
        "exact_letter_palindrome": bool(tape) and tape == "".join(reversed(tape)),
        "length_band": 30 <= len(tape) <= 80,
        "word_form": bool(re.fullmatch(r"[A-Za-z][A-Za-z '\-.,;:!?]*", text)),
        "lexicon_words": bool(units) and all(is_real_word(unit, lexicon) for unit in normalized_units),
        "distinct_words": len(units) == len(set(units)),
        "no_self_palindromic_word": all(unit != "".join(reversed(unit)) for unit in normalized_units),
        "not_word_order_symmetry": not boundary_aligned_word_mirror_independently(units),
        "no_repeated_nontrivial_unit": not repeated_nontrivial_unit_independently(normalized_units),
        "not_forbidden_catalogue_control": tape not in FORBIDDEN_CATALOGUE_TAPES,
        "not_catalogue_family_derivative": not catalogue_family_independently(units),
        "absent_from_local_catalogue": tape not in normalized_catalogue,
        "local_catalogue_absent": tape not in normalized_catalogue,
    }


def repeated_nontrivial_unit_independently(units: tuple[str, ...]) -> bool:
    for width in range(2, len(units) // 2 + 1):
        for left in range(0, len(units) - width):
            for right in range(left + width, len(units) - width + 1):
                if units[left:left + width] == units[right:right + width]:
                    return True
    return False


def source_contract_errors(result: dict) -> list[str]:
    """Reject stale promoted artifacts before writing any verification report."""
    errors = []
    if result.get("status") != "complete_rejected_catalogue_family_ablation":
        errors.append("source status is not the rejected catalogue-family ablation")
    if result.get("promotion") != "forbidden; retain only as a rejected ablation":
        errors.append("source promotion contract is missing or altered")
    if result.get("mechanically_admitted") != []:
        errors.append("source contains mechanically admitted records")
    if not isinstance(result.get("vocabulary"), dict):
        errors.append("source vocabulary is malformed")
    if not isinstance(result.get("exact_closures"), list):
        errors.append("source exact-closure inventory is malformed")
    expected_generator = hashlib.sha256(
        (ROOT / "experiments" / "possessive_name_relexicalizer.py").read_bytes()
    ).hexdigest()
    expected_catalogue = hashlib.sha256(
        (ROOT / "data" / "catalogue_provenance.json").read_bytes()
    ).hexdigest()
    if result.get("generator_source_sha256") != expected_generator:
        errors.append("generator source hash does not match the frozen ablation")
    if result.get("catalogue_provenance_sha256") != expected_catalogue:
        errors.append("catalogue provenance hash does not match the frozen control")
    expected_vocabulary = {
        "subjects": SUBJECTS, "recipients": RECIPIENTS,
        "owners": OWNERS, "objects": OBJECTS,
    }
    if isinstance(result.get("vocabulary"), dict):
        try:
            actual_vocabulary = {
                key: tuple(values) for key, values in result["vocabulary"].items()
            }
        except TypeError:
            errors.append("source vocabulary is malformed")
        else:
            if actual_vocabulary != expected_vocabulary:
                errors.append("source vocabulary differs from the frozen ablation")
            expected_vocabulary_hash = hashlib.sha256(
                json.dumps(expected_vocabulary, sort_keys=True).encode()
            ).hexdigest()
            if result.get("vocabulary_sha256") != expected_vocabulary_hash:
                errors.append("source vocabulary hash does not match the frozen ablation")
    if isinstance(result.get("exact_closures"), list):
        try:
            source_closures = [
                {"rendered": row["rendered"], "slots": row["slots"]}
                for row in result["exact_closures"]
            ]
        except (KeyError, TypeError):
            errors.append("source exact-closure inventory is malformed")
        else:
            if source_closures != independently_enumerated_closures():
                errors.append("source exact closures differ from independent enumeration")
    return errors


def verify(path: Path) -> dict:
    result = json.loads(path.read_text())
    contract_errors = source_contract_errors(result)
    if contract_errors:
        raise ValueError("; ".join(contract_errors))
    local_catalogue_path = ROOT / "data" / "known_palindromes.json"
    local_catalogue = set(json.loads(local_catalogue_path.read_text()))
    expected_vocabulary = {
        "subjects": SUBJECTS, "recipients": RECIPIENTS,
        "owners": OWNERS, "objects": OBJECTS,
    }
    actual_vocabulary = {key: tuple(values) for key, values in result["vocabulary"].items()}
    expected_closures = independently_enumerated_closures()
    source_closures = [
        {"rendered": row["rendered"], "slots": row["slots"]}
        for row in result["exact_closures"]
    ]
    enumeration_matches = source_closures == expected_closures
    rows = []
    for row in result["exact_closures"]:
        tape = normalize_independently(row["rendered"])
        recomputed = hard_gates_independently(row["rendered"], local_catalogue)
        rows.append({
            "rendered": row["rendered"],
            "letters": len(tape),
            "exact_letter_palindrome": tape == "".join(reversed(tape)),
            "recomputed_checks": recomputed,
            "source_checks_match_independent_checks": row["checks"] == recomputed,
            "independently_promotion_eligible": all(recomputed.values()),
        })
    return {
        "source": str(path),
        "normalizer": "stdlib casefold plus ASCII-letter regex",
        "local_catalogue_sha256": hashlib.sha256(local_catalogue_path.read_bytes()).hexdigest(),
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_source_sha256": result["generator_source_sha256"],
        "catalogue_provenance_sha256": result["catalogue_provenance_sha256"],
        "forbidden_catalogue_tapes": sorted(FORBIDDEN_CATALOGUE_TAPES),
        "frozen_vocabulary_matches": actual_vocabulary == expected_vocabulary,
        "independently_enumerated_derivations": len(SUBJECTS) * len(RECIPIENTS) * len(OWNERS) * len(OBJECTS),
        "independently_enumerated_exact_closures": expected_closures,
        "source_exact_closures_match_independent_enumeration": enumeration_matches,
        "records": rows,
        "all_exact_closures_independently_verified": (
            bool(rows) and actual_vocabulary == expected_vocabulary and enumeration_matches
            and all(row["exact_letter_palindrome"] and row["source_checks_match_independent_checks"] for row in rows)
        ),
        "promotion_eligible_records": sum(row["independently_promotion_eligible"] for row in rows),
        "promotion": "forbidden for this rejected catalogue-family ablation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    try:
        result = verify(args.input)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        parser.error(f"refusing unverifiable source: {exc}")
    if not result["all_exact_closures_independently_verified"]:
        parser.error("refusing an audit whose final independent invariants did not pass")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "all_exact_closures_independently_verified": result[
                          "all_exact_closures_independently_verified"],
                      "promotion_eligible_records": result["promotion_eligible_records"]}, indent=2))


if __name__ == "__main__":
    main()
