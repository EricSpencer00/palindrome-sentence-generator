"""Exhaustive exact search over complete sentences with relative clauses.

This is a topology repair for the earlier flat-slot searches.  A candidate has
one matrix dependency and one restrictive relative-clause dependency; the
palindrome centre may fall inside any lexical item.  Search is exhaustive for
the frozen domains and uses literal outside-in character cancellation.  A
mechanical survivor is still only a reader-study candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from experiments.semantic_slot_solver import solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

NAMES = """alice alicia amanda amber andrea andrew angela anthony arthur ashley
barbara benjamin brenda brian bridget bruce carla caroline catherine charles
christine claire daniel diana edward elena elizabeth emily eric erika frank
gabriel george grace greg hannah harry helen henry holly irene james janet
jason jennifer jeremy jessica john jonathan joseph julia julie justin karen
katherine kevin laura lauren leon linda lisa louise lucas lucy mark maria
marie mary matthew megan michael michelle natalie nicholas nicole nina noel
nora oliver olivia pamela patricia paul peter rachel rebecca richard robert
rose ruby ruth samantha sarah scott sean sophia stephen susan sylvia thomas
timothy valerie victor victoria vincent walter wendy william""".split()

OBJECTS = """album anchor apron basket bicycle blanket book bottle bracelet
camera candle canvas carpet chair clock coat compass computer diary drawing
envelope flag folder frame game garden guitar hammer jacket key lamp letter
map medal mirror model necklace notebook painting paper parcel photograph
picture pillow postcard radio recipe report ring rope rug scarf schedule
sketch song suitcase table ticket toolbox train vase wallet watch window""".split()

# Verb pairs are selected together, not independently: every pair denotes a
# plausible main event and a relative-clause event concerning the same object.
VERB_PAIRS = (
    ("kept", "sent"), ("found", "lost"), ("opened", "sealed"),
    ("read", "wrote"), ("returned", "borrowed"), ("displayed", "painted"),
    ("carried", "packed"), ("repaired", "damaged"), ("photographed", "built"),
    ("examined", "found"), ("stored", "delivered"), ("moved", "made"),
)


def letters(text: str) -> str:
    return normalize_letters(text)


def words(text: str) -> tuple[str, ...]:
    return tokenize(text)


def audit(text: str, catalogue: set[str]) -> dict[str, bool]:
    shared = mechanical_admission_checks(
        text, local_catalogue=catalogue, min_letters=30, max_letters=80
    )
    return shared | {"at_least_30_letters": shared["length_band"]}


def run() -> dict:
    catalogue_path = ROOT / "data" / "known_palindromes.json"
    catalogue = set(json.loads(catalogue_path.read_text()))
    records: list[dict] = []
    runs: list[dict] = []
    for matrix_verb, relative_verb in VERB_PAIRS:
        # NAME V the OBJECT that NAME V.  The relative clause is a genuine
        # attachment to OBJECT, and its subject is independently lexicalized.
        domains = [NAMES, [matrix_verb], ["the"], OBJECTS, ["that"], NAMES,
                   [relative_verb]]
        found, stats = solve(domains, min_letters=30, max_letters=80)
        runs.append({"matrix_verb": matrix_verb, "relative_verb": relative_verb,
                     **stats})
        for raw in found:
            text = raw[0].upper() + raw[1:] + "."
            checks = audit(text, catalogue)
            records.append({
                "text": text,
                "letters": len(letters(text)),
                "dependency_witness": {
                    "matrix_root": matrix_verb,
                    "matrix_subject": words(text)[0],
                    "matrix_object": words(text)[3],
                    "relative_marker": "that",
                    "relative_subject": words(text)[5],
                    "relative_root": relative_verb,
                },
                "checks": checks,
                "reader_status": "unreviewed; mechanical validity is not readability",
            })
    admitted = [row for row in records if all(row["checks"].values())]
    provenance = {"names": NAMES, "objects": OBJECTS,
                  "verb_pairs": VERB_PAIRS,
                  "surface": "NAME MATRIX_VERB the OBJECT that NAME RELATIVE_VERB"}
    return {
        "status": "complete_relative_clause_character_intersection",
        "operator": "lexicalized matrix-plus-relative dependency tree with character-crossing centre",
        "provenance_sha256": hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
        "runs": runs,
        "records": records,
        "mechanically_admitted": admitted,
        "scope": "Exhaustive only for the frozen original template domains; no reader claim is automatic.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "runs": len(result["runs"]),
                      "records": len(result["records"]),
                      "mechanically_admitted": len(result["mechanically_admitted"])}, indent=2))
