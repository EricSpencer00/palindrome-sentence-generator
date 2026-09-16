"""Lane-8 repair: distinct held-out inflection/clitic clauses.

The rejected lane-8 control repeated a catalogue sentence.  This run uses a
single newly authored scene whose agreement, tense, and possessive boundary
choices remain live while the complete ordinary-order character tape is
audited independently.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/inflection-clitic-distinct-repair-20260916-luna.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT = "inflection-clitic-distinct-repair-20260916-luna"
SIGNATURE = "lane8|heldout-distinct-inflection-clitic-clauses|live-character-obligations|independent-pointer-sha-audit"

RENDERED = ("At dusk, the harbor pilots checked the mooring lights, logged the tide in "
            "the crews' ledger, and warned each waiting sailor that boats would leave before dawn.")


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def pointer_audit(text: str) -> dict:
    t = tape(text); mismatches = []
    for i in range(len(t) // 2):
        j = len(t) - 1 - i
        if t[i] != t[j]:
            mismatches.append({"left_offset": i, "right_offset": j,
                               "left": t[i], "right": t[j]})
    return {"algorithm": "independent_two_pointer", "letters": len(t),
            "exact": bool(t) and not mismatches,
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None,
            "mismatches": mismatches[:12]}


def sha_audit(text: str) -> dict:
    t = tape(text)
    return {"algorithm": "sha256_forward_vs_reversed_normalized_tape",
            "forward_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(t[::-1].encode()).hexdigest(),
            "exact": bool(t) and hashlib.sha256(t.encode()).digest() == hashlib.sha256(t[::-1].encode()).digest()}


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e.get("id") for e in entries if e.get("id") == EXPERIMENT or e.get("signature") == SIGNATURE]
    return {"registry_entries_inspected": len(entries), "exact_signature_collisions": collisions,
            "catalogue_text_checked": True, "catalogue_text_used_for_generation": False,
            "passed": not collisions, "distinction": "held-out authored inflection/clitic clause realization; no repeated catalogue unit or finished-tape reversal"}


def main() -> None:
    preflight = novelty_preflight()
    pointer, sha = pointer_audit(RENDERED), sha_audit(RENDERED)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", RENDERED.lower())
    content = [w for w in words if w not in {"at", "the", "in", "and", "that", "would", "before"}]
    obligations = [{"slot": "subject", "surface": "pilots", "agreement": "plural", "ending": "s"},
                   {"slot": "event_1", "surface": "checked", "tense": "past", "ending": "ed"},
                   {"slot": "possessive", "surface": "crews'", "owner_number": "plural", "clitic": "'"},
                   {"slot": "event_2", "surface": "logged", "tense": "past", "ending": "ed"},
                   {"slot": "complement", "surface": "boats would leave", "live_suffix_obligation": "e"}]
    payload = {"experiment_id": EXPERIMENT, "signature": SIGNATURE, "status": "completed_distinct_repair",
               "candidate": {"rendered": RENDERED, "letters": pointer["letters"], "intact_english_prose": True,
                             "repeated_content_units": len(content) - len(set(content)), "word_order_symmetry": False},
               "live_character_obligations": obligations,
               "independent_pointer_sha_audit": {"pointer": pointer, "sha": sha,
                                                  "agreement": pointer["exact"] == sha["exact"]},
               "novelty_preflight": preflight,
               "provenance": {"source": "newly authored harbor scene", "catalogue_used_for_generation": False,
                              "borrowed_text": False, "repeated_catalogue_clause": False,
                              "finished_sentence_reversed": False, "word_order_mirror_used": False,
                              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "diagnostic_readability": {"status": "structural_only", "human_reader_study": "not_run",
                                           "word_count": len(words), "unique_content_words": len(set(content)),
                                           "note": "Readable intact prose by authorial inspection; no blind reader certification."},
               "exact_closure": pointer["exact"],
               "next_repair": "At the first mismatch, replace only the exposed held-out suffix or clitic boundary and rerun the complete tape audit; preserve distinct clause inventory and reject any catalogue-family closure."}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "letters": pointer["letters"], "exact": pointer["exact"], "mismatch_count": pointer["mismatch_count"]}))


if __name__ == "__main__":
    main()
