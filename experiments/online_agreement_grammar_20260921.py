"""Bounded online grammar lane: agreement is encoded before tape search.

This is deliberately a new lexical inventory, not a repair pass.  Each
alternative is a complete, hand-authored clause fragment with subject/verb
agreement and compatible argument number.  The parent engine intersects its
character transitions online; completed Cartesian sentences are never
enumerated.
"""
from pathlib import Path
import hashlib
import json

from online_regular_language_palindrome_20260921 import independent_audit, intersect, self_test

ROOT = Path(__file__).resolve().parents[1]


def main():
    # The two clause frames are intentionally ordinary English.  Agreement is
    # carried by the atomic subject+verb alternatives, while the object slot
    # carries number-compatible complements.  No palindrome seed or catalogue
    # sentence occurs in this inventory.
    slots = [
        [
            "A quiet archivist records ",
            "A patient teacher guides ",
            "The young sailor maps ",
            "An honest witness marks ",
        ],
        ["the old harbor; ", "a narrow inlet; ", "the winter road; ", "a quiet garden; "],
        [
            "a quiet archivist records ",
            "a patient teacher guides ",
            "the young sailor maps ",
            "an honest witness marks ",
        ],
        ["the old harbor.", "a narrow inlet.", "the winter road.", "a quiet garden."],
    ]
    result = intersect(slots)
    controls = [
        "A quiet archivist records the old harbor; a quiet archivist records the old harbor.",
        "A patient teacher guides a narrow inlet; a patient teacher guides a narrow inlet.",
        "The young sailor maps the winter road; the young sailor maps the winter road.",
    ]
    payload = dict(
        experiment_id="online-agreement-grammar-20260921",
        method="Online equal-character product of agreement-carrying clause alternatives",
        grammar=slots,
        result=result,
        controls=[dict(text=t, audit=independent_audit(t), readable_control=True) for t in controls],
        provenance={
            "source": "Fresh hand-authored agreement grammar; no catalogue sentence, seed, or generated palindrome is an input.",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "engine": "online_regular_language_palindrome_20260921.intersect",
        },
        independent_checks={
            "engine_self_test": self_test(),
            "candidate_audit": [r["audit"] for r in result["candidates"]],
            "pointer_and_sha_requirements": "two_pointer_exact, forward_sha256, reverse_sha256, and hash_equal are computed independently for every returned path",
        },
        novelty_preflight={
            "status": "fresh bounded lexical grammar, not counted as a breakthrough",
            "distinction": "Agreement is carried in clause alternatives before online character intersection; no completed-path repair or post-hoc filtering is used.",
            "checked_against": [
                "online-regular-language-palindrome-20260921",
                "shared-tape-finite-automata-role-ledger-20260921",
                "lexicalized-macro-grammar-20260921",
            ],
        },
        reader_evidence=None,
        reader_gate="Closed: no exact candidate was admitted; controls are not palindrome evidence.",
        next_frontier="Add a second agreement-bearing frame with tense and transitive-object compatibility, while retaining live character obligations; do not repair any rendered path.",
    )
    out = ROOT / "runs/online-agreement-grammar-20260921.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"exact": len(result["candidates"]), "stats": result["stats"], "controls": len(controls)}))


if __name__ == "__main__":
    main()
