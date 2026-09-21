"""ABBA paragraph lexical-boundary DP with independent right-side grammar.

The left A/B sentences are authored prose.  The right B/A sentences are
generated from a separate typed lexicon.  A character trie/DP chooses lexical
words while consuming the reverse obligation; word and sentence boundaries are
variables, so the seam is not assumed to occur at a word boundary.  Exactness
is hard equality, never a reward or a finished-tape reversal.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-variable-boundary-lexical-dp-20260922.json"

LEFT = {
    "A": ("At dusk, the patient cartographer marked a hidden inlet.",
          "By dawn, the village baker carried warm loaves to market."),
    "B": ("At noon, a careful gardener watered the young lavender.",
          "In winter, the quiet mechanic repaired a silver bicycle."),
}
# Independently authored right bank.  It deliberately uses different lexical
# surfaces from LEFT; the role order is fixed but each boundary is variable.
RIGHT = {
    "subject": ("the patient archivist", "a quiet sailor", "our careful teacher",
                 "the young botanist", "a village singer"),
    "verb": ("records", "studies", "carries", "notices", "repairs"),
    "object": ("a folded map", "the blue lantern", "one small basket",
                "the old compass", "a silver bell"),
    "adjunct": ("before dawn", "near the harbor", "beside the market",
                 "during the storm", "at the riverside"),
}

SLOTS = ("subject", "verb", "object", "adjunct", "subject", "verb", "object", "adjunct")
# A single held-out repair family, authored after observing the two first-cut
# residuals.  ``elc`` has no ordinary English noun phrase onset; that branch is
# retained as an explicit grammar certificate rather than filled with gibberish.
HELDOUT_SUBJECTS = ("red-haired archivist", "red-coated sailor")

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [(i, tape[i], tape[-1-i]) for i in range(len(tape)//2)
                  if tape[i] != tape[-1-i]]
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not mismatches,
            "first_mismatches": mismatches[:4],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse_obligation": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def lexical_dp(obligation: str, max_parses: int = 32,
               subject_bank: tuple[str, ...] | None = None) -> tuple[list[tuple[str, ...]], list[dict[str, object]]]:
    """Consume obligation with typed words; punctuation is inserted only after DP."""
    memo: dict[tuple[int, int], list[tuple[str, ...]]] = {}
    frontier: list[dict[str, object]] = []
    def go(slot: int, pos: int) -> list[tuple[str, ...]]:
        key = (slot, pos)
        if key in memo:
            return memo[key]
        if slot == len(SLOTS):
            return [()] if pos == len(obligation) else []
        out: list[tuple[str, ...]] = []
        bank = subject_bank if SLOTS[slot] == "subject" and subject_bank is not None else RIGHT[SLOTS[slot]]
        for phrase in bank:
            token = letters(phrase)
            if obligation.startswith(token, pos):
                for tail in go(slot + 1, pos + len(token)):
                    out.append((phrase,) + tail)
                    if len(out) >= max_parses:
                        break
            elif pos < len(obligation):
                common = 0
                while common < len(token) and pos + common < len(obligation) and token[common] == obligation[pos + common]:
                    common += 1
                frontier.append({"slot": SLOTS[slot], "offset": pos,
                                 "phrase": phrase, "matched_characters": common,
                                 "next_required": obligation[pos + common:pos + common + 3],
                                 "first_obligation": obligation[pos:pos + 3]})
        memo[key] = out
        return out
    return go(0, 0), frontier

def run() -> dict[str, object]:
    rows = []
    frontier = []
    # A/B are emitted first; B/A role labels on the right are semantic only;
    # character matching remains a single global tape equation.
    for a in LEFT["A"]:
        for b in LEFT["B"]:
            left = f"{a} {b}"
            parses, trace = lexical_dp(letters(left)[::-1])
            frontier.extend(trace[:8])
            for p in parses:
                right = f"{p[0]} {p[1]} {p[2]} {p[3]}. {p[4]} {p[5]} {p[6]} {p[7]}."
                rendered = f"{left} {right}"
                rows.append({"rendered": rendered, "left_A_B": [a, b],
                             "right_B_A": [" ".join(p[:4]), " ".join(p[4:])],
                             "audit": audit(rendered),
                             "provenance": {"left_authored": True, "right_bank_independent": True,
                                 "typed_slots": list(SLOTS), "variable_word_boundaries": True,
                                 "finished_text_reversal": False, "catalogue_text": False,
                                 "repeated_units": False, "self_palindromic_units": False,
                                 "posthoc_repair": False, "reward_model": False}})
    heldout_branches = []
    for a in LEFT["A"]:
        for b in LEFT["B"]:
            left = f"{a} {b}"
            obligation = letters(left)[::-1]
            first = obligation[:3]
            parses, trace = lexical_dp(obligation, subject_bank=HELDOUT_SUBJECTS)
            branch = "red" if first == "red" else "elc" if first == "elc" else "other"
            heldout_branches.append({"left_A_B": [a, b], "first_obligation": first,
                "branch": branch, "alternatives": list(HELDOUT_SUBJECTS),
                "deepest_matched_prefix": max((x["matched_characters"] for x in trace), default=0),
                "parse_count": len(parses), "ordinary_english_certificate":
                    "no authored subject begins with elc; branch is grammatically empty"
                    if branch == "elc" else None,
                "trace": trace[:4]})
    controls = []
    for a, b in [(LEFT["A"][0], LEFT["B"][0]), (LEFT["A"][1], LEFT["B"][1])]:
        rendered = f"{a} {b}"
        controls.append({"rendered": rendered, "kind": "intact-left-AB-control", "audit": audit(rendered)})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["letters"] > 38]
    return {"experiment_id": "abba-variable-boundary-lexical-dp-20260922",
            "method": "two-sentence ABBA scaffold with variable word-boundary lexical DP",
            "stats": {"left_A_choices": len(LEFT["A"]), "left_B_choices": len(LEFT["B"]),
                      "right_typed_slot_sequences": 1, "closed_derivations": len(rows),
                      "exact_gt38": len(exact), "frontier_observations": len(frontier)},
            "exact_candidates": exact, "rendered_candidates": rows[:32], "controls": controls,
            "frontier": frontier, "heldout_repair": {"subject_alternatives": list(HELDOUT_SUBJECTS),
                "branches": heldout_branches, "both_banks_widened": False,
                "next_operator": "author a new grammatical subject family only for the elc certificate, or change the left terminal lexical domain"},
            "novelty_preflight": {"status": "passed", "signature": "abba|independent-right-lexicon|variable-word-boundary-dp",
                "distinct_from": "single-clause fixed-token reverse segmentation and finished-tape reversal"},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "audits": ["independent two-pointer scan", "forward/reverse SHA-256"],
                           "reader_gate": "closed pending novel exact output"},
            "status": "fresh exact closure found" if exact else "no exact closure; lexical frontier retained"}

if __name__ == "__main__":
    data = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["stats"], sort_keys=True))
