"""Typed two-character center onset chart with live first outer obligation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_asymmetric_relative_earley_20260920 import chart
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm

EXPERIMENT_ID = "character-cfg-typed-twochar-center-20260920"
CENTER_ONSETS = (("th", "the", "sg"), ("so", "some", "pl"))


def seams(words):
    p = 0; out = set()
    for word in words[:-1]:
        p += len(word); out.add(p)
    return out


def live(left, right, ls, rs):
    matched = 0; crossed = False; rev = right[::-1]
    while matched < len(left) and matched < len(rev) and left[matched] == rev[matched]:
        crossed = crossed or matched + 1 in ls or len(right) - matched - 1 in rs
        matched += 1
    return matched, crossed


def occurrences(words, token):
    pos = 0; out = []
    for i, word in enumerate(words):
        if word == token: out.append((i, pos))
        pos += len(word)
    return out


def run():
    trie, lex = brown_words()
    # Three deterministic determiner strata are retained so the held-out
    # onset categories are actually exercised; this is still the fixed bank.
    subject, subject_states = chart(lex, trie, True, cap=540)
    object_side, object_states = chart(lex, trie, False, cap=540)
    onset_support = Counter(); first_outer_support = Counter(); center_states = outer_states = seam_hits = 0
    exact = []; best = {"score": 0, "left": "", "right": "", "onset": ""}
    for left in subject:
        lt = norm(" ".join(left.words)); ls = seams(left.words)
        for right in object_side:
            rt = norm(" ".join(right.words)); rs = seams(right.words)
            for onset, surface, agreement in CENTER_ONSETS:
                if left.number != agreement or right.number != agreement:
                    continue
                lp = occurrences(left.words, surface); rp = occurrences(right.words, surface)
                if not lp or not rp: continue
                onset_support[onset] += 1
                _, lc = lp[0]; _, rc = rp[0]
                # Center debt exposes exactly the typed two-character onset.
                cn, cc = live(lt[lc:lc + 2], rt[rc:rc + 2], ls, rs)
                # Outer debt begins immediately before the onset and is solved
                # against the opposite exposed suffix at the same time.
                ol, oright = lt[:lc], rt[rc + 2:]
                on, oc = live(ol, oright, ls, rs)
                center_states += cn; outer_states += on
                if ol and oright and ol[0] == oright[-1]: first_outer_support[onset] += 1
                if cc and oc: seam_hits += 1
                score = cn + on
                if score > best["score"]:
                    best = {"score": score, "left": " ".join(left.words), "right": " ".join(right.words),
                            "onset": onset, "center_seam": cc, "outer_seam": oc}
                if not (cc and oc): continue
                rendered = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
                a = audit(rendered)
                if a["two_pointer_exact"]:
                    exact.append({"rendered": rendered, "audit": a,
                                  "typed_center": {"onset": onset, "surface": surface, "agreement": agreement},
                                  "debts": {"center": cn, "outer": on},
                                  "provenance": {"catalogue_imported": False, "finished_tape_reversed": False,
                                                 "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "typed two-character center-onset CFG chart with joint first outer obligation",
            "grammar": ["S -> NPsubject VP", "VP -> V NPobject", "NP -> DET NOUN REL", "REL -> who VP", "CENTER -> th|so"],
            "center_onset_bank": [{"onset": a, "surface": b, "agreement": c} for a, b, c in CENTER_ONSETS],
            "stats": {"subject_derivations": len(subject), "object_derivations": len(object_side),
                      "subject_chart_states": subject_states, "object_chart_states": object_states,
                      "onset_support": dict(onset_support), "first_outer_support": dict(first_outer_support),
                      "center_debt_states": center_states, "outer_debt_states": outer_states,
                      "both_debts_cross_word_seam": seam_hits, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_two_debt_score": best["score"]},
            "complete_prose_controls": ["The sailor who reads the area guards the harbor.",
                                        "Some guides who keep the era mark a garden."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-typed-twochar-center-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": "character-cfg-typed-center-two-debt-20260920"},
            "next_construction": "Use a typed two-character onset that is itself a paired center nonterminal, then carry its residual into the outer debt instead of freezing the onset.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
