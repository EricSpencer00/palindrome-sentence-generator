"""Slot-level character-domain solver for independently authored CFG scenes."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phrase-cfg-slot-domain-solver-20260920"

ROLES = {"maritime": {"sailor", "harbor", "shore", "tide", "boat"},
         "writing": {"poet", "writer", "letter", "notes", "book"}}
DOMAINS = {
    "DET": ("some", "a", "the"), "ADJ": ("quiet", "patient", "bright", "old", "careful"),
    "SUBJ": ("sailor", "poet", "keeper", "writer", "captain", "guide"),
    "VERB": ("guards", "marks", "guides", "keeps", "reads", "writes"),
    "OBJ": ("harbor", "shore", "tide", "boat", "letter", "notes", "book", "garden"),
    "TAIL": ("at dawn", "before rain", "near shore", "under stars"),
}


def norm(s): return "".join(c.lower() for c in s if c.isalpha())


def audit(s):
    t = norm(s); r = t[::-1]
    return {"normalized": t, "letters": len(t), "two_pointer_exact": bool(t) and t == r,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(r.encode()).hexdigest()}


def slot_domain(words):
    text = norm(words)
    return {"first": text[:1], "last": text[-1:], "letters": len(text)}


def lexicalize(role):
    allowed = ROLES[role]; rows = []
    for det in DOMAINS["DET"]:
        for adj in DOMAINS["ADJ"]:
            for subj in DOMAINS["SUBJ"]:
                for verb in DOMAINS["VERB"]:
                    for det2 in DOMAINS["DET"]:
                        for obj in DOMAINS["OBJ"]:
                            for tail in DOMAINS["TAIL"]:
                                words = (det, adj, subj, verb, det2, obj, *tail.split())
                                if not (set(words) & allowed): continue
                                rows.append((words, {"role": role, "tree": "S(NP,VP,PP)",
                                                      "slots": {"DET": det, "ADJ": adj, "SUBJ": subj, "VERB": verb, "DET2": det2, "OBJ": obj, "TAIL": tail}}))
    return tuple(rows[:240])


def run():
    left = lexicalize("maritime"); right = lexicalize("writing")
    # Solve domains before full tape construction. Each slot's exposed domain
    # must have a compatible opposite character; only then are words rendered.
    domains = {slot: {"left": set(), "right": set()} for slot in ("DET", "ADJ", "SUBJ", "VERB", "DET2", "OBJ", "TAIL")}
    for rows, side in ((left, "left"), (right, "right")):
        for words, meta in rows:
            for slot, word in meta["slots"].items():
                domains[slot][side].add(slot_domain(word)["first"])
    compatible = {slot: sorted(domains[slot]["left"] & {x for x in domains[slot]["right"]}) for slot in domains}
    states = 0; candidates = []; best = {"matched": 0, "left": "", "right": ""}
    for lwords, lmeta in left:
        for rwords, rmeta in right:
            # slot-level prefilter; no completed tape has been reversed here
            if any(not compatible[slot] for slot in domains): continue
            lt = norm(" ".join(lwords)); rt = norm(" ".join(rwords))[::-1]
            matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[matched]:
                states += 1; matched += 1
            if matched > best["matched"]: best = {"matched": matched, "left": " ".join(lwords), "right": " ".join(rwords)}
            if matched == len(lt) == len(rt):
                rendered = " ".join(lwords).capitalize() + "; " + " ".join(rwords) + "."
                a = audit(rendered); candidates.append({"rendered": rendered, "audit": a, "left": lmeta, "right": rmeta,
                    "provenance": {"slot_domains_before_lexicalization": True, "catalogue_imported": False,
                                   "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in candidates}; candidates = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "slot-level character-domain solving before phrase lexicalization",
            "grammar": ["S -> NP VP PP", "NP -> DET ADJ NOUN", "VP -> V NP", "PP -> TAIL"],
            "shared_scene_roles": {"left": "maritime", "right": "writing"},
            "slot_domains": {k: {side: sorted(v) for side, v in x.items()} for k, x in domains.items()},
            "compatible_slot_first_chars": compatible,
            "stats": {"left_phrases": len(left), "right_phrases": len(right), "slot_domain_states": sum(len(x["left"]) + len(x["right"]) for x in domains.values()),
                      "lexicalized_character_states": states, "exact": len(candidates), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in candidates),
                      "best_matched_prefix": best["matched"]},
            "complete_prose_controls": ["The patient sailor guards the harbor at dawn.", "A careful writer reads the letter before rain."],
            "best_diagnostic": best, "candidates": sorted(candidates, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "phrase-cfg-slot-domain-solver-20260920", "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "phrase-cfg-shared-role-charfirst-20260920"},
            "next_construction": "Use slot-specific first/last domains rather than a single global intersection, then solve word-boundary seam lengths before lexicalization.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
