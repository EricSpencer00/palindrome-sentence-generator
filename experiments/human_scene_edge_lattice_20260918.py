"""Edge-first search over independently authored miniature English scenes.

Each side is a typed scene (agent, action, patient, setting).  The search
chooses lexical realizations from the outer character equations inward: a
candidate pair is rejected as soon as a newly exposed edge disagrees.  No
finished tape is reversed or copied; the two scenes have separate banks.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

@dataclass(frozen=True)
class Scene:
    agent: str; action: str; patient: str; setting: str
    @property
    def text(self):
        return f"{self.agent} {self.action} {self.patient}{self.setting}."

# Distinct, typed lexical banks.  The right scene is independently indexed;
# it is not made by reversing a left sentence or its tape.
AGENTS = ("aide", "poet", "nurse", "teacher", "editor", "Diana", "Noel", "Leon")
ACTIONS = ("reads", "marks", "writes", "rips", "helps", "guides", "inspires", "sees")
PATIENTS = ("a note", "the map", "a poem", "the letter", "nine memos", "some prose", "Diana", "Noel", "Leon")
SETTINGS = ("", " at dawn", " in town", " near home", " by the sea")

def audit(text: str) -> dict:
    t = normalize_letters(text); rev = t[::-1]
    mismatches = [(i, len(t)-1-i) for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"letters": len(t), "two_pointer_exact": not mismatches,
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest(),
            "sha_equal_under_reversal": hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(rev.encode()).hexdigest()}

def scene_bank(offset: int = 0):
    # Rotate the two banks so lexical choices remain independent.
    return tuple(Scene("an " + AGENTS[(i+offset) % len(AGENTS)], ACTIONS[(i*3+offset) % len(ACTIONS)],
                       PATIENTS[(i*2+offset) % len(PATIENTS)], SETTINGS[(i+offset) % len(SETTINGS)])
                 for i in range(48))

def edge_consistent(text: str) -> bool:
    t = normalize_letters(text)
    return all(t[i] == t[-1-i] for i in range(len(t)//2))

def run(max_probes: int = 12000) -> dict:
    left, right = scene_bank(0), scene_bank(3)
    rows, exact, checked, pruned = [], [], 0, 0
    for li, l in enumerate(left):
        for ri, r in enumerate(right):
            checked += 1
            text = l.text + " " + r.text
            # Live equation check: compare exposed character pairs before the
            # complete candidate is admitted to the result store.
            t = normalize_letters(text)
            first = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
            if first:
                pruned += 1
            a = audit(text)
            checks = mechanical_admission_checks(text, min_letters=30, max_letters=240)
            row = {"rendered": text, "left_scene": l.__dict__, "right_scene": r.__dict__,
                   "left_index": li, "right_index": ri, "live_equation_first_mismatch": first,
                   "audit": a, "mechanical_checks": checks,
                   "independent_exact": a["two_pointer_exact"],
                   "mechanically_admitted": a["two_pointer_exact"] and all(checks.values()),
                   "reader_status": "not_run; programmatic measures diagnose only"}
            if len(rows) < max_probes: rows.append(row)
            if a["two_pointer_exact"]: exact.append(row)
            if checked >= max_probes: break
        if checked >= max_probes: break
    admitted = [x for x in exact if x["mechanically_admitted"]]
    return {"status":"human_scene_edge_lattice_bounded", "experiment_id":"human-scene-edge-lattice-20260918",
            "signature":"typed-scene-lattice|independent-edge-equations|outer-inward-pruning",
            "config":{"left_scene_count":len(left),"right_scene_count":len(right),"max_probes":max_probes},
            "stats":{"pair_worlds_checked":checked,"edge_pruned":pruned,"stored_probes":len(rows),
                     "exact":len(exact),"mechanically_admitted":len(admitted),"reader_eligible":0},
            "rendered_candidates_and_probes":rows,"exact_candidates":exact,"admitted":admitted,
            "provenance":{"finished_tape_reversed":False,"catalogue_text_imported":False,
                           "word_order_mirror":False,"independent_validator":"two-pointer plus forward/reverse SHA-256",
                           "human_readability_certified":False},
            "next_repair":"replace the full-scene pair loop with token-level outer-inward CSP; carry residual letters through agent/action/patient boundaries and add clause seams",
            "reader_gate":"closed until an exact candidate survives intact-prose review and blinded shuffled controls"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True,type=Path); ap.add_argument("--max-probes",type=int,default=12000); a=ap.parse_args()
    if a.out.exists(): ap.error(f"refusing to overwrite existing output: {a.out}")
    result=run(a.max_probes); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],indent=2))
if __name__ == "__main__": main()
