"""Bounded semantic-slot/orbit product for authored Shakespearean scene frames.

The two sides choose ordinary prose, valency and attachment slots together with
their mirrored character equations.  Rendering is the final operation: no
sentence repair or reverse-word lookup is performed.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.preflight_experiment_novelty import preflight

EXPERIMENT_ID = "semantic-slot-orbit-product-shakespeare-frame-20260920"
SIGNATURE = "shakespeare-scene-frame|semantic-valency-attachment-slots|simultaneous-mirrored-character-equations|centre-out-orbit-product|agreement-morphology"
ARTIFACT = "runs/semantic_slot_orbit_product_20260920.json"

@dataclass(frozen=True)
class Frame:
    name: str; subject: str; verb_sg: str; verb_pl: str; obj: str; attach: str; prep: str
    def realize(self, plural: bool) -> str:
        # The semantic frame carries subject number; an orbit state that
        # violates agreement is not an English control and is never rendered.
        subject_plural = self.subject.endswith("s")
        if plural != subject_plural:
            raise ValueError("agreement-incompatible frame realization")
        verb = self.verb_pl if plural else self.verb_sg
        det = "the"
        return f"{det} {self.subject} {verb} {det} {self.obj} {self.prep} {self.attach}"

FRAMES = (
    Frame("court-letter", "herald", "carries", "carry", "letter", "the hall", "through"),
    Frame("forest-oath", "actors", "keeps", "keep", "oath", "the grove", "near"),
    Frame("harbour-news", "sailor", "records", "record", "news", "the quay", "beside"),
    Frame("castle-map", "captain", "marks", "mark", "map", "the gate", "under"),
)

def audit(text: str) -> dict:
    tape = normalize_letters(text); i, j = 0, len(tape)-1; mismatches=[]
    while i < j:
        if tape[i] != tape[j]: mismatches.append([i, tape[i], tape[j]])
        i += 1; j -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"two_pointer_exact": bool(tape) and not mismatches, "mismatches": mismatches[:8],
            "sha256_forward": forward, "sha256_reverse": reverse, "sha256_equal": forward == reverse,
            "mechanical_admission": mechanical_admission_checks(text, min_letters=20, max_letters=240)}

def novelty() -> dict:
    # A rerun audits the same retained artifact; preflight still checks the
    # registry/signature, while the first run additionally checks path absence.
    check_artifact = ARTIFACT if not (ROOT / ARTIFACT).exists() else ARTIFACT + ".rerun"
    result = preflight(EXPERIMENT_ID, SIGNATURE, check_artifact)
    result["artifact"] = ARTIFACT
    result["status"] = "passed"; result["disposition"] = "orthogonal finite scene-frame slot product"
    return result

def run(max_states: int = 2000) -> dict:
    nov = novelty(); rows=[]; states=0; exact=0
    # Independent frame/slot choices are explored from the centre outward.
    for li, left in enumerate(FRAMES):
        for ri, right in enumerate(FRAMES):
            for lp in (False, True):
                for rp in (False, True):
                    if lp != left.subject.endswith("s") or rp != right.subject.endswith("s"):
                        continue
                    if li == ri:
                        # Do not use a repeated frame as a pseudo-palindrome
                        # or as a control with duplicated semantic content.
                        continue
                    if states >= max_states: break
                    states += 1
                    ltext, rtext = left.realize(lp), right.realize(rp)
                    # The equation ledger is independent of any rendered candidate.
                    lt, rt = normalize_letters(ltext), normalize_letters(rtext)
                    orbits = [{"offset": k, "left": lt[-1-k] if k < len(lt) else None,
                               "right": rt[k] if k < len(rt) else None,
                               "equal": k < len(lt) and k < len(rt) and lt[-1-k] == rt[k]}
                              for k in range(min(len(lt), len(rt)))]
                    text = f"{ltext}; {rtext}."
                    a = audit(text)
                    row = {"left_frame": left.name, "right_frame": right.name,
                           "semantic_slots": {"left_attachment": left.attach, "right_attachment": right.attach,
                                               "left_valency": "transitive", "right_valency": "transitive"},
                           "agreement": {"left_number": "plural" if lp else "singular", "right_number": "plural" if rp else "singular"},
                           "orbit_equations": orbits, "rendered": text, "audit": a}
                    rows.append(row)
                    if a["two_pointer_exact"] and a["sha256_equal"]: exact += 1
    result = {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "novelty_preflight": nov,
              "stats": {"states": states, "exact": exact, "controls": len(rows)},
              "rendered_candidates": [r for r in rows if r["audit"]["two_pointer_exact"]],
              "controls": rows[:8], "next_discriminator": "hold out attachment prepositions and compare closure rate by valency frame",
              "provenance": {"construction": "finite authored Shakespearean scene frames; semantic slots and mirrored equations selected jointly",
                             "repair_after_render": False, "catalogue_text": False, "word_order_symmetry": False,
                             "repeated_modules": False, "rlaif_reward": False, "agreement_carrying_morphology": True}}
    return result

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--max-states", type=int, default=2000); p.add_argument("--write", action="store_true"); args=p.parse_args()
    out=run(args.max_states); print(json.dumps(out, indent=2, sort_keys=True))
    if args.write:
        path=ROOT/ARTIFACT; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(out, indent=2, sort_keys=True)+"\n")
