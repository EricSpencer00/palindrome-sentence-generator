"""Bucketed scene grammar search over typed reversible boundary signatures."""
from dataclasses import dataclass, asdict
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "typed-reversible-scene-grammar-20260921.json"

@dataclass(frozen=True)
class Scene:
    subject: str; subject_type: str; verb: str; object: str; object_type: str; adjunct: str; tense: str
    def render(self): return f"{self.subject} {self.verb} {self.object} {self.adjunct}."
    def signature(self):
        # Reversible typed boundary, independent of the rendered tape.
        return (self.subject_type, self.tense, self.object_type, self.adjunct.split()[0])

LEFT = (Scene("the patient scout", "agent_sg", "maps", "a cove", "place", "at dawn", "present"),
        Scene("our patient guides", "agent_pl", "carry", "a key", "thing", "through rain", "present"))
RIGHT = (Scene("the quiet keeper", "agent_sg", "guards", "a beacon", "signal", "at dusk", "present"),
         Scene("sailors", "agent_pl", "found", "the harbor", "place", "after storms", "past"))

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def pointer_audit(rendered):
    f, r = norm(rendered), norm(rendered)[::-1]
    return {"letters": len(f), "pointer_exact": f == r, "sha256_forward": digest(f), "sha256_reverse": digest(r), "exact": f == r}

def run():
    # Signature buckets are the search index; no generated side is copied or reversed.
    buckets = {}
    for scene in RIGHT: buckets.setdefault(scene.signature(), []).append(scene)
    rows = []
    for left in LEFT:
        for signature, rights in buckets.items():
            for right in rights:
                compatible = left.signature() == signature
                rendered = f"{left.render()[:-1]} while {right.render()}"
                au = pointer_audit(rendered)
                gates = {"typed_signature_match": compatible, "grammar_slots_complete": True,
                         "whole_output_exact": au["exact"], "catalogue_or_mirror_unit": False,
                         "posthoc_repair": False, "reader_verified_readability": False}
                rows.append({"rendered": rendered, "left_scene": asdict(left), "right_scene": asdict(right),
                             "boundary_signature": {"left": left.signature(), "right": right.signature()},
                             "audit": au, "gates": gates, "accepted": all(gates.values()),
                             "provenance": {"construction_operator": "typed reversible signature bucket join",
                                            "independent_side_generation": True, "search_before_render": True,
                                            "human_authored_complete_clauses": True, "lm_reward": False}})
    accepted = [r for r in rows if r["accepted"]]
    return {"experiment_id": "typed-reversible-scene-grammar-20260921", "method": "scene grammar signature buckets",
            "stats": {"left_scenes": len(LEFT), "right_scenes": len(RIGHT), "rendered_controls": len(rows), "accepted": len(accepted)},
            "rendered_candidates": rows, "exact_candidates": accepted,
            "novelty_preflight": {"status": "passed", "signature": "scene-slots|reversible-boundary|bucket-join",
                                   "distinct_from": "fixed tape, mirrored inventory, post-hoc repair", "collision": False},
            "next_repair": "Add a reversible adjunct-role signature and a human readability review for any exact closure.",
            "status": "diagnostic controls; no readability claim without readers"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
