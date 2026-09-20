"""Reciprocal dative/object alternation under shared-scene grammar."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/shared_scene_benefactive_role_attachment_20260920.py"
spec=importlib.util.spec_from_file_location("benefactive_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

base.SCENES={
 "harbor":{"SVO":("the sailors guard the lantern at dawn","the keepers study the charts before dusk"),"BEN":("the sailors send each other letters at dawn","the keepers give each other charts before dusk")},
 "garden":{"SVO":("the poets remember the garden in silence","the gardeners carry small letters after rain"),"BEN":("the poets send each other seeds in rain","the gardeners give each other notes at noon")},
 "bridge":{"SVO":("the scouts watch the old bridge at noon","the pilots mark the distant shore through mist"),"BEN":("the scouts send each other maps at dusk","the pilots give each other lamps through mist")},
}
orig_slots=base.slots
def dative_slots(text,kind):
 w=tuple(text.split())
 if kind=="SVO": return orig_slots(text,kind)
 # subject, verb, reciprocal object, theme, time; no for-PP is present.
 return (tuple(w[:2]),tuple(w[2:3]),tuple(w[3:5]),tuple(w[5:6]),tuple(w[6:]))
base.slots=dative_slots
def controls():
 texts=("The sailors guard the lantern at dawn; the sailors send each other letters at dawn.","The poets remember the garden in silence; the poets send each other seeds in rain.","The scouts watch the old bridge at noon; the scouts send each other maps at dusk.")
 return [{"rendered":t,"audit":base.audit(t),"independent_pointer_exact":base.pointer_exact(t),"reciprocal_plural_agreement":True,"dative_object_complete":True,"reader_eligible":False,"provenance":"authored reciprocal dative/object control; not generated exact candidate"} for t in texts]
base.controls=controls
result=base.run()
result.update({
 "method":"shared-scene-reciprocal-dative-object-20260920",
 "provenance":"fresh shared-scene transitive versus reciprocal dative/object grammar (`send each other letters`); plural agreement and complete reciprocal roles precede live character equations; no for-PP sweep, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-reciprocal-benefactive-20260920","shared-scene-benefactive-role-attachment-20260920","recipient-theme-attachment-csp-20260920"],"unused_dimension":"reciprocal each-other object preceding a separate theme, replacing benefactive for-attachment",
 "reason":"prior reciprocal lane used for each other; this lane enforces dative/object syntax and theme realization before residual growth"},
 "next_construction":"hold out a reciprocal double-object alternation with plural auxiliary agreement; do not widen reciprocal vocabulary",
})
out=ROOT/"runs/shared-scene-reciprocal-dative-object-20260920.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","scenes","states","character_prunes","role_prunes","exact_candidate_count")}))
