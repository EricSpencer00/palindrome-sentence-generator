"""Plural auxiliary reciprocal double-object shared-scene CSP."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/shared_scene_reciprocal_dative_object_20260920.py"
spec=importlib.util.spec_from_file_location("dative_base",BASE); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
base=mod.base

base.SCENES={
 "harbor":{"SVO":("the sailors have guarded the lantern at dawn","the keepers have studied the charts before dusk"),"BEN":("the sailors have sent each other letters at dawn","the keepers have given each other charts before dusk")},
 "garden":{"SVO":("the poets have remembered the garden in silence","the gardeners have carried small letters after rain"),"BEN":("the poets have sent each other seeds in rain","the gardeners have given each other notes at noon")},
 "bridge":{"SVO":("the scouts have watched the old bridge at noon","the pilots have marked the distant shore through mist"),"BEN":("the scouts have sent each other maps at dusk","the pilots have given each other lamps through mist")},
}
def controls():
 texts=("The sailors have guarded the lantern at dawn; the sailors have sent each other letters at dawn.","The poets have remembered the garden in silence; the poets have sent each other seeds in rain.","The scouts have watched the old bridge at noon; the scouts have sent each other maps at dusk.")
 return [{"rendered":t,"audit":base.audit(t),"independent_pointer_exact":base.pointer_exact(t),"plural_have_agreement":True,"reciprocal_double_object":True,"reader_eligible":False,"provenance":"authored plural auxiliary/double-object control; not generated exact candidate"} for t in texts]
base.controls=controls
result=base.run()
result.update({
 "method":"shared-scene-reciprocal-double-object-auxiliary-20260920",
 "provenance":"fresh plural have auxiliary agreement plus reciprocal double-object syntax (each other + theme) under shared scene keys; complete roles and live cross-word equations precede rendering; no vocabulary widening, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-reciprocal-dative-object-20260920","plural-auxiliary-source-instrument-20260920","shared-scene-reciprocal-benefactive-20260920"],"unused_dimension":"plural perfect auxiliary agreement coupled to reciprocal double-object realization","reason":"prior reciprocal dative lane was simple present; prior plural auxiliary lane used source/instrument PP, not reciprocal double-object syntax"},
 "next_construction":"hold out a past-perfect had auxiliary with reciprocal double-object agreement; do not widen reciprocal vocabulary",
})
out=ROOT/"runs/shared-scene-reciprocal-double-object-auxiliary-20260920.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","scenes","states","character_prunes","role_prunes","exact_candidate_count")}))
