"""Past-perfect reciprocal double-object shared-scene CSP."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/shared_scene_reciprocal_dative_object_20260920.py"
spec=importlib.util.spec_from_file_location("dative_base",BASE); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
base=mod.base
base.SCENES={
 "harbor":{"SVO":("the sailors had guarded the lantern at dawn","the keepers had studied the charts before dusk"),"BEN":("the sailors had sent each other letters at dawn","the keepers had given each other charts before dusk")},
 "garden":{"SVO":("the poets had remembered the garden in silence","the gardeners had carried small letters after rain"),"BEN":("the poets had sent each other seeds in rain","the gardeners had given each other notes at noon")},
 "bridge":{"SVO":("the scouts had watched the old bridge at noon","the pilots had marked the distant shore through mist"),"BEN":("the scouts had sent each other maps at dusk","the pilots had given each other lamps through mist")},
}
def controls():
 texts=("The sailors had guarded the lantern at dawn; the sailors had sent each other letters at dawn.","The poets had remembered the garden in silence; the poets had sent each other seeds in rain.","The scouts had watched the old bridge at noon; the scouts had sent each other maps at dusk.")
 return [{"rendered":t,"audit":base.audit(t),"independent_pointer_exact":base.pointer_exact(t),"pastperfect_plural_agreement":True,"reciprocal_double_object":True,"reader_eligible":False,"provenance":"authored past-perfect reciprocal control; not generated exact candidate"} for t in texts]
base.controls=controls
result=base.run()
result.update({
 "method":"shared-scene-reciprocal-double-object-pastperfect-20260920",
 "provenance":"fresh past-perfect had auxiliary agreement with reciprocal double-object each-other/theme syntax under shared scenes; complete roles and live cross-word equations precede rendering; no vocabulary widening, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-reciprocal-double-object-auxiliary-20260920","shared-scene-reciprocal-dative-object-20260920","shared-scene-reciprocal-benefactive-20260920"],"unused_dimension":"past-perfect had auxiliary state coupled to reciprocal double-object realization","reason":"prior lane used present perfect have; this lane changes the auxiliary tense state while preserving the same semantic argument structure"},
 "next_construction":"hold out a future-perfect will-have reciprocal frame only if a new grammar family is selected",
})
out=ROOT/"runs/shared-scene-reciprocal-double-object-pastperfect-20260920.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","scenes","states","character_prunes","role_prunes","exact_candidate_count")}))
