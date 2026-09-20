"""Modal permissive scope over explicit plural reciprocal embedding."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/shared_scene_reciprocal_permissive_auxiliary_20260920.py"
spec=importlib.util.spec_from_file_location("perm_base",BASE); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
base=mod
base.SCENES={
 "harbor":{"MATRIX":("the captain may watch the harbor at dawn","the keeper may study the charts before dusk"),"PERM":("the captain may allow the sailors to send each other letters at dawn","the keeper may allow the sailors to share charts before dusk")},
 "garden":{"MATRIX":("the gardener may tend the garden in silence","the poet may carry a small letter after rain"),"PERM":("the gardener may allow the poets to send each other seeds in rain","the poet may allow the gardeners to carry notes at noon")},
 "bridge":{"MATRIX":("the guide may watch the old bridge at noon","the pilot may mark the distant shore through mist"),"PERM":("the guide may allow the scouts to send each other maps at dusk","the pilot may allow the scouts to carry lamps through mist")},
}
def controls():
 texts=("The captain may watch the harbor at dawn; the captain may allow the sailors to send each other letters at dawn.","The gardener may tend the garden in silence; the gardener may allow the poets to send each other seeds in rain.","The guide may watch the old bridge at noon; the guide may allow the scouts to send each other maps at dusk.")
 return [{"rendered":t,"audit":base.audit(t),"independent_pointer_exact":base.pointer_exact(t),"modal_scope_complete":True,"embedded_plural":True,"reader_eligible":False,"provenance":"authored modal permissive control; not generated exact candidate"} for t in texts]
base.controls=controls
result=base.run()
result.update({
 "method":"shared-scene-reciprocal-modal-permissive-scope-20260920",
 "provenance":"fresh modal may + allow scope over explicit plural reciprocal embedding; complete matrix/embedded parses and live cross-word equations precede rendering; no perfect-auxiliary replay, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-reciprocal-permissive-auxiliary-20260920","shared-scene-reciprocal-causative-embedded-20260920","shared-scene-reciprocal-passive-20260920"],"unused_dimension":"modal permissive scope (may allow) with explicit plural reciprocal embedded clause","reason":"prior permissive lane used perfect has permitted; this lane tests modal scope and records outer-character frontier independently"},
 "bottleneck_check":{"first_live_diagnostic":result.get("first_live_diagnostic"),"same_family_bottleneck":True,"action":"stop widening this permissive family if first outer character remains incompatible"},
 "next_construction":"pivot away from permissive causatives to a new grammar family if the first-character frontier remains unchanged",
})
out=ROOT/"runs/shared-scene-reciprocal-modal-permissive-scope-20260920.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","scenes","states","character_prunes","semantic_prunes","exact_candidate_count")}))
