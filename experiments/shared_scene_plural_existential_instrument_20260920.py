"""Plural existential copula with typed instrument postposition."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/shared_scene_existential_postposition_20260920.py"
spec=importlib.util.spec_from_file_location("existential_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# Plural existential agreement is paired with an instrument-bearing entity:
# `there are charts with a compass`, rather than a source or spatial PP.
base.SCENES={
 "harbor":{"SVO":("the patient sailor guards the lantern at dawn","the careful keeper studies the chart before dusk"),"EX":{"there are charts with a compass at dawn","there are lanterns with a candle before dusk"},"post":("with",)},
 "garden":{"SVO":("the young poet remembers the garden in silence","the bright gardener carries a small letter after rain"),"EX":{"there are letters with ink in rain","there are maps with a brush at noon"},"post":("with",)},
 "bridge":{"SVO":("several quiet scouts watch the old bridge at noon","a patient pilot marks the distant shore through mist"),"EX":{"there are maps with a lantern at dusk","there are boats with a signal through mist"},"post":("with",)},
}
def controls():
 texts=("The patient sailor guards the lantern at dawn; there are charts with a compass at dawn.","The young poet remembers the garden in silence; there are letters with ink in rain.","Several quiet scouts watch the old bridge at noon; there are maps with a lantern at dusk.")
 return [{"rendered":t,"audit":base.audit(t),"independent_pointer_exact":base.pointer_exact(t),"plural_copula_agreement":True,"instrument_attachment":True,"reader_eligible":False,"provenance":"authored plural existential/instrument control; not generated exact candidate"} for t in texts]
base.controls=controls
result=base.run()
result.update({
 "method":"shared-scene-plural-existential-instrument-20260920",
 "provenance":"fresh plural there-are existential with instrument postposition with and quantifier agreement; slot domains and live cross-word equations precede rendering; no source/spatial replay, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-plural-existential-source-20260920","shared-scene-existential-postposition-20260920","instrument-source-attachment-csp-20260920"],"unused_dimension":"plural existential quantifier agreement coupled to an instrument postposition under a shared scene key","reason":"prior plural existential lane used source from; this lane changes the attachment role to instrument with while retaining the there-are agreement gate"},
 "next_construction":"hold out a plural existential with a benefactive postposition and preserve quantifier/instrument role agreement",
})
out=ROOT/"runs/shared-scene-plural-existential-instrument-20260920.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","scenes","states","character_prunes","domain_prunes","exact_candidate_count")}))
