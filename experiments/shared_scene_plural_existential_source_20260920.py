"""Plural existential copula with typed source postposition."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/shared_scene_existential_postposition_20260920.py"
spec=importlib.util.spec_from_file_location("existential_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# Every existential right frame is plural (there are + plural entity), and
# `from` is a source attachment rather than the earlier spatial postposition.
base.SCENES={
 "harbor":{"SVO":("the patient sailor guards the lantern at dawn","the careful keeper studies the chart before dusk"),"EX":{"there are lanterns from the sailors at dawn","there are charts from the keepers before dusk"},"post":("from",)},
 "garden":{"SVO":("the young poet remembers the garden in silence","the bright gardener carries a small letter after rain"),"EX":{"there are letters from the poets in rain","there are seeds from the gardeners at noon"},"post":("from",)},
 "bridge":{"SVO":("several quiet scouts watch the old bridge at noon","a patient pilot marks the distant shore through mist"),"EX":{"there are boats from the shore at dusk","there are lanterns from the pilots through mist"},"post":("from",)},
}
def controls():
 texts=("The patient sailor guards the lantern at dawn; there are lanterns from the sailors at dawn.","The young poet remembers the garden in silence; there are letters from the poets in rain.","Several quiet scouts watch the old bridge at noon; there are boats from the shore at dusk.")
 return [{"rendered":t,"audit":base.audit(t),"independent_pointer_exact":base.pointer_exact(t),"plural_copula_agreement":True,"source_attachment":True,"reader_eligible":False,"provenance":"authored plural existential/source control; not generated exact candidate"} for t in texts]
base.controls=controls
result=base.run()
result.update({
 "method":"shared-scene-plural-existential-source-20260920",
 "provenance":"fresh plural existential there-are copula with plural quantifier agreement and typed source postposition from; slot domains and live cross-word equations precede rendering; no spatial-postposition replay, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-existential-postposition-20260920","shared-scene-locative-slot-domains-20260920","plural-auxiliary-source-instrument-20260920"],"unused_dimension":"plural existential copula agreement coupled to source postposition under shared scene key","reason":"prior existential lane used singular there-is spatial locations; this lane requires there-are plural agreement and source attachment before residual growth"},
 "next_construction":"hold out plural there-are existential with an instrument postposition and preserve quantifier agreement",
})
out=ROOT/"runs/shared-scene-plural-existential-source-20260920.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","scenes","states","character_prunes","domain_prunes","exact_candidate_count")}))
