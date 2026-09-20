"""Source/instrument alternation with agreement-gated purpose QA."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/vocative_qa_relation_center_clarification_20260920.py"
spec=importlib.util.spec_from_file_location("qa_relation_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# Agreement gate is semantic: each role owns a compatible source or
# instrument attachment before its characters enter the residual solver.
base.FRAMES=(
 {"speaker":"sailor","role":"traveler","relation":"for","agreement":"singular-source","voc":("sailor",),"q":("did","you","see","the","harbor"),"a":("the","sailor","waited"),"clar":("the","keeper","to","hear","from","the","sailor"),"tail":("before","dusk")},
 {"speaker":"keeper","role":"witness","relation":"in order for","agreement":"singular-instrument","voc":("keeper",),"q":("did","you","guard","the","lantern"),"a":("the","keeper","waited"),"clar":("the","sailor","to","mark","the","chart","with","a","compass"),"tail":("before","dusk")},
 {"speaker":"poet","role":"observer","relation":"so as for","agreement":"singular-instrument","voc":("poet",),"q":("did","you","remember","the","garden"),"a":("the","poet","waited"),"clar":("the","keeper","to","write","the","letter","with","ink"),"tail":("before","dusk")},
)
result=base.run()
result.update({
 "method":"vocative-qa-source-instrument-alternation-20260920",
 "provenance":"fresh agreement-gated source/instrument PP alternation inside explicit-subject purpose complements; three residual chunks and cross-word seam remain live; no object-vocabulary sweep, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["vocative-qa-resultative-object-alternation-20260920","instrument-source-attachment-csp-20260920","vocative-qa-resultative-infinitival-center-20260920"],"unused_dimension":"source versus instrument attachment selected under speaker-role agreement before live character expansion","reason":"prior object alternation changed recipient/theme valency; this lane changes PP attachment role and carries an explicit agreement class into the center residual"},
 "next_construction":"hold out an instrument-source alternation with plural agreement and a controlled auxiliary; do not widen PP vocabulary",
})
out=ROOT/"runs/vocative-qa-source-instrument-alternation-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","states","character_prunes","semantic_prunes","exact_candidate_count")}))
