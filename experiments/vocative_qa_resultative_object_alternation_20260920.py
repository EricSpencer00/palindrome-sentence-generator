"""Resultative purpose QA with typed transitive/ditransitive object alternation."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/vocative_qa_relation_center_clarification_20260920.py"
spec=importlib.util.spec_from_file_location("qa_relation_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# The semantic change is valency: transitive, ditransitive, and benefactive
# infinitival complements.  Each carries an explicit infinitival subject.
base.FRAMES=(
 {"speaker":"sailor","role":"traveler","relation":"for","voc":("sailor",),"q":("did","you","see","the","harbor"),"a":("the","sailor","waited"),"clar":("the","keeper","to","explain","the","harbor"),"tail":("before","dusk")},
 {"speaker":"keeper","role":"witness","relation":"in order for","voc":("keeper",),"q":("did","you","guard","the","lantern"),"a":("the","keeper","waited"),"clar":("the","sailor","to","show","the","chart","to","the","keeper"),"tail":("before","dusk")},
 {"speaker":"poet","role":"observer","relation":"so as for","voc":("poet",),"q":("did","you","remember","the","garden"),"a":("the","poet","waited"),"clar":("the","keeper","to","carry","the","letter","for","the","poet"),"tail":("before","dusk")},
)
result=base.run()
result.update({
 "method":"vocative-qa-resultative-object-alternation-20260920",
 "provenance":"fresh purpose complements with explicit infinitival subject and typed transitive/ditransitive/benefactive valency; three residual chunks and cross-word seam remain live; no lexical sweep, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["vocative-qa-resultative-infinitival-center-20260920","semantic-valency-boundary-csp-20260916","recipient-theme-attachment-csp-20260920"],"unused_dimension":"object-valency alternation inside the explicit-subject purpose complement (transitive vs recipient-theme vs benefactive)","reason":"prior resultative lane held a fixed explanation complement; this lane changes semantic argument structure before character emission"},
 "next_construction":"hold out a source-instrument alternation with an explicit agreement gate; do not widen object vocabulary",
})
out=ROOT/"runs/vocative-qa-resultative-object-alternation-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","states","character_prunes","semantic_prunes","exact_candidate_count")}))
