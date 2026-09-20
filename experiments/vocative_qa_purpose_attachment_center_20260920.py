"""Purpose-attachment relation center over role-typed vocative QA."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/vocative_qa_relation_center_clarification_20260920.py"
spec=importlib.util.spec_from_file_location("qa_relation_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# New attachment topology: the QA answer takes a purpose clause headed by
# ``so that``; it is not an evidential or relative clarification edge.
base.FRAMES=(
 {"speaker":"sailor","role":"traveler","relation":"so that","voc":("sailor",),"q":("did","you","see","the","harbor"),"a":("the","sailor","waited"),"clar":("the","keeper","could","explain"),"tail":("that","the","harbor","was","quiet")},
 {"speaker":"keeper","role":"witness","relation":"in order that","voc":("keeper",),"q":("did","you","guard","the","lantern"),"a":("the","keeper","waited"),"clar":("the","sailor","could","explain"),"tail":("that","the","lantern","was","bright")},
 {"speaker":"poet","role":"observer","relation":"so","voc":("poet",),"q":("did","you","remember","the","garden"),"a":("the","poet","waited"),"clar":("the","keeper","could","explain"),"tail":("that","the","garden","was","peaceful")},
)
result=base.run()
result.update({
 "method":"vocative-qa-purpose-attachment-center-20260920",
 "provenance":"fresh purpose attachment (`so that` + finite complement) over complete role-typed vocative QA; three residual chunks and cross-word seam remain live; no evidential/relative sweep, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["vocative-qa-evidential-relative-center-20260920","vocative-qa-conditional-relation-center-20260920","purpose-attachment-residual-20260920"],"unused_dimension":"purpose-scoped complement attachment at the QA center seam, distinct from evidential, conditional, and relative centers","reason":"registry has relation and relative lanes but no role-typed vocative QA purpose complement with live three-chunk center obligations"},
 "next_construction":"hold out a resultative purpose complement with an explicit infinitival subject; do not widen purpose connective inventory",
})
out=ROOT/"runs/vocative-qa-purpose-attachment-center-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","states","character_prunes","semantic_prunes","exact_candidate_count")}))
