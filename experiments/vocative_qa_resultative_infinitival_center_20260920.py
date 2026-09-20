"""Resultative purpose complement with explicit infinitival subject."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/vocative_qa_relation_center_clarification_20260920.py"
spec=importlib.util.spec_from_file_location("qa_relation_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# Explicit subject of the infinitive is carried as a semantic attachment, not
# flattened into the finite answer clause.
base.FRAMES=(
 {"speaker":"sailor","role":"traveler","relation":"for","voc":("sailor",),"q":("did","you","see","the","harbor"),"a":("the","sailor","waited"),"clar":("the","keeper","to","explain"),"tail":("that","the","harbor","was","quiet")},
 {"speaker":"keeper","role":"witness","relation":"in order for","voc":("keeper",),"q":("did","you","guard","the","lantern"),"a":("the","keeper","waited"),"clar":("the","sailor","to","explain"),"tail":("that","the","lantern","was","bright")},
 {"speaker":"poet","role":"observer","relation":"so as for","voc":("poet",),"q":("did","you","remember","the","garden"),"a":("the","poet","waited"),"clar":("the","keeper","to","explain"),"tail":("that","the","garden","was","peaceful")},
)
result=base.run()
result.update({
 "method":"vocative-qa-resultative-infinitival-center-20260920",
 "provenance":"fresh resultative purpose complement with explicit infinitival subject (for NP to explain) over role-typed vocative QA; three residual chunks and cross-word seam remain live; no finite purpose sweep, evidential/relative relation, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["vocative-qa-purpose-attachment-center-20260920","vocative-qa-evidential-relative-center-20260920","vocative-qa-three-chunk-clarification-20260920"],"unused_dimension":"explicit infinitival subject attachment at the purpose center (for/in order for/so as for + NP + to-infinitive)","reason":"prior purpose lane used finite complements; no role-typed vocative QA lane carries an explicit infinitival subject through the three residual chunks"},
 "next_construction":"hold out a resultative infinitive with a controlled object alternation; do not widen purpose connectives",
})
out=ROOT/"runs/vocative-qa-resultative-infinitival-center-20260920.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","states","character_prunes","semantic_prunes","exact_candidate_count")}))
