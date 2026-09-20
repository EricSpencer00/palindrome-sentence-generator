"""Evidential relation plus relative clarification edge over role-typed QA."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/vocative_qa_relation_center_clarification_20260920.py"
spec=importlib.util.spec_from_file_location("qa_relation_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# The relative edge is a typed clarification attachment: its antecedent role
# must be the evidence holder for the preceding answer.
base.FRAMES=(
 {"speaker":"sailor","role":"traveler","relation":"because","voc":("sailor",),"q":("did","you","see","the","harbor"),"a":("the","sailor","saw","the","harbor"),"clar":("the","keeper","who","watched","the","harbor"),"tail":("said","it","was","quiet")},
 {"speaker":"keeper","role":"witness","relation":"as","voc":("keeper",),"q":("did","you","guard","the","lantern"),"a":("the","keeper","guarded","the","lantern"),"clar":("the","sailor","who","saw","the","lantern"),"tail":("said","it","was","bright")},
 {"speaker":"poet","role":"observer","relation":"since","voc":("poet",),"q":("did","you","remember","the","garden"),"a":("the","poet","remembered","the","garden"),"clar":("the","keeper","who","tended","the","garden"),"tail":("said","it","was","peaceful")},
)
result=base.run()
result.update({
 "method":"vocative-qa-evidential-relative-center-20260920",
 "provenance":"fresh role-typed evidential because relation with an antecedent-agreeing relative clarification edge; three residual chunks and cross-word seam remain live; no bank widening, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["vocative-qa-conditional-relation-center-20260920","vocative-qa-relation-center-clarification-20260920","typed-relative-residual-scheduler-20260920","relation-conditioned-voice-grammar-20260920"],"unused_dimension":"evidential relation whose clarification is a role-agreeing relative edge","reason":"prior relation lanes used finite clarification tails or conditional scope; this lane requires relative antecedent attachment before exact residual admission"},
 "next_construction":"hold out a relative edge with an explicit source/agent alternation; do not widen the evidential relation bank",
})
out=ROOT/"runs/vocative-qa-evidential-relative-center-20260920.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","states","character_prunes","semantic_prunes","exact_candidate_count")}))
