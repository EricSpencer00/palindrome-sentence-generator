"""Plural auxiliary agreement over source/instrument purpose QA."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/vocative_qa_relation_center_clarification_20260920.py"
spec=importlib.util.spec_from_file_location("qa_relation_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# Every answer has plural subject + have auxiliary; the PP attachment must
# retain that agreement class before entering the character residual solver.
base.FRAMES=(
 {"speaker":"sailors","role":"traveler","relation":"for","agreement":"plural-have-source","voc":("sailors",),"q":("did","you","see","the","harbor"),"a":("the","sailors","have","waited"),"clar":("the","keepers","to","hear","from","the","sailors"),"tail":("before","dusk")},
 {"speaker":"keepers","role":"witness","relation":"in order for","agreement":"plural-have-instrument","voc":("keepers",),"q":("did","you","guard","the","lantern"),"a":("the","keepers","have","waited"),"clar":("the","sailors","to","mark","the","charts","with","a","compass"),"tail":("before","dusk")},
 {"speaker":"poets","role":"observer","relation":"so as for","agreement":"plural-have-instrument","voc":("poets",),"q":("did","you","remember","the","gardens"),"a":("the","poets","have","waited"),"clar":("the","keepers","to","write","the","letters","with","ink"),"tail":("before","dusk")},
)
result=base.run()
result.update({
 "method":"vocative-qa-plural-auxiliary-source-instrument-20260920",
 "provenance":"fresh plural subject + controlled have auxiliary agreement over source/instrument purpose attachments; three residual chunks and cross-word seam remain live; no lexical sweep, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["vocative-qa-source-instrument-alternation-20260920","weighted-morphology-fst-lockstep-20260916","agreement-valency-wfsa-decoder-20260924"],"unused_dimension":"plural have auxiliary agreement coupled to source/instrument attachment inside vocative QA purpose center","reason":"prior source/instrument lane used singular finite answers; this lane requires plural subject, have auxiliary, and plural PP arguments to agree before residual expansion"},
 "next_construction":"hold out a plural had-plus-participle auxiliary with instrument/source alternation; do not widen the lexical PP bank",
})
out=ROOT/"runs/vocative-qa-plural-auxiliary-source-instrument-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","states","character_prunes","semantic_prunes","exact_candidate_count")}))
