import json
from dataclasses import replace
from pathlib import Path
from llm_palindrome.typed_phrase_compiler import PhraseRule,compile_phrase_rules,repetition_penalty
from llm_palindrome.bilateral_seams import Chunk,OwedSideBilateralGrammar,AllChunkCentersGrammar,norm

TARGET='Do geese on a cedar trade canoe see God?'
INTERFACES={('START','SUBJECT_PP'),('SUBJECT_PP','CANOE_HEAD'),('CANOE_HEAD','BARE_VP'),('BARE_VP','END')}
common=(('observer','geese'),('subject_number','plural'),('perceived','God'))
head=PhraseRule('START','SUBJECT_PP','Do geese ','plural auxiliary question subject',common,'019-derived compiler control')
tail=PhraseRule('BARE_VP','END','see God?','base-form VP licensed by question do',common,'019-derived compiler control')
def pp(material):
    return PhraseRule('SUBJECT_PP','CANOE_HEAD','on a '+material+' ','locative PP determiner and material modifier',(('boat_entity','boat1'),('material',material)),'independent donor')
def noun(use):
    return PhraseRule('CANOE_HEAD','BARE_VP',use+' canoe ','canoe noun compound completes the PP',(('boat_entity','boat1'),('use',use)),'independent donor')
# Both fully grammatical donor sentences are nonpalindromic. The successful
# cross-combination is never provided as a complete source proposal.
donors=[[head,pp('cedar'),noun('fishing'),tail],[head,pp('wooden'),noun('trade'),tail]]
for donor in donors:
    t=norm(''.join(r.text for r in donor));assert t!=t[::-1]
private=[]
for i,donor in enumerate(donors):
    states=['START']+[f'donor{i}-{j}' for j in range(3)]+['END']
    private.extend(Chunk(f'donor{i}-{j}',r.text,states[j],states[j+1],r.role,r.source,r.bindings) for j,r in enumerate(donor))
flat=[r for donor in donors for r in donor]
shared=compile_phrase_rules('observe-geese',flat,INTERFACES)
receipts={}
for name,cls in [('outside_in',OwedSideBilateralGrammar),('center_out',AllChunkCentersGrammar)]:
    baseline=cls(private).search(max_steps=4,max_states=1000,max_outputs=100,seconds=2)
    changed=cls(shared).search(max_steps=4,max_states=1000,max_outputs=100,seconds=2)
    assert baseline['receipt']['status']=='complete_bounded_depth'
    assert changed['receipt']['status']=='complete_bounded_depth'
    assert not baseline['outputs']
    assert {o['text'] for o in changed['outputs']}=={TARGET}
    # Same exact letters cannot bypass incompatible entity and number features.
    bad_entity=[replace(r,bindings=tuple((k,'boat2' if k=='boat_entity' else v) for k,v in r.bindings)) if r.text=='trade canoe ' else r for r in flat]
    assert not cls(compile_phrase_rules('observe-geese',bad_entity,INTERFACES)).search(max_steps=4,max_states=1000,max_outputs=100,seconds=2)['outputs']
    bad_number=[replace(r,bindings=tuple((k,'singular' if k=='subject_number' else v) for k,v in r.bindings)) if r.text=='see God?' else r for r in flat]
    assert not cls(compile_phrase_rules('observe-geese',bad_number,INTERFACES)).search(max_steps=4,max_states=1000,max_outputs=100,seconds=2)['outputs']
    receipts[name]={'private':baseline['receipt'],'shared':changed['receipt'],'output':changed['outputs'][0]}
assert repetition_penalty(TARGET)==0
assert repetition_penalty('Draw a ward. I draw a ward.')>0
# Penalty is not consulted by compilation or exactness admission.
repeat_chunks=[Chunk('repeat-l','Draw a ward. ','START','MID','imperative','soft-cost diagnostic'),Chunk('repeat-r','I draw a ward.','MID','END','acknowledgment','soft-cost diagnostic')]
assert OwedSideBilateralGrammar(repeat_chunks).search(max_steps=2,seconds=2)['outputs']
result={'control_only':True,'new_quality_claim':False,'input_complete_donors':len(donors),'private_chunks':len(private),'shared_chunks':len(shared),'known_target_generated_from_unenumerated_combination':True,'no_new_words_or_phrases':True,'entity_conflict_rejected':True,'number_conflict_rejected':True,'repetition_penalized_not_blocked':True,'methods':receipts}
Path('/Users/eric/Documents/Codex/2026-10-09/task-3/compiler-regression-024.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='methods'},indent=2))
