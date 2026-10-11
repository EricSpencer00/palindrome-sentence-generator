import pytest
from llm_palindrome.phrase_api_loop import validate_proposal,blind_batch,validate_ratings,scored_rows

def proposal(text='Step on no pets.'):
 return dict(proposal_id='fixture',method='phrase_bridge_overhang',text=text,construction_blocks=['Step on',' no pets.'],block_roles=['verb/preposition','negative noun phrase'],seams=[dict(left_block=0,right_block=1,grammar_relation='preposition takes negative noun phrase')],intended_scene='Avoid stepping on animals.',copied_sources=['known catalogue control'])
def rating(bid,**kw):return dict(blind_id=bid,grammar=4,readability=4,coherence=3,repetition_burden=0,meaningful_progression=True,padding_or_loop=False,rationale='fixture',**kw)
def test_phrase_control_and_rendering():
 a=validate_proposal(proposal());assert a['mechanically_admitted'];assert a['independently_palindromic_blocks']==0
 assert not validate_proposal(proposal('Step on no pet.'))['mechanically_admitted']
 assert not validate_proposal(proposal('Step on no pets.1'))['mechanically_admitted']
 assert not validate_proposal(proposal('Stép on no pets.'))['mechanically_admitted']
def test_repetition_allowed_and_blocks_fail_closed():
 p=proposal('Step on no pets. Step on no pets.');p['construction_blocks']=['Step on no pets.','Step on no pets.'];a=validate_proposal(p)
 assert a['mechanically_admitted'];assert a['repetition']['sentence_repeat_fraction']==.5
 p['construction_blocks']=['other'];assert not validate_proposal(p)['mechanically_admitted']
def test_ABCBA_checked():
 p=proposal('Step on no pets.');p['method']='phrase_ABCBA';assert 'ABCBA_block_invariant' in validate_proposal(p)['errors']
 p.update(text='Eve, refer Eve.',construction_blocks=['Eve','re','f','er','Eve'],block_roles=['person','partial verb','partial verb','partial verb','person'],seams=[dict(left_block=i,right_block=i+1,grammar_relation='lexical seam') for i in range(4)])
 assert validate_proposal(p)['mechanically_admitted']
def test_blinding_dedup_and_fail_closed_reviews():
 rows,m=blind_batch([dict(text='Step on no pets.',source_id='a')],[validate_proposal(proposal())]);assert len(rows)==1;assert len(next(iter(m.values()))['occurrences'])==2;assert set(rows[0])=={'blind_id','text'}
 r=rating(rows[0]['blind_id']);assert validate_ratings([r],m)
 for bad in [[],[r,r],[dict(r,readability=True)],[dict(r,blind_id='unknown')],[dict(r,text_sha256='bad')]]:
  with pytest.raises(ValueError):validate_ratings(bad,m)
def test_length_padding_and_soft_repetition():
 rows,m=blind_batch([dict(text='Step on no pets. '*20,source_id='a')],[]);r=rating(rows[0]['blind_id']);r.update(repetition_burden=4,padding_or_loop=True)
 s=scored_rows([r],m)[0];assert s['length_bonus']==0;assert s['utility']==5.5;assert s['pareto_frontier']

def test_immutable_end_to_end_batches(tmp_path,monkeypatch):
 import json
 import experiments.phrase_api_pilot_011 as pilot
 monkeypatch.setattr(pilot,'OUT',tmp_path)
 (tmp_path/'baseline-provenance.json').write_text(json.dumps([dict(text='Live. Ma is selfless. I am evil.',source_id='baseline-control')]))
 src=tmp_path/'proposals.jsonl';src.write_text(json.dumps(proposal())+'\n'+json.dumps(dict(proposal('Step on no pet.'),proposal_id='failure'))+'\n')
 dest=pilot.proposals(src,'fixture');receipt=json.loads((dest/'receipt.json').read_text());assert receipt['attempts']==2 and receipt['accepted']==1 and receipt['failed']==1
 with pytest.raises(ValueError):pilot.proposals(src,'fixture')
 blind=[json.loads(x) for x in (dest/'blind-comparison.jsonl').read_text().splitlines()];assert len(blind)==2
 scores=tmp_path/'scores.jsonl';scores.write_text(''.join(json.dumps(rating(r['blind_id']))+'\n' for r in blind));output=pilot.ratings(scores,'fixture')
 assert json.loads((output/'receipt.json').read_text())['missing']==0
 assert json.loads((output/'comparison.json').read_text())['groups']['phrase_bridge_overhang']['unique_texts']==1
 with pytest.raises(ValueError):pilot.ratings(scores,'fixture')

def test_grades_do_not_force_repetition_veto():
 rows,m=blind_batch([dict(text='Step on no pets. Step on no pets.',source_id='repeat')],[])
 r=rating(rows[0]['blind_id']);r['repetition_burden']=4
 scored=scored_rows([r],m);assert len(scored)==1;assert scored[0]['utility']>0

def test_exact_seam_debt():
 from llm_palindrome.phrase_api_loop import mirror_debt
 assert mirror_debt('Noel, deliver','reviled Leon')['remaining_debt_letters']==0
 assert mirror_debt('Noel, deliver Anna','reviled Leon')['inner_required_suffix']=='anna'
 assert mirror_debt('Noel, deliver','Anna reviled Leon')['inner_required_prefix']=='anna'
 assert not mirror_debt('Noel, deliver','reviled Eve')['completion_possible_by_middle_only']
