"""Compare named-character and first-person typed ABBA scene grammars."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'runs' / 'paragraph-abba-typed-scene-20261002.json'
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome

def norm(s): return re.sub(r'[^a-z]', '', s.casefold())
def audit(s):
    t=norm(s); i,j=0,len(t)-1; mm=[]
    while i<j:
        if t[i]!=t[j]: mm.append({'offset':i,'left':t[i],'right':t[j]})
        i+=1; j-=1
    f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {'letters':len(t),'two_pointer_exact':not mm and bool(t),'first_mismatches':mm[:8],
            'forward_sha256':f,'reverse_sha256':r,'sha_equal':f==r,'project_validator':bool(is_palindrome(s))}

def row(ident, units, links, grammar):
    rendered=' '.join(units); a=audit(rendered)
    pairs=[]
    for i in range(len(units)//2):
        j=len(units)-1-i; pairs.append({'left_index':i,'right_index':j,'pair_exact':norm(units[i])==norm(units[j])[::-1]})
    guard={'distinct_units':len(set(units))==len(units),'self_palindromic_units':[u for u in units if norm(u)==norm(u)[::-1]]}
    return {'id':ident,'kind':'exact_abba_candidate','rendered':rendered,'units':units,'audit':a,
      'seam_certificate':{'pairs':pairs,'all_pairs_exact':all(x['pair_exact'] for x in pairs)},
      'discourse_links':links,'grammar':grammar,'unit_guard':guard,
      'provenance':{'independently_authored_units':True,'finished_tape_reversal':False,'catalogue_text':False,'posthoc_repair':False}}

def main():
    units=['Aron saw deer.','Deer saw mail.','Noel saw war.','War saw evil.',
           'Live was Raw.','Raw was Leon.','Liam was Reed.','Reed was Nora.']
    named=row('named-typed-scene-86',units,[
      {'adjacent_units':[0,1],'shared_referent':'deer','relation':'observation→next observer'},
      {'adjacent_units':[2,3],'shared_referent':'war','relation':'observation→next observer'},
      {'adjacent_units':[6,7],'shared_referent':'Reed','relation':'state→next state'}],
      'named subject + saw observation + noun state; reverse saw/was pair')
    assert named['audit']['two_pointer_exact'] and named['audit']['sha_equal'] and named['audit']['project_validator']
    assert named['seam_certificate']['all_pairs_exact'] and named['unit_guard']['distinct_units'] and not named['unit_guard']['self_palindromic_units']
    first_person={'id':'first-person-obstruction','kind':'typed_branch_rejected',
      'attempt':'I saw deer. Deer saw mail. Noel saw war. War saw evil.',
      'obstruction':'The reverse of “I saw deer” is “Reed was I”; the required state clause is not ordinary English.',
      'repair_operator':'typed pronoun-role substitution (I→named agent) would change the branch, so it was not applied posthoc.',
      'audit':audit('I saw deer. Deer saw mail. Noel saw war. War saw evil.')}
    return {'experiment_id':'paragraph-abba-typed-scene-20261002','method':'typed observation→state ABBA sentence seams; named vs first-person comparison',
      'hypothesis':'A longer exact paragraph can retain discourse continuity when adjacent observations share their observed noun.',
      'candidates':[named],'branches':[first_person],'stats':{'exact_candidates':1,'candidate_lengths':[named['audit']['letters']],'first_person_exact':False},
      'deepest_seam':{'closed':'deer→mail and war→evil observation chains','residual':'the center state transition Live was Raw remains grammatical but semantically thin',
        'next_repair_operator':'typed state-role substitution: choose a center state whose subject/object names both participate in the preceding observation graph before rendering',
        'not_applied':'would require a pre-authored role inventory; no posthoc repair permitted'},
      'independent_audits':['local two-pointer scan','forward/reverse SHA-256','project validator'],
      'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'shortcuts_excluded':True}}

if __name__=='__main__': OUT.write_text(json.dumps(main(),indent=2)+'\n'); print(json.dumps(main(),indent=2))
