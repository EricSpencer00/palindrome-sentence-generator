"""Headless phrase proposal gate and fail-closed blinded pilot integration.
No model calls, inferred ratings, or grammar/readability certification.
"""
import hashlib,json,re
from collections import Counter
from .admission import normalize_letters,ALLOWED_RENDERING,tokenize
METHODS=('phrase_ABCBA','phrase_bridge_overhang','non_ABCBA_role_scene')
REQUIRED=('proposal_id','method','text','construction_blocks','block_roles','seams','intended_scene','copied_sources')
def repetition_features(text):
 words=tokenize(text);counts=Counter(words)
 try:sentences=[normalize_letters(s) for s in re.split(r'[.!?]+',text) if normalize_letters(s)]
 except ValueError:return dict(invalid_alphabet=True,token_repeat_fraction=None,trigram_repeat_fraction=None,sentence_repeat_fraction=None,tokens=len(words),sentences=None)
 grams=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
 return dict(token_repeat_fraction=(len(words)-len(counts))/max(1,len(words)),trigram_repeat_fraction=(len(grams)-len(set(grams)))/max(1,len(grams)),sentence_repeat_fraction=(len(sentences)-len(set(sentences)))/max(1,len(sentences)),tokens=len(words),sentences=len(sentences))
def validate_proposal(p):
 errors=[]
 for k in REQUIRED:
  if k not in p:errors.append('missing:'+k)
 text=p.get('text','');tape=''
 if not isinstance(text,str):errors.append('text_type');text=''
 if not ALLOWED_RENDERING.fullmatch(text):errors.append('unsupported_rendering')
 try:tape=normalize_letters(text)
 except ValueError:errors.append('unsupported_alphabet')
 if not tape or tape!=tape[::-1]:errors.append('not_exact')
 if p.get('method') not in METHODS:errors.append('unknown_method')
 if not isinstance(p.get('proposal_id'),str) or not p.get('proposal_id'):errors.append('invalid_proposal_id')
 blocks=p.get('construction_blocks',[]);roles=p.get('block_roles',[]);seams=p.get('seams',[])
 if not isinstance(blocks,list) or not blocks or not all(isinstance(b,str) and b for b in blocks):errors.append('invalid_blocks');blocks=[]
 bt=[]
 try:bt=[normalize_letters(b) for b in blocks]
 except ValueError:errors.append('unsupported_block_alphabet')
 if ''.join(bt)!=tape:errors.append('block_tape_mismatch')
 if not isinstance(roles,list) or len(roles)!=len(blocks) or not all(isinstance(x,str) and x for x in roles):errors.append('invalid_block_roles')
 if not isinstance(seams,list) or len(seams)!=max(0,len(blocks)-1):errors.append('invalid_seam_count')
 else:
  for i,s in enumerate(seams):
   if not isinstance(s,dict) or s.get('left_block')!=i or s.get('right_block')!=i+1 or not isinstance(s.get('grammar_relation'),str) or not s['grammar_relation']:errors.append('invalid_seam:'+str(i))
 if not isinstance(p.get('intended_scene'),str) or not p.get('intended_scene'):errors.append('invalid_scene')
 if not isinstance(p.get('copied_sources'),list):errors.append('invalid_sources')
 mirror=None
 if p.get('method')=='phrase_ABCBA':
  mirror=len(bt)==5 and bt[0]==bt[4][::-1] and bt[1]==bt[3][::-1] and bt[2]==bt[2][::-1]
  if not mirror:errors.append('ABCBA_block_invariant')
 # These are mechanical disclosures; no phrase role label licenses grammar.
 independently_palindromic=sum(bool(b) and b==b[::-1] for b in bt)
 return dict(proposal_id=p.get('proposal_id'),method=p.get('method'),text=text,tape=tape,letters=len(tape),exact=bool(tape) and tape==tape[::-1],mechanically_admitted=not errors,errors=errors,block_count=len(blocks),block_ABCBA_verified=mirror,independently_palindromic_blocks=independently_palindromic,all_blocks_independently_palindromic=bool(bt) and independently_palindromic==len(bt),repetition=repetition_features(text),proposal=p,grammar_verified=False,human_label=None)
def blind_batch(baseline,admitted,seed='phrase-api-loop-011'):
 rows=[];mapping={};seen={}
 for source,r in [('baseline',x) for x in baseline]+[('proposal',x) for x in admitted if x['mechanically_admitted']]:
  text=r['text'];tape=normalize_letters(text);key=hashlib.sha256(text.encode()).hexdigest()
  if key in seen:
   mapping[seen[key]]['occurrences'].append(dict(source=source,source_id=r.get('proposal_id',r.get('source_id',r.get('blind_id')))));continue
  bid='p'+hashlib.sha256((seed+'|'+key).encode()).hexdigest()[:16];seen[key]=bid
  rows.append(dict(blind_id=bid,text=text));mapping[bid]=dict(text=text,text_sha256=key,tape_sha256=hashlib.sha256(tape.encode()).hexdigest(),letters=len(tape),occurrences=[dict(source=source,source_id=r.get('proposal_id',r.get('source_id',r.get('blind_id'))))],repetition=repetition_features(text))
 rows.sort(key=lambda r:hashlib.sha256((seed+'|shuffle|'+r['blind_id']).encode()).hexdigest())
 return rows,mapping

def validate_ratings(rows,mapping,expected_ids=None):
 expected=set(mapping) if expected_ids is None else set(expected_ids)
 if not expected<=set(mapping):raise ValueError('unknown expected batch IDs')
 ids=[r.get('blind_id') for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError('duplicate rating IDs')
 if set(ids)!=expected:raise ValueError('missing or unknown rating IDs')
 for r in rows:
  for k in ['grammar','readability','coherence','repetition_burden']:
   if type(r.get(k)) is not int or not 0<=r[k]<=4:raise ValueError('invalid scale:'+k)
  for k in ['meaningful_progression','padding_or_loop']:
   if type(r.get(k)) is not bool:raise ValueError('invalid boolean:'+k)
  if not isinstance(r.get('rationale'),str) or not r['rationale'].strip():raise ValueError('missing rationale')
  if 'text_sha256' in r and r['text_sha256']!=mapping[r['blind_id']]['text_sha256']:raise ValueError('text binding mismatch')
 return True

def scored_rows(ratings,mapping):
 validate_ratings(ratings,mapping)
 out=[]
 for r in ratings:
  m=mapping[r['blind_id']];length_bonus=0 if r['padding_or_loop'] else .5*min(m['letters'],180)/180
  utility=r['readability']+.5*r['coherence']+.5*r['grammar']+length_bonus-.5*r['repetition_burden']
  out.append(dict(**r,**m,utility=utility,length_bonus=length_bonus,quality_source='new blinded model pilot; not human'))
 # Repetition never hard-rejects an otherwise admitted exact proposal.
 def dominates(a,b):
  av=(a['readability'],a['coherence'],a['grammar'],0 if a['padding_or_loop'] else min(a['letters'],180),-a['repetition_burden'])
  bv=(b['readability'],b['coherence'],b['grammar'],0 if b['padding_or_loop'] else min(b['letters'],180),-b['repetition_burden'])
  return all(x>=y for x,y in zip(av,bv)) and any(x>y for x,y in zip(av,bv))
 for r in out:r['pareto_frontier']=not any(dominates(a,r) for a in out if a is not r)
 return sorted(out,key=lambda r:(-r['utility'],r['blind_id']))

def mirror_debt(left,right):
 """Exact outside-in seam constraint for a missing middle, no grammar claim."""
 l=normalize_letters(left);rr=normalize_letters(right)[::-1];k=min(len(l),len(rr))
 mismatches=[i for i in range(k) if l[i]!=rr[i]]
 if mismatches:return dict(compatible=False,first_mismatch=mismatches[0],left_char=l[mismatches[0]],mirrored_right_char=rr[mismatches[0]],completion_possible_by_middle_only=False)
 if len(l)>len(rr):return dict(compatible=True,matched_outer_letters=k,inner_required_suffix=l[k:][::-1],inner_required_prefix='',remaining_debt_letters=len(l)-k,grammar_verified=False)
 return dict(compatible=True,matched_outer_letters=k,inner_required_prefix=rr[k:],inner_required_suffix='',remaining_debt_letters=len(rr)-k,grammar_verified=False)
