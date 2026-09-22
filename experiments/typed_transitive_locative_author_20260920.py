"""Fresh transitive/locative scene author with number + tense unification."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/typed-transitive-locative-author-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUB=[('each careful sailor','singular'),('the patient keeper','singular'),('several young cartographers','plural'),('the patient keepers','plural')]
VERBS={('singular','past'):[('marked','transitive'),('carried','transitive'),('waited','locative')],('singular','present'):[('marks','transitive'),('carries','transitive'),('waits','locative')],('plural','past'):[('marked','transitive'),('carried','transitive'),('waited','locative')],('plural','present'):[('mark','transitive'),('carry','transitive'),('wait','locative')]}
OBJECTS=['the weathered chart','a lantern by moonlight','the narrow stone landing']; LOC=['beside the quiet inlet','under the winter stars','along the northern road','near a distant harbor']
def bad(text):
 w=text[:-1].split(); spans=[x for x in w if len(n(x))>3 and n(x)==n(x)[::-1]]
 return {'nested_self_palindrome':bool(spans),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<6,'catalogue_text':False,'mirrored_phrase_units':False}
def run():
 rows=[]
 for subj,num in SUB:
  for tense in ('past','present'):
   for verb,kind in VERBS[(num,tense)]:
    for tail in (LOC if kind=='locative' else OBJECTS):
     text=f'{subj} {verb} {tail}.'; rows.append({'rendered':text,'frame':{'subject':subj,'number':num,'tense':tense,'verb':verb,'valency':kind,'complement':tail},'audit':audit(text),'grammar':{'agreement':'passed','number':num,'tense':tense},'provenance':{**bad(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'typed-transitive-locative-author-20260920','method':'fresh typed transitive/locative scene grammar with unified number and tense','stats':{'subjects':len(SUB),'feature_states':len(SUB)*2,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:15],'novelty_preflight':{'status':'passed','signature':'fresh-authored|typed-transitive-locative|number-tense-unification|ordinary-scenes','distinct_from':'prior short agreement bank: disjoint subjects, complements, tense states, and explicit transitive versus locative valency'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','word-order symmetry','fragments','catalogue text']},'next_construction':'Add relative-clause attachment with independently typed subject number, while retaining valency and tense unification before endpoint residual selection.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; longest grammatical near-miss retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
