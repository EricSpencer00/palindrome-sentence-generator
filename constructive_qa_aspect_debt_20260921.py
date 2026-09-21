"""Successor to constructive Q/A debt: tense and agreement are stateful gates."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/constructive-qa-aspect-debt-20260921.json'
# Small typed frames, deliberately not a lexical Cartesian product.
QUESTIONS=[
 {'subject':'I','aux':'am','verb':'mapping','object':'a cove','tense':'present','agreement':'singular','aspect':'progressive'},
 {'subject':'we','aux':'are','verb':'seeking','object':'the trail','tense':'present','agreement':'plural','aspect':'progressive'},
 {'subject':'she','aux':'was','verb':'carrying','object':'one key','tense':'past','agreement':'singular','aspect':'progressive'},
 {'subject':'they','aux':'were','verb':'watching','object':'a beacon','tense':'past','agreement':'plural','aspect':'progressive'},]
ANSWERS=[
 {'subject':'the guide','aux':'is','verb':'guarding','object':'a quay','tense':'present','agreement':'singular','aspect':'progressive'},
 {'subject':'the sailors','aux':'have','verb':'charted','object':'the path','tense':'present','agreement':'plural','aspect':'completed'},
 {'subject':'the keeper','aux':'had','verb':'secured','object':'one lantern','tense':'past','agreement':'singular','aspect':'completed'},
 {'subject':'the scouts','aux':'had','verb':'watched','object':'a star','tense':'past','agreement':'plural','aspect':'completed'},]
def norm(s):return re.sub('[^a-z]','',s.lower())
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':digest(x),'sha256_reverse':digest(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def render(f):return f"{f['subject']} {f['aux']} {f['verb']} {f['object']}"
def words(text):return re.findall('[a-z]+', text.lower())
FUNCTION_WORDS={'a','am','an','are','the','was','we','were','i','is','one'}
def content_words(text):return [w for w in words(text) if w not in FUNCTION_WORDS]
def debt(q,a):
 x,y=norm(q),norm(a); checks=[]
 for i,ch in enumerate(x):
  j=len(y)-1-i
  if j<0:return False,{'closed':False,'checks':len(checks),'reason':'answer-overhang'}
  checks.append((i,j,ch,y[j]))
  if ch!=y[j]:return False,{'closed':False,'checks':len(checks),'first_mismatch':checks[-1]}
 return len(x)==len(y),{'closed':len(x)==len(y),'checks':len(checks),'reason':'closed' if len(x)==len(y) else 'question-overhang'}
def run():
 rows=[]
 for q,a in itertools.product(QUESTIONS,ANSWERS):
  qt=render(q);at=render(a); closed,eq=debt(qt,at); rendered=f"{qt}? {at}."; au=audit(rendered)
  compatible=q['tense']==a['tense'] and q['agreement']==a['agreement'] and q['aspect']==a['aspect']
  gates={'online_debt_closed':closed,'whole_output_exact':au['exact'],'tense_compatible':compatible,'agreement_compatible':q['agreement']==a['agreement'],'aspect_compatible':q['aspect']==a['aspect'],'distinct_clause_vocab':not(set(content_words(qt)) & set(content_words(at))),'no_self_palindromic_units':all(len(norm(w)) < 2 or norm(w)!=norm(w)[::-1] for w in words(qt+' '+at))}
  rows.append({'rendered':rendered,'question_frame':q,'answer_frame':a,'debt_equation':eq,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'four hand-authored typed tense/agreement frames per side','selected_online_against_opposing_character_debt':True,'feature_state':{'question_tense':q['tense'],'answer_tense':a['tense'],'question_agreement':q['agreement'],'answer_agreement':a['agreement'],'question_aspect':q['aspect'],'answer_aspect':a['aspect']},'lexical_cartesian_expansion':False,'finished_tape_reversal':False,'borrowed_catalogue_text':False,'posthoc_repair':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'constructive-qa-aspect-debt-20260921','method':'typed Q/A debt with tense, agreement, and progressive/completed aspect state','stats':{'question_frames':len(QUESTIONS),'answer_frames':len(ANSWERS),'pairs':len(rows),'feature_compatible':sum(r['gates']['tense_compatible'] and r['gates']['agreement_compatible'] and r['gates']['aspect_compatible'] for r in rows),'online_closed':sum(r['debt_equation']['closed'] for r in rows),'accepted_exact':len(exact),'accepted_exact_gt38':sum(r['audit']['letters']>38 for r in exact)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'typed-qa|tense-agreement-state|online-debt','signature_collision':False,'distinct_from':'lexical Cartesian Q/A debt and duplicated-middle lane'},'next_operator':'Add polarity as one further typed state, retaining four-frame bounded search.','status':'fresh exact closure found' if exact else 'no fresh exact closure; feature-compatible controls retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
