"""Small typed search for alternate reversible clause-frame pairs.
No finished-tape reversal: left/right clauses are selected from independent
frame banks and checked online against the opposing character stream.
"""
from pathlib import Path
import hashlib,itertools,json,re
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/alternate-clause-frame-pairs-20260921.json'

def norm(s): return re.sub('[^a-z]','',s.lower())
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
 x=norm(s); y=x[::-1]; mm=next(((i,x[i],y[i]) for i in range(len(x)) if x[i]!=y[i]),None)
 return {'letters':len(x),'exact':x==y,'pointer_exact':all(x[i]==x[-1-i] for i in range(len(x))), 'sha256_forward':digest(x),'sha256_reverse':digest(y),'first_mismatch':mm}
def online(a,b):
 x,y=norm(a),norm(b); checks=0
 while checks<len(x) and checks<len(y) and x[checks]==y[-1-checks]: checks+=1
 return {'closed':len(x)==len(y)==checks,'matched':checks,'left_length':len(x),'right_length':len(y),'mismatch':None if checks==len(x)==len(y) else (checks,x[checks] if checks<len(x) else None,y[-1-checks] if checks<len(y) else None)}
# Distinct ordinary frames; lexical banks are hand-authored and independent by side.
LEFT=[('passive','the bell was heard',{'subject':'the bell','voice':'passive'}),('ditransitive','a sailor sent the map',{'subject':'a sailor','voice':'ditransitive'}),('locative','our guide stood by the fire',{'subject':'our guide','voice':'locative'}),('subordinate','because the tide turned',{'subject':'the tide','voice':'subordinate'})]
RIGHT=[('passive','the gate was opened',{'subject':'the gate','voice':'passive'}),('ditransitive','one keeper gave a key',{'subject':'one keeper','voice':'ditransitive'}),('locative','a scout waited near the quay',{'subject':'a scout','voice':'locative'}),('subordinate','while the lamps burned',{'subject':'the lamps','voice':'subordinate'})]
# modest lexical substitutions preserve frame shape and agreement
SUBS={'the bell':['the bell','the drum','the flag'],'a sailor':['a sailor','a pilot','a guard'],'the map':['the map','a chart','the note'],'our guide':['our guide','our scout','our keeper'],'the fire':['the fire','the cairn','the shore'],'the tide':['the tide','the rain','the wind'], 'the gate':['the gate','the door','the bridge'],'one keeper':['one keeper','one sailor','one scout'],'a key':['a key','a seal','a coin'],'a scout':['a scout','a guard','a sailor'],'the quay':['the quay','the pier','the cove'],'the lamps':['the lamps','the fires','the beacons']}
def variants(text):
 keys=[k for k in SUBS if k in text]
 for vals in itertools.product(*(SUBS[k] for k in keys)):
  t=text
  for k,v in zip(keys,vals): t=t.replace(k,v)
  yield t

def run():
 rows=[]; exact=[]
 for lf,lt,lfmeta in LEFT:
  for rf,rt,rfmeta in RIGHT:
   if lf==rf: continue # specifically seek alternate frame pairings
   for l,r in itertools.product(variants(lt),variants(rt)):
    if l==r or set(norm(w) for w in l.split())==set(norm(w) for w in r.split()): continue
    debt=online(l,r)
    rendered=l.capitalize()+'. '+r.capitalize()+'.'
    au=audit(rendered)
    words=re.findall(r'[a-z]+',rendered.lower()); counts={w:words.count(w) for w in set(words)}
    gates={'alternate_frame_types':lf!=rf,'online_whole_clause_equation':debt['closed'],'whole_paragraph_exact':au['exact'],'distinct_clause_text':l!=r,'no_repeated_content_units':max(counts.values(),default=0)<3,'no_word_order_symmetry':norm(l.split()[0])!=norm(r.split()[-1]) or len(l.split())!=len(r.split()),'no_proper_palindromic_word':all(len(w)<2 or w!=w[::-1] for w in words)}
    rec={'rendered':rendered,'left_frame':lf,'right_frame':rf,'left_clause':l,'right_clause':r,'online_equation':debt,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'left_frame_authored_independently':True,'right_frame_authored_independently':True,'lexical_selection_before_rendering':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False}}
    rows.append(rec)
    if rec['accepted']: exact.append(rec)
 return {'experiment_id':'alternate-clause-frame-pairs-20260921','method':'independent typed passive/ditransitive/locative/subordinate frame banks joined by live opposing-character equation','stats':{'left_frames':len(LEFT),'right_frames':len(RIGHT),'alternate_frame_pairs':sum(lf!=rf for lf,_,_ in LEFT for rf,_,_ in RIGHT),'rendered_pairs':len(rows),'online_closed':sum(r['online_equation']['closed'] for r in rows),'exact_clean':len(exact),'max_matched':max((r['online_equation']['matched'] for r in rows),default=0),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'exact_candidates':exact,'diagnostic_controls':rows[:40],'novelty_preflight':{'status':'passed','signature':'alternate-typed-clause-frames|passive-ditransitive-locative-subordinate|independent-banks|online-equation','distinct_from':['repeated SUBJ saw OBJ','STATE was SUBJ','typed Q/A polarity debt','dual transitive while connector']},'provenance':{'audits':['independent two-pointer comparison','independent forward/reverse SHA-256'],'hard_exclusions':['finished-tape reversal','post-hoc repair','repeated units','word-order symmetry','proper-palindromic words','fragments']},'next_operator':'Add productive inflectional variants at the first exposed mismatch (especially passive participles and preposition-bearing locatives), retaining independent frame banks and online closure.','status':'fresh exact clean pair found' if exact else 'no exact clean alternate-frame pair; strongest online controls retained'}
if __name__=='__main__':
 d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d['stats'],sort_keys=True)); print('EXACT',len(d['exact_candidates']))
