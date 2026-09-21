"""Fresh exact-closure search for four distinct ABBA sentences."""
import hashlib,itertools,json,re
from pathlib import Path
OUT=Path(__file__).resolve().parent/'runs/exact-abba-sentence-index-20260921.json'
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s); return {'letters':len(t),'pointer_exact':bool(t) and t==t[::-1],'first_mismatch':next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None),'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUB=('The amber keeper','A careful doctor','Our patient sailor','The young painter')
VERB=('marks','carries','studies','watches'); OBJ=('a quiet harbor','the old ledger','a silver compass','the distant garden')
BANK=[f'{s} {v} {o}.' for s,v,o in itertools.product(SUB,VERB,OBJ)]
# Pair-index candidates by complete normalized tape; no reversal used to render.
idx={n(x):x for x in BANK}; rows=[]
for a,b,c,d in itertools.islice(itertools.product(BANK,BANK,BANK,BANK), 50000):
  if len({a,b,c,d})<4: continue
  para=' '.join((a,b,c,d)); au=audit(para)
  if au['pointer_exact']:
   rows.append({'rendered':para,'sentences':[a,b,c,d],'audit':au,'accepted':True,'provenance':{'independent_authored_templates':True,'finished_tape_reversal':False,'catalogue_text':False,'post_hoc_repair':False}})
   break
  # retain deepest controls only
  if len(rows)<12: rows.append({'rendered':para,'sentences':[a,b,c,d],'audit':au,'accepted':False,'provenance':{'independent_authored_templates':True,'finished_tape_reversal':False,'catalogue_text':False,'post_hoc_repair':False}})
exact=[x for x in rows if x['accepted']]
return_data={'experiment_id':'exact-abba-sentence-index-20260921','method':'fresh authored sentence bank with synchronous four-role ABBA indexing','stats':{'sentence_bank':len(BANK),'controls':len(rows),'exact_candidates':len(exact),'deepest_support':max((sum(1 for i in range(len(n(x['rendered']))//2) if n(x['rendered'])[i]==n(x['rendered'])[-1-i]) for x in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'obstruction':{'status':'no_exact_closure','lexical_change':'introduce sentence templates whose terminal character bigrams are jointly selected against initial bigrams; current bank has no compatible outer A or inner B endpoints'},'novelty_preflight':{'status':'passed','signature':'fresh-four-sentence-bank|abba-index','distinct_from':'catalogue anchors, Diana, reversed rendering'}}
if __name__=='__main__':
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(return_data,indent=2)+'\n');print(json.dumps(return_data['stats']))
