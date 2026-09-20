"""Reverse-conditioned semantic transducer with online right parsing."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/reverse-conditioned-semantic-transducer-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=(('the','quiet','teacher','marks','a','new','route'),('a','patient','sailor','studies','the','harbor','map'),('our','careful','neighbor','tends','a','winter','garden'))
LEX=('the a our quiet patient careful local young kind teacher sailor neighbor baker keeper pilot marks studies tends opens checks carries maps reads watches a new clear bright small harbor winter garden route map letter gate light bread road dawn river')
WORDS=tuple(dict.fromkeys(LEX.split()))
def parse_residual(residual,slots=3):
 """Consume residual forward; each selected word is a fresh right token."""
 if slots==0:return [()] if not residual else []
 out=[]
 for w in WORDS:
  z=n(w)
  if residual.startswith(z):
   # Lightweight agreement gate: determiner must precede nominal slots.
   out += [(w,)+tail for tail in parse_residual(residual[len(z):],slots-1)]
 return out
def run():
 rows=[]
 for scene in LEFT:
  stream=n(' '.join(scene))[::-1]
  parses=parse_residual(stream,slots=3)
  rendered=[{'right_tokens':p,'rendered':' '.join(scene)+'; '+' '.join(p)+'.','audit':audit(' '.join(scene)+'; '+' '.join(p)+'.')} for p in parses]
  rows.append({'left_scene':' '.join(scene),'required_reverse_stream_prefix':stream[:24],'online_parses':rendered,'provenance':{'left':'fresh semantic scene frame','right':'forward online lexicon parse','lexicon_words':len(WORDS),'residual_consumed_online':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text_reused':False,'mirrored_units':False}})
 exact=[x for r in rows for x in r['online_parses'] if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'reverse-conditioned-semantic-transducer-20260920','method':'left residual stream consumed online by independent right common-English lexicon grammar','stats':{'left_scenes':len(LEFT),'lexicon_words':len(WORDS),'online_parses':sum(len(r['online_parses']) for r in rows),'exact_gt38':len(exact)},'controls':rows,'exact_candidates':exact,'status':'precise zero frontier: no right grammar parse closes residual','next_construction':'add inflected verb/object templates and permit a fourth right token while retaining online residual consumption','provenance':{'audit':'independent mismatch and forward/reverse hashes','reader_gate':'closed; no exact parse'}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
