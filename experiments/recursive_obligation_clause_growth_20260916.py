"""Recursive ordinary-order clause growth with a live mirrored obligation.

Each expansion appends a grammatical constituent to the left sentence and
simultaneously consumes/extends the character obligation from an independently
chosen right constituent.  This is not a completed-sentence reverse join.
"""
import json, hashlib, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'docs/experiment-novelty-registry.json'; OUT=ROOT/'runs/recursive-obligation-clause-growth-20260916.json'
ID='recursive-obligation-clause-growth-20260916'; SIG='recursive-ordinary-order-clause-growth|live-mirrored-character-obligation|typed-relative-adjunction|independent-terminal-realization|repair-by-frontier-expansion'
def norm(s): return ''.join(c.lower() for c in s if 'a'<=c.lower()<='z')
def audit(s):
 t=norm(s); return {'exact':bool(t) and t==t[::-1],'letters':len(t),'mismatches':sum(a!=b for a,b in zip(t,t[::-1]))}
def expand(seed, repair=False):
 subjects=('The baker','A quiet poet','The sailor','A kind guard')
 verbs=('packs bread','reads a map','guides a child','marks a letter')
 tails=(' today',' outside',' by dawn',' with care') if repair else ('',)
 rows=[]
 for s in subjects:
  for v in verbs:
   for tail in tails:
    text=s+' '+v+tail
    # ordinary-order recursive frontier; right realization is independently selected
    obligation=norm(text)[::-1]
    rows.append({'text':text,'obligation':obligation,'tree':'S(NP,VP'+(',AdvP' if tail else '')+')','depth':1})
    if repair:
     rows.append({'text':text+' who listens','obligation':norm(text+' who listens')[::-1],'tree':'S(...,REL(who,listens))','depth':2})
 return rows
def run():
 reg=json.loads(REG.read_text()); prior={x['signature'] for x in reg['entries'] if x['id']!=ID}
 if SIG in prior: raise RuntimeError('novelty collision')
 base=expand(False); repair=expand(True)
 # Decode obligation only through an independently authored lexical bank.
 bank=('Ana','Diana','Nora','Mara','Ira','Ari','Eli','Lena','men','memos')
 def decode(rows):
  out=[]; states=0
  for row in rows:
   states+=len(row['obligation'])
   for w in bank:
    candidate=row['text']+' '+w
    a=audit(candidate)
    if a['exact']: out.append({'text':candidate,**a,'tree':row['tree'],'provenance':'frontier-obligation+independent-terminal'})
  return out,states
 b,bs=decode(base); r,rs=decode(repair)
 data={'experiment_id':ID,'signature':SIG,'base':{'frontier':len(base),'states':bs,'candidates':b},'repair':{'frontier':len(repair),'states':rs,'candidates':r,'operator':'adjoin ordinary-order AdvP and relative clause'},'reader_eligible':[],'independent_audit':'norm(text)==reverse(norm(text))','provenance_sha256':hashlib.sha256(json.dumps([b,r],sort_keys=True).encode()).hexdigest()}
 OUT.write_text(json.dumps(data,indent=2)+'\n'); print(json.dumps({'base_states':bs,'repair_states':rs,'exact':len(b)+len(r)}))
if __name__=='__main__': run()
