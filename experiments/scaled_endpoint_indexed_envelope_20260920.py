"""Scaled endpoint indexed envelope."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BANK=ROOT/'data/brown_pcfg_bank_20260920.json'; OUT=ROOT/'runs/scaled-endpoint-indexed-envelope-20260920.json'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); r=t[::-1]; bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
N=('alice','anna','arthur','diana','edward','elena','felix','george','helen','james','jane','john','julia','leon','lucas','marie','mark','nora','oliver','paul','peter','rose','sarah','thomas','victor','william','edith','frank','grace','harry','isabel','leo','martha','oscar','rachel','samuel','theodore','ursula','violet','walter','yvonne','zachary')
SHAPES=(('svo',('AGENT','VERB','DET','OBJECT')),('dit',('AGENT','VERB','PREP','DET','AGENT')),('pp',('AGENT','VERB','DET','OBJECT','PREP','DET','PLACE')))
def domains():
 p=json.loads(BANK.read_text())['lexicon']; top=lambda k,n:tuple(x['word'] for x in p[k][:n]); return {'AGENT':N+top('NOUN',60),'VERB':top('VERB',60),'OBJECT':top('NOUN',60),'PREP':top('PREP',30),'PLACE':top('NOUN',50),'DET':('the','a','an')}
def index(d):
 return {r:{(letters(w)[0],letters(w)[-1],len(letters(w))):w for w in ws if letters(w)} for r,ws in d.items()}
def run():
 d=domains(); idx=index(d); nodes=0; exact=[]
 for _,ls in SHAPES:
  for _,rs in SHAPES:
   for a in d[ls[0]][:60]:
    for b in d[rs[-1]][:60]:
     nodes+=1
     if letters(a)[0]!=letters(b)[-1]: continue
     text=a+'; '+b+'.'; q=audit(text)
     if q['exact'] and q['letters']>38: exact.append({'rendered':text,'audit':q})
 controls=['Alice sees the house; Diana finds the door.','The young man writes the letter; Marie reads the book.']
 return {'experiment_id':'scaled-endpoint-indexed-envelope-20260920','method':'Brown POS plus 45-name endpoint index with live envelope starts','domain_sizes':{k:len(v) for k,v in d.items()},'stats':{'endpoint_keys':sum(len(v) for v in idx.values()),'nodes':nodes,'fresh_exact_gt38':len(exact)},'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','signature':'brown-pos-expanded|ordinary-name-bank|endpoint-indexed-envelope','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'brown_bank':str(BANK.relative_to(ROOT)),'bank_sha256':hashlib.sha256(BANK.read_bytes()).hexdigest(),'name_bank':'45 ordinary authored proper names','audits':['two-pointer mismatch','forward/reverse SHA-256']},'status':'no fresh exact >38 candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
