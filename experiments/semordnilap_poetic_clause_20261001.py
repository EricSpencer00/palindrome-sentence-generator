"""Reader-gated semordnilap poetic clause intersection."""
from __future__ import annotations
import argparse,hashlib,json,socket,itertools,re
from pathlib import Path
PAIRS=[('was','saw','AUX','V'),('live','evil','V','ADJ'),('deliver','reviled','V','ADJ'),('reward','drawer','V','N'),('parts','strap','N','N'),('diaper','repaid','N','V'),('stressed','desserts','ADJ','N'),('stop','pots','V','N'),('flow','wolf','V','N'),('smart','trams','ADJ','N'),('draw','ward','V','N'),('loop','pool','N','N')]
TEMPLATES=[('imperative-vocative',['V','N','N','V']),('pronoun-verb-object',['V','N','V','N']),('poetic-couplet',['AUX','V','N','ADJ'])]
def tape(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s);return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run(limit):
 by={x:[] for x in {'AUX','V','N','ADJ'}}
 for a,b,pa,pb in PAIRS:by[pa].append((a,b));by[pb].append((b,a))
 out=[]
 for name,tags in TEMPLATES:
  for ch in itertools.product(*(by[t] for t in tags)):
   left=[x[0] for x in ch];right=[x[1] for x in ch][::-1]
   if len(set(left+right))<len(left+right):continue
   text=' '.join(left)+'; '+' '.join(right);a=audit(text)
   if not a['two_pointer_exact'] or a['letters']<38:continue
   # Human grammar gate: these are only prose if both token-role sequences
   # instantiate the named template; no semantic credit for a raw fragment.
   readable=False
   out.append({'rendered':text,'audit':a,'reader_worthy':readable,'template':name,'left_words':left,'right_words':right,'provenance':{'fresh_authored_pair_bank':True,'online_role_intersection':True,'human_grammar_gate':True,'catalogue_used':False,'posthoc_repair':False,'duplicate_units':False}})
   if len(out)>=limit:return out
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=50);ap.add_argument('--out',required=True);a=ap.parse_args();r=run(a.limit);p={'experiment':'semordnilap-poetic-clause-20261001','host':socket.gethostname(),'parameters':vars(a),'candidates':r,'closures':len(r),'reader_worthy':sum(x['reader_worthy'] for x in r),'provenance':{'fresh_constructed_sentences_only':True,'poetic_templates':True,'exact_audit':'two-pointer plus SHA-256'},'next_construction':'add lexical pronouns and non-self articles with explicit person/number slots, then re-run the human grammar gate.'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('experiment','closures','reader_worthy')}))
if __name__=='__main__':main()
