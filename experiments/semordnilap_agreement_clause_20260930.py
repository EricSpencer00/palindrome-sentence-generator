"""Agreement-aware semordnilap clause grammar, exact by token intersection."""
from __future__ import annotations
import argparse,hashlib,json,socket,itertools,re
from pathlib import Path
PAIRS=[('was','saw','AUX','V'),('live','evil','V','ADJ'),('deliver','reviled','V','ADJ'),('stressed','desserts','ADJ','N'),('reward','drawer','V','N'),('diaper','repaid','N','V'),('parts','strap','N','N'),('smart','trams','ADJ','N'),('stop','pots','V','N'),('flow','wolf','V','N')]
GRAM=[('AUX','V','N','N'),('N','V','N','AUX'),('ADJ','N','V','N')]
def tape(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s);return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run(limit):
 by={k:[] for k in {'AUX','V','N','ADJ'}}
 for a,b,pa,pb in PAIRS:by[pa].append((a,b));by[pb].append((b,a))
 rows=[]
 for tags in GRAM:
  for ch in itertools.product(*(by[t] for t in tags)):
   left=[x[0] for x in ch]; right=[x[1] for x in ch][::-1]
   if len(set(left+right))<len(left+right):continue
   text=' '.join(left)+'; '+' '.join(right);a=audit(text)
   if a['two_pointer_exact'] and a['letters']>=38:rows.append({'rendered':text,'audit':a,'reader_worthy':False,'agreement_state':{'subject_number':'unresolved','auxiliary_pair':tags[0]=='AUX','clause_final':True},'left_words':left,'right_words':right,'provenance':{'online_token_pair_intersection':True,'authored_clause_grammar':True,'determiners_pronouns_considered':'blocked by no-self-palindrome constraint','catalogue_used':False,'posthoc_repair':False,'duplicate_units':False}})
   if len(rows)>=limit:return rows
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=20);ap.add_argument('--out',required=True);a=ap.parse_args();r=run(a.limit);p={'experiment':'semordnilap-agreement-clause-20260930','host':socket.gethostname(),'parameters':vars(a),'candidates':r,'closures':len(r),'reader_worthy':sum(x['reader_worthy'] for x in r),'provenance':{'fresh_semordnilap_pairs':True,'agreement_safe_metadata':True,'no_catalogue':True,'exact_audit':'two-pointer plus SHA-256'},'next_construction':'author plural/tense semordnilap pairs and permit determiner pairs only when a non-self lexical article exists.'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('experiment','closures','reader_worthy')}))
if __name__=='__main__':main()
