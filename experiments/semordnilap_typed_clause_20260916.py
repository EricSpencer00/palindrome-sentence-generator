"""Typed clause lane using a hand-audited common reversible lexicon."""
from pathlib import Path
import hashlib,json,sys;sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.wordpair_graph import tape
from llm_palindrome.admission import mechanical_admission_checks
PAIRS=[('stressed','desserts'),('diaper','repaid'),('drawer','reward'),('deliver','reviled'),('gateman','nametag')]
def audit(s):
 t=tape(s);i,j=0,len(t)-1
 while i<len(t) and t[i]==t[j]:i+=1;j-=1
 return {'exact':i==len(t),'matched':i,'mismatch':None if i==len(t) else t[i:i+12]}
def main():
 witness='Stressed desserts.'; near='The tired baker served stressed desserts, then repaired a drawer while a quiet traveler delivered bread to the old river.'
 rows=[]
 for kind,s in [('witness',witness),('near_miss',near)]:
  t=tape(s); rows.append({'kind':kind,'text':s,'letters':len(t),'exact':audit(s)['exact'],'audit':audit(s),'forward_hash':hashlib.sha256(t.encode()).hexdigest(),'reverse_hash':hashlib.sha256(t[::-1].encode()).hexdigest(),'hash_equal':t==t[::-1],'mechanical_checks':mechanical_admission_checks(s,min_letters=16,max_letters=1000),'admitted':False})
 out={'method':'semordnilap_typed_clause_v1','reversible_pairs':PAIRS,'candidates':rows,'provenance':{'lexicon':'five common reversible word pairs, manually POS-checked','grammar':'typed declarative clauses; subject-verb-object and adjunct slots','distinct_content':True},'novelty_preflight':{'digests':[r['forward_hash'] for r in rows],'action':'compare with prior run digests'},'next_repair':'Add Brown frequency-ranked reversible pairs by POS, then solve clause slots jointly so every selected reversal remains grammatical at both boundaries.'}
 p=Path(__file__).parents[1]/'runs/semordnilap-typed-clause-2026-09-16.json';p.write_text(json.dumps(out,indent=2)+'\n');print(str(p))
if __name__=='__main__':main()
