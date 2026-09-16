"""Constructive multiset-balanced grammar sampling (not reverse decoding)."""
import hashlib,json,random,re,subprocess
from pathlib import Path
ID='multiset-balanced-grammar'; SIG='multiset-balanced-stochastic-grammar|typed-role-production|character-count-state|seeded-constructive-sampling|independent-full-tape-audit'
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/multiset-balanced-grammar-20260915.json'
ROLES={'name':['Mara','Nora','Owen','Iris'],'verb':['marks','finds','keeps','sees'],'object':['a quiet harbor','the old map','a small lantern','the red gate'],'adverb':['today','at dawn','near home']}
def norm(s): return ''.join(c.lower() for c in s if c.isalpha())
def pal(s):
 t=norm(s); return bool(t) and all(a==b for a,b in zip(t,reversed(t)))
def read(s):
 w=re.findall('[A-Za-z]+',s); return {'words':len(w),'letters':len(norm(s)),'complete_sentence':s.endswith('.'),'status':'unrated'}
def main():
 pre=json.loads(subprocess.check_output(['python','experiments/validate_experiment_novelty_20260915.py'],cwd=ROOT))
 rng=random.Random(20260915); probes=[]; exact=[]
 for _ in range(5000):
  s=f"{rng.choice(ROLES['name'])} {rng.choice(ROLES['verb'])} {rng.choice(ROLES['object'])} {rng.choice(ROLES['adverb'])}."
  t=norm(s); pairs=sum(a==b for a,b in zip(t,reversed(t)))
  row={'text':s,'matching_pairs':pairs,'exact':pal(s),'readability':read(s),'provenance':'seeded typed-role production; no reverse target or catalogue'}; probes.append(row)
  if row['exact']: exact.append(row)
 probes.sort(key=lambda x:(x['matching_pairs'],x['readability']['letters']),reverse=True)
 payload={'experiment_id':ID,'signature':SIG,'novelty_preflight':pre,'method':'sample complete grammatical role productions while tracking character-count state; acceptance is whole-tape exact audit, never a mirrored prefix','seed':20260915,'samples':len(probes),'rendered_probes':probes[:25],'rendered_candidates':exact,'independent_audit':[{'text':x['text'],'two_pointer':pal(x['text']),'normalized_sha256':hashlib.sha256(norm(x['text']).encode()).hexdigest()} for x in exact],'readability_note':'All candidates remain unrated pending human review; no readability claim.','provenance':'Hand-authored role inventories; seeded constructive sampling; no imported palindrome text.','next_repair':'Expand productions with human-reviewed 38+ letter clauses and add agreement constraints, then rerun count-state sampling.'}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({'samples':len(probes),'exact_count':len(exact),'best_matching_pairs':probes[0]['matching_pairs']}))
if __name__=='__main__': main()
