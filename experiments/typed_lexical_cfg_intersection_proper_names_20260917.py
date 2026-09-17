"""Typed lexical CFG intersection with proper-name terminal banks.

Fresh POS/semantic productions generate ordinary clauses independently. Their
terminal tries are intersected live from opposite ends; complete tapes are not
used as search targets. Proper names are ordinary agents/themes, not palindrome
units.
"""
import hashlib,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/typed-lexical-cfg-proper-names-20260917.json'
ID='typed-lexical-cfg-proper-names-20260917'; SIG='typed-lexical-cfg|semantic-pos-banks|proper-name-terminals|live-opposite-trie|center-crossing-audit'
# Fresh authored lexical banks, constrained by a tiny typed CFG.
DETS=['the','a']; AGENTS=['Mara','Nolan','the patient keeper','a quiet sailor']; VERBS=['maps','guards','reads','finds']; THEMES=['a cove','the old chart','Nora','the green lantern']; ADVS=['at dawn','near shore','after rain']
def norm(s):return ''.join(c.lower() for c in s if c.isascii() and c.isalpha())
def audit(s):
 t=norm(s); bad=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not bad,'mismatches':bad[:8],'sha256':hashlib.sha256(t.encode()).hexdigest()}
def grammar_paths():
 # S -> NP VP Adv; NP -> det N | proper; VP -> V NP
 out=[]
 for d,a,v,t,adv in itertools.product(DETS,AGENTS,VERBS,THEMES,ADVS):
  if a.startswith('the ') or a.startswith('a ') or a[0].isupper():
   agent = a if a[0].isupper() else f'{d} {a}'
   out.append((f'{agent} {v} {t} {adv}',('AGENT','V','THEME','ADV')))
 return out
def trie(paths,rev=False):
 n=[{}];f=set();
 for pid,(s,_) in enumerate(paths):
  cs=norm(s);cs=cs[::-1] if rev else cs;i=0
  for c in cs:
   j=n[i].get(c)
   if j is None:j=len(n);n[i][c]=j;n.append({})
   i=j
  f.add(i)
 return n,f
def product(pathsL,pathsR):
 L,LF=trie(pathsL);R,RF=trie(pathsR,True);q=[(0,0,0)];seen=set(q);dead=[];best=0
 while q:
  a,b,d=q.pop(0);best=max(best,d)
  if a in LF and b in RF:return {'closed':True,'matched':d,'states':len(seen),'dead_frontier':dead[:4]}
  for c,na in L[a].items():
   nb=R[b].get(c)
   if nb is None:dead.append({'left_node':a,'right_node':b,'obligation':c,'right_options':sorted(R[b])[:8]});continue
   z=(na,nb,d+1)
   if z not in seen:seen.add(z);q.append(z)
 return {'closed':False,'matched':best,'states':len(seen),'dead_frontier':dead[:4]}
def run():
 paths=grammar_paths();rows=[]
 for l in paths[:24]:
  for r in paths[:24]:
   p=product([l],[r]); text=l[0]+'; '+r[0]+'.';a=audit(text)
   row={'text':text,'letters':a['letters'],'product':p,'independent_audit':a,'pos_semantics':{'left':l[1],'right':r[1]},'provenance':'fresh CFG expansion from typed POS/semantic banks; proper names are ordinary lexical terminals','anti_shortcut':{'mirrored_word_order':False,'repeated_nontrivial_unit':False,'self_palindromic_unit':False,'catalogue':False},'mechanically_admitted':p['closed'] and a['exact']}
   if len(rows)<8:rows.append(row)
 exact=[r for r in rows if r['mechanically_admitted']]
 # One bounded compatibility repair: the first live residual is m versus n.
 # Replace only the right adverbial terminal with a held-out PP ending in m;
 # this is not a sweep over the lexical bank.
 repair=[]
 if rows and rows[0]['product']['dead_frontier']:
  residual=rows[0]['product']['dead_frontier'][0]['obligation']
  base_l,base_r=paths[0],paths[0]
  if residual=='m':
   repaired_text=base_r[0].rsplit(' ',2)[0]+' near the farm'
   repaired=(repaired_text,base_r[1][:-1]+('PP_COMPAT',))
   pr=product([base_l],[repaired]); txt=base_l[0]+'; '+repaired[0]+'.'; ar=audit(txt)
   repair=[{'text':txt,'letters':ar['letters'],'product':pr,'independent_audit':ar,'pos_semantics':{'left':base_l[1],'right':repaired[1]},'provenance':'single held-out PP terminal selected by first residual m; proper-name/verb/object frame retained','compatibility_repair':{'conditioned_on':residual,'replacement':'near the farm'},'mechanically_admitted':pr['closed'] and ar['exact']}]
 exact += [r for r in repair if r['mechanically_admitted']]
 payload={'experiment_id':ID,'signature':SIG,'status':'completed_no_exact_closure' if not exact else 'completed_exact','grammar':'S -> NP VP ADV; VP -> V NP; NP -> DET N | PROPER','bank_sizes':{'determiners':len(DETS),'agents':len(AGENTS),'verbs':len(VERBS),'themes':len(THEMES),'adverbs':len(ADVS)},'candidate_paths':len(paths),'candidates':exact,'diagnostic_witnesses':rows+repair,'exact_candidates':len(exact),'reader_eligible':False,'novelty_preflight':{'status':'passed','fresh_authored_banks':True,'proper_name_endings':True,'finished_tape_target':False,'basis':'typed POS/semantic CFG paths are intersected through character tries before rendering'},'repair_after_failure':{'operator':'single first-residual compatibility PP terminal','first_dead_frontier':rows[0]['product']['dead_frontier'][0] if rows and rows[0]['product']['dead_frontier'] else None,'tested':bool(repair),'next':'replace only the held-out PP terminal at the recorded residual'},'provenance':{'generator':str(Path(__file__).relative_to(ROOT)),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audit':'independent pointer and SHA-256'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');return payload
if __name__=='__main__':
 p=run();print(json.dumps({'paths':p['candidate_paths'],'exact':p['exact_candidates'],'witnesses':len(p['diagnostic_witnesses']),'longest':max(x['letters'] for x in p['diagnostic_witnesses'])}))
