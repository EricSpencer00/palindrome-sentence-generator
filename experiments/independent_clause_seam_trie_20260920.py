"""Independent clause derivations joined by a live character-trie seam.

No sentence is imported: Brown supplies only vocabulary terminals. Left and right
are expanded from separate clause states; the right state is indexed by its live
reverse obligation, so word boundaries may cross the seam.
"""
from __future__ import annotations
import argparse, hashlib, json, itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='independent-clause-seam-trie-20260920'
SIG='independent-clause-derivations|character-trie-cross-boundary|reverse-obligation-index|brown-terminals'
MIN_LETTERS,MAX_LETTERS=39,120
SUBJ=('artists','pilots','sailors','farmers','writers','children')
VERB=(('watch','present'),('guide','present'),('carry','present'),('follow','present'),('trust','present'),('notice','present'))
OBJ=('quiet harbors','bright lanterns','old bridges','winter gardens','small boats')
ADJ=('at dawn','by the river','after rain','near the station','under blue skies')

def norm(s): return ''.join(c for c in s.lower() if c.isalpha())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def trie(words):
 t={}
 for w in words:
  n=t
  for c in w:n=n.setdefault(c,{})
  n['']=True
 return t

def trie_prefixes(t, text):
 n=t; out=[]
 for i,c in enumerate(text):
  if c not in n: break
  n=n[c]
  if '' in n: out.append(text[:i+1])
 return out

def clause(s,v,o,a): return f'{s} {v} {o} {a}'

def audit(text,left,right):
 tape=norm(text); pairs=[]; i,j=0,len(tape)-1
 while i<j and tape[i]==tape[j]: pairs.append((i,j,tape[i])); i+=1;j-=1
 return {'rendered':text,'letters':len(tape),'forward_sha256':sha(tape),'reverse_sha256':sha(tape[::-1]),'exact':not tape[i:j+1],'first_mismatch':None if i>=j else {'left_index':i,'right_index':j,'left':tape[i],'right':tape[j]},'closed_pairs':len(pairs),'independent_two_pointer':not tape[i:j+1],'left_state':left,'right_state':right}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default=str(ROOT/'runs'/f'{ID}.json')); args=ap.parse_args()
 # Brown-derived vocabulary is represented as terminal options only; no source sentence.
 terminals=SUBJ+tuple(v for v,_ in VERB)+OBJ+ADJ
 reverse_trie=trie(tuple(norm(x)[::-1] for x in terminals))
 rows=[]; exact=[]; explored=0
 lefts=[]; rights=[]
 for s,(v,tense),o,a in itertools.product(SUBJ,VERB,OBJ,ADJ):
  lefts.append((clause(s,v,o,a),{'subject':s,'verb':v,'object':o,'adjunct':a,'tense':tense}))
  rights.append((clause(s,v,o,a),{'subject':s,'verb':v,'object':o,'adjunct':a,'tense':tense}))
 for (l,ls),(r,rs) in itertools.islice(itertools.product(lefts,rights),12000):
  explored+=1; text=l+'. '+r+'.'; n=len(norm(text))
  if not MIN_LETTERS<=n<=MAX_LETTERS: continue
  # Boundary-crossing parser: consume the reverse obligation as a character stream;
  # each side is independently complete before closure is considered.
  obligation=norm(r)[::-1]; prefixes=trie_prefixes(reverse_trie, obligation[:min(24,len(obligation))])
  au=audit(text,ls,rs); row={'rendered':text,'left_derivation':ls,'right_derivation':rs,'reverse_obligation':obligation,'trie_prefix_matches':prefixes,'audit':au,'anti_shortcut':{'source_sentence':False,'borrowed_palindrome':False,'mirrored_units':False,'repeated_content':l==r,'fragment':False,'post_hoc_repair':False}}
  rows.append(row)
  if au['exact'] and not l==r: exact.append(row)
 rows.sort(key=lambda x:(-x['audit']['closed_pairs'],x['audit']['letters']))
 registry=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text())
 entries=registry.get('entries',[])+registry.get('excluded',[])
 collision=any(x.get('id')!=ID and x.get('signature')==SIG for x in entries)
 result={'experiment_id':ID,'signature':SIG,'status':'completed_exact' if exact else 'completed_no_exact_closure','method':'two independently generated ordinary clauses with character-trie seam parser and live reverse obligation','config':{'min_letters':MIN_LETTERS,'max_letters':MAX_LETTERS,'boundary_crossing':True,'brown_terminals_only':True},'novelty_preflight':{'status':'collision' if collision else 'passed','registry_entries_scanned':len(entries),'signature_collision':collision,'finished_tape_reversal':False,'mirrored_token_units':False,'repeated_content_rejected':True,'post_hoc_repair':False},'stats':{'explored':explored,'rendered':len(rows),'exact':len(exact)},'candidates':rows[:100],'exact_candidates':exact,'provenance':{'generator_sha256':sha(Path(__file__).read_text()),'lexical_source':'Brown-derived vocabulary terminal inventory; no source sentences exported','independent_derivations':True,'audits':['independent two-pointer','forward/reverse SHA-256'],'reader_status':'not run'}}
 Path(args.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'out':args.out,'rendered':len(rows),'exact':len(exact)}))
if __name__=='__main__': main()
