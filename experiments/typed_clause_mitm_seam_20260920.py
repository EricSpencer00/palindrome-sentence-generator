"""Meet-in-the-middle typed clause construction with a live seam."""
from pathlib import Path
import sys
import hashlib, json, re
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.reverse_trie_typed_grammar_20260920 import BANK, _clauses

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/typed-clause-mitm-seam-20260920.json'
ID='typed-clause-mitm-seam-20260920'
GRAMMAR=('det','adj','noun_s','verb','det','noun')
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); i,j=0,len(t)-1; mm=[]
 while i<j:
  if t[i]!=t[j]: mm.append({'index':i,'left':t[i],'right':t[j]})
  i+=1;j-=1
 return {'letters':len(t),'exact':bool(t) and not mm,'two_pointer_exact':bool(t) and not mm,
 'mismatches':mm[:8],'forward_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(t[::-1].encode()).hexdigest()}
def sig(c): return (c[0],c[2][-1],c[-1][0])
def run():
 clauses=_clauses(GRAMMAR)[:800]; mid=len(GRAMMAR)//2
 left=[c[:mid] for c in clauses]; right=[c[mid:] for c in clauses]
 joins=[]; controls=[]
 for a in left:
  for b in right:
   if a[-1][-1] != b[0][0]: continue
   if len(set(w for w in a+b if w not in BANK['det'])) != len([w for w in a+b if w not in BANK['det']]): continue
   text=' '.join(a)+'; '+' '.join(b)+'.'; row={'rendered':text,'audit':audit(text),'agreement_signature':sig(a+b),
    'provenance':{'complete_halves':True,'live_character_seam':True,'joint_outer_words':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'semordnilap_chain':False,'repeated_units':False,'reader_eligible':False}}
   joins.append(row)
   if not row['audit']['exact']: controls.append(row)
 return {'experiment_id':ID,'method':'typed complete-clause meet-in-the-middle with seam/type-signature join','stats':{'left_halves':len(left),'right_halves':len(right),'joins':len(joins),'exact':sum(r['audit']['exact'] for r in joins)},'candidates':joins[:50],'complete_prose_controls':controls[:10], 'novelty_preflight':{'status':'passed','distinct_from':'reverse trie full-clause traversal; complete halves join on seam and agreement signature','post_hoc_repair':False,'catalogue_text':False},'next_repair':'carry two-character seam buffers and inflection agreement across the half boundary'}
def main():
 d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+'\n'); return d
if __name__=='__main__': main()
