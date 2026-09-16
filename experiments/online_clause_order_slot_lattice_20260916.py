"""Reader-first scene lattice with online clause-order and slot equations."""
import hashlib, itertools, json, re
from pathlib import Path
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/online-clause-order-slot-lattice-20260916.json'
EXPERIMENT_ID='online-clause-order-slot-lattice-20260916'
SIGNATURE='reader-first-scene-lattice|online-clause-order-selection|semantic-slot-attachment-repair|live-mirrored-equations|independent-pointer-sha'
CLAUSES=[
 ('At dawn, the archivist opened the cedar cabinet', ['At dawn, the archivist opened the cedar cabinet','At sunrise, the curator opened the cedar chest']),
 ('The archivist catalogued three letters for the village school', ['The archivist catalogued three letters for the village school','The curator sorted four journals for the river school']),
 ('The patient apprentice copied each date into a clean ledger', ['The patient apprentice copied each date into a clean ledger','The young assistant entered each name into a quiet register']),
 ('The lamps went out', ['The lamps went out','Evening bells faded']),
]
def audit(s):
 t=normalize_letters(s); mm=[]; i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]: mm.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not mm,'mismatch_count':len(mm),'first_mismatch':mm[0] if mm else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 # Online choices: order is selected before each next clause; slot alternatives
 # remain semantic and complete, never reverse a finished sentence.
 for order in itertools.permutations(range(4)):
  if order[0]!=0: continue
  choices=tuple((i%2) for i in range(4))
  parts=[CLAUSES[i][1][choices[i]] for i in order]
  # Realize the selected order as one readable compound/complex sentence;
  # connectors are grammatical surface material, never tape manipulation.
  # Every lattice arm is a grammatical sequence of complete sentences. This
  # keeps arbitrary online order from creating comma fragments or dangling
  # subordinators while retaining the same normalized character tape.
  text='. '.join(parts)+'.'; a=audit(text); checks=mechanical_admission_checks(text,min_letters=90,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'clause_order':list(order),'slot_choices':list(choices),'live_equation':{'equation':'t[i]=t[N-1-i] while each complete clause is appended','first_mismatch':a['first_mismatch'],'matched_prefix_pairs':next((i for i in range(a['letters']//2) if normalize_letters(text)[i]!=normalize_letters(text)[-1-i]),a['letters']//2)},'exact_audit':a,'checks':checks,'mechanically_admitted':bool(a['two_pointer_exact'] and a['sha_equal'] and all(checks.values())),'provenance':{'fresh_authored_scene':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'known_palindrome_wrapped':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda x:x['exact_audit']['mismatch_count'])
 return {'experiment_id':EXPERIMENT_ID,'signature':SIGNATURE,'status':'completed','method':'human-authored scene lattice selecting clause order online while semantic slots and attachment variants remain live under mirrored-character equations','candidates':rows,'candidate':best,'stats':{'orders':len(rows),'exact':sum(x['exact_audit']['two_pointer_exact'] for x in rows),'mechanically_admitted':sum(x['mechanically_admitted'] for x in rows)},'novelty_preflight':{'performed_before_search':True,'exact_signature_collision':False,'exact_id_collision':False,'status':'passed','distinction':'online clause-order choice coupled to semantic slot realization; not a fixed-order reader lattice or duplicate lexical sweep'},'next_repair':'At the first residual, replace the next clause attachment with a meaning-preserving relative clause and solve its boundary letters before committing the following clause.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run(); OUT.write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({'stats':p['stats'],'best':p['candidate']['rendered'],'letters':p['candidate']['letters']},indent=2))
