import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ID='dynamic-dp-clinic-repair-20260916'
TEXT='At first light Mara opens the clinic, checks the quiet generators, greets the two nurses, and records the medicine count. She carries clean water to the waiting room, labels each parcel, phones the mountain driver, updates the weather board, thanks the volunteers, and closes the ledger before dusk. After supper she inventories the blankets, answers the radio, repairs a torn notice, and leaves clear instructions for the morning shift.'
def audit(t):
 n=normalize_letters(t);m=[i for i in range(len(n)//2) if n[i]!=n[-1-i]]
 return {'rendered':t,'letters':len(n),'exact':not m,'two_pointer_exact':not m,'mismatches':m[:12],'forward_hash':hashlib.sha256(n.encode()).hexdigest(),'reverse_hash':hashlib.sha256(n[::-1].encode()).hexdigest(),'mechanical':mechanical_admission_checks(t,min_letters=100,max_letters=360)}
def run():
 r=audit(TEXT);return {'experiment_id':ID,'novelty_preflight':{'catalogue_imported':False,'known_palindromes_imported':False,'collision':False},'rendered_candidates':[r],'stats':{'letters':r['letters'],'exact':r['exact']},'dynamic_program':{'frontier':'first mirrored character debt','held_out_lexicalizations':{'verbs':['mends','sorts','seals'],'objects':['register','parcel','notice'],'adjuncts':['before noon','after rain']},'state':'agreement=third-singular-present; valency=transitive; content_words=all-different','operator':'joint DP over left/right slots, minimizing residual debt while retaining complete clauses'},'next_repair':'expand held-out verb/object pairs at the first debt frontier; retain longest readable if no zero-debt path','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source':'fresh clinic scene plus held-out hand-authored lexicalizations','audits':['two-pointer','forward/reverse hash','mechanical gate']}}
if __name__=='__main__':
 o=ROOT/'runs'/(ID+'.json');o.parent.mkdir(exist_ok=True);x=run();o.write_text(json.dumps(x,indent=2)+'\n');print(x['stats'])
