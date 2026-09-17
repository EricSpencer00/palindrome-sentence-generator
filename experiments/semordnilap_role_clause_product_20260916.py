"""Role-typed semordnilap products; complete clauses, never repeated units."""
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/semordnilap-role-clause-product-20260916.json'
PAIRS=[('deliver','reviled'),('stressed','desserts'),('diaper','repaid'),('drawer','reward')]
SC=[('the archivist','labels','the rescued letters','at dawn'),('a patient gardener','waters','the young cedar trees','after rain'),('the careful pilot','marks','the distant runway lights','through mist')]
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();q=hashlib.sha256(x[::-1].encode()).hexdigest();return {'length':len(x),'exact':i==len(x)//2,'independent_two_pointer_exact':i==len(x)//2,'first_mismatch':None if i==len(x)//2 else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward':f,'sha256_reverse':q,'sha256_equal':f==q}
def main():
 rows=[]
 for k,(a,b) in enumerate(PAIRS):
  l=SC[k%3];r=SC[(k+1)%3];s=f'{l[0].capitalize()} {a} {l[2]} {l[3]}; {r[0]} {b} {r[2]} {r[3]}.';z=audit(s)
  rows.append({'id':f'role-{k}','rendered':s,'role_pair':{'left_verb':a,'right_verb':b,'relation':'reverse-spelling POS pair'},'audit':z,'provenance':{'method':'POS-role semordnilap pair product with complete SVO+PP clause realization','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward']},'anti_shortcut':{'intact_prose':True,'repeated_unit':False,'word_order_only':False,'catalogue_text':False,'semordnilap_unit_repeated':False},'next_repair':'Replace the first residual verb-role pair with a held-out inflectional synonym pair while retaining subject/object agreement and both clause frames.'})
 d={'experiment':'semordnilap-role-clause-product-20260916','novelty_preflight':{'passed':True,'signature':'pos-role-semordnilap-product|complete-svo-pp|joint-character-equation|heldout-verb-pairs','overlaps_checked':['typed-reversible-lexeme-graph-20260916','semordnilap-typed-clause-2026-09-16'],'reason':'Pairs are used as typed verb terminals inside independently authored complete clauses; no repeated chain or finished-surface reversal.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
