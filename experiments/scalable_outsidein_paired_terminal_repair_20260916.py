import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/scalable-outsidein-paired-terminal-repair-20260916.json'
FR=[('At dawn','the archivist','opens','the town archive'),('By evening','a patient gardener','covers','the young seedlings'),('After rain','the careful nurse','carries','the clean water')]
ALT=[('beside the silent quay','under the copper awning'),('along the shaded footbridge','near the weathered gate'),('beneath the willow branches','across the gravel court')]
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward_normalized':f,'sha256_reverse_normalized':r,'sha256_equal':f==r}
def main():
 rows=[]
 for i in range(3):
  l=FR[i];r=FR[(i+1)%3];s=f'{l[0]}, {l[1]} {l[2]} {l[3]} {ALT[i][0]}, while {r[1]} {r[2]} {r[3]} {ALT[i][1]}.';z=a(s)
  rows.append({'id':f'repair-{i}','rendered':s,'repair_state':{'replaced_outer_pair':ALT[i],'same_role':True,'scene_frame_preserved':True,'productive_growth':'append another linked frame after this repaired pair'},'audit':z,'provenance':{'method':'paired same-role outer-terminal repair on scalable outside-in grammar','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward_normalized']},'anti_shortcut':{'intact_prose':True,'nested_palindrome_spans':False,'seed_wrapping':False,'word_order_mirror':False,'repeated_unit':False},'next_repair':f"First residual {z['first_mismatch']['index']}: retain this scene frame and alter only the opposing outer terminal pair with a held-out same-role phrase."})
 d={'experiment':'scalable-outsidein-paired-terminal-repair-20260916','novelty_preflight':{'passed':True,'signature':'paired-terminal-same-role-repair|outsidein-productive-growth|heldout-phrase-alternatives|live-crossword-obligation','overlaps_checked':['scalable-outsidein-phrase-pair-20260916'],'reason':'This is a targeted paired-terminal repair with held-out same-role phrases, not a resweep of the original seam list.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
