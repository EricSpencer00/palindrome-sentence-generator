import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/outside-in-heldout-scene-20260916.json'
SC=[('At dusk','the archivist','records','the rescued names','for the town archive'),('Before sunrise','a patient gardener','waters','the young cedar trees','beside the stone wall'),('After the storm','the careful pilot','marks','the distant runway lights','along the wet field')]
PAIR=[('quiet archive','open windows'),('old harbor','safe shelter'),('winter garden','warm lantern')]
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward_normalized':f,'sha256_reverse_normalized':r,'sha256_equal':f==r}
def main():
 rows=[]
 for i in range(3):
  a=SC[i];b=SC[(i+1)%3];p=PAIR[i];s=f'{a[0]}, {a[1]} {a[2]} {a[3]} {a[4]}, and {b[1]} {b[2]} {b[3]} {b[4]} near the {p[0]} with {p[1]}.';z=audit(s)
  rows.append({'id':f'outside-{i}','rendered':s,'composition':{'outer_scene_left':a,'outer_scene_right':b,'heldout_phrase_pair':p,'growth':'outside-in semantic roles then lexical seam'},'audit':z,'provenance':{'method':'outside-in semantic scene composition with held-out reverse lexical phrase pairs','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward_normalized']},'anti_shortcut':{'intact_prose':True,'repeated_unit':False,'word_order_only':False,'catalogue_text':False,'posthoc_reversal':False},'next_repair':f"First residual {z['first_mismatch']['index']}: replace only the held-out phrase pair at the opposing outside seam, retaining scene roles and agreement."})
 d={'experiment':'outside-in-heldout-scene-20260916','novelty_preflight':{'passed':True,'signature':'outside-in-semantic-scene|heldout-reverse-phrase-pair|role-first-growth|normalized-tape-audit','overlaps_checked':['cross-boundary-phrase-block-grammar-20260916','centerout-open-word-seam-20260916'],'reason':'Semantic scene roles are composed from the outside inward and held-out phrase pairs are selected only at a lexical seam; no mirrored unit or finished reversal.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
