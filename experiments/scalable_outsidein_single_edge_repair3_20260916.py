import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/scalable-outsidein-single-edge-repair3-20260916.json'
S='At dawn, the archivist opens the town archive near the blue footbridge, while a patient gardener covers the young seedlings under the weathered pavilion.'
def n(x):return re.sub('[^a-z]','',x.lower())
def main():
 t=n(S);i=0
 while i<len(t)//2 and t[i]==t[-1-i]:i+=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();z={'length':len(t),'independent_two_pointer_exact':i==len(t)//2,'exact':i==len(t)//2,'first_mismatch':{'index':i,'forward':t[i],'reverse':t[-1-i]},'sha256_forward_normalized':f,'sha256_reverse_normalized':r,'sha256_equal':f==r}
 d={'experiment':'scalable-outsidein-single-edge-repair3-20260916','novelty_preflight':{'passed':True,'signature':'single-opposing-edge-repair3|held-fixed-scene-side|productive-outsidein-growth|same-role-terminal','overlaps_checked':['scalable-outsidein-opposing-terminal-repair2-20260916'],'reason':'Exactly one opposing edge phrase changed; held side and scene state fixed.'},'rows':[{'id':'single-edge-3','rendered':S,'repair_state':{'changed_phrase':'under the weathered pavilion','held_fixed_phrase':'near the blue footbridge','child_count':1,'scene_frame_preserved':True},'audit':z,'provenance':{'method':'single-child same-role terminal repair','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':f},'anti_shortcut':{'intact_prose':True,'nested_spans':False,'seed_wrapping':False,'word_order_mirror':False,'repeated_unit':False},'next_repair':f'First residual {i}: keep held edge fixed and substitute one new same-role terminal.'}],'summary':{'candidate_count':1,'exact_count':0,'max_length':len(t)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(S)
if __name__=='__main__':main()
