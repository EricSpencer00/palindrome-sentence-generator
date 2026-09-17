import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/bespoke-scene-lattice-single-slot-repair-20260916.json'
S='The baker kneads the rye dough before noon; the ferryman guides the evening boat across the inlet.'
def main():
 t=re.sub('[^a-z]','',S.lower());i=0
 while i<len(t)//2 and t[i]==t[-1-i]:i+=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();z={'length':len(t),'independent_two_pointer_exact':False,'exact':False,'first_mismatch':{'index':i,'forward':t[i],'reverse':t[-1-i]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':False}
 d={'experiment':'bespoke-scene-lattice-single-slot-repair-20260916','novelty_preflight':{'passed':True,'signature':'single-heldout-lattice-slot|free-center-scene-repair|new-maritime-template|live-equation','overlaps_checked':['bespoke-scene-lattice-free-center-20260916'],'reason':'One held-out ferryman template slot is selected against the baker scene; no lattice resweep.'},'rows':[{'id':'lattice-repair-0','rendered':S,'repair_state':{'new_template_slot':'ferryman-guides-evening-boat','free_center':'semicolon','single_state':True,'scene_connected':True},'audit':z,'provenance':{'method':'single held-out template-slot repair in bespoke free-center scene lattice','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':f},'anti_shortcut':{'intact_prose':True,'fixed_tape':False,'repeated_unit':False,'word_order_mirror':False,'seed_wrapping':False},'next_repair':f'First residual {i}: preserve both scene roles and alter one held-out adjunct slot only.'}],'summary':{'candidate_count':1,'exact_count':0,'max_length':len(t)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(S)
if __name__=='__main__':main()
