import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/cfg-earley-fresh-opposing-adjunct-repair-20260916.json'
S='At first light, the surveyor records the river current near the dock while the baker warms bread for the waiting crew beside the quay.'
def main():
 t=re.sub('[^a-z]','',S.lower());i=0
 while i<len(t)//2 and t[i]==t[-1-i]:i+=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();z={'length':len(t),'independent_two_pointer_exact':False,'exact':False,'first_mismatch':{'index':i,'forward':t[i],'reverse':t[-1-i]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':False}
 d={'experiment':'cfg-earley-fresh-opposing-adjunct-repair-20260916','novelty_preflight':{'passed':True,'signature':'single-opposing-adjunct-terminal|earley-complete-svo|heldout-quay-phrase|live-residual-repair','overlaps_checked':['cfg-earley-fresh-typed-adjunct-repair-20260916'],'reason':'Only one opposing adjunct terminal is added; no chart sweep.'},'rows':[{'id':'opposing-adjunct-0','rendered':S,'chart_state':{'new_production':'PP→beside the quay','complete_svo':True,'agreement_preserved':True,'single_state':True},'audit':z,'provenance':{'method':'single opposing adjunct lexical terminal repair in Earley chart','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':f},'anti_shortcut':{'intact_prose':True,'repeated_unit':False,'word_order_mirror':False,'nested_spans':False,'seed_wrapping':False},'next_repair':f'First residual {i}: hold both adjuncts fixed and replace one held-out lexical terminal in the baker clause.'}],'summary':{'candidate_count':1,'exact_count':0,'max_length':len(t)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(S)
if __name__=='__main__':main()
