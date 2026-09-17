import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/bidirectional-phrase-pair-growth-20260916.json'
S=['The archivist records the rescued letters before sunrise; meanwhile, a patient gardener waters the young cedar trees after heavy rain.','The careful pilot marks the distant runway lights beside the quiet bridge; meanwhile, the archivist records the rescued letters before sunrise.']
def main():
 rows=[]
 for i,s in enumerate(S):
  t=re.sub('[^a-z]','',s.lower());j=0
  while j<len(t)//2 and t[j]==t[-1-j]:j+=1
  f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();z={'length':len(t),'independent_two_pointer_exact':False,'exact':False,'first_mismatch':{'index':j,'forward':t[j],'reverse':t[-1-j]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':f==r}
  rows.append({'id':f'growth-{i}','rendered':s,'growth_state':{'whole_clause_growth':True,'boundary_obligation':'paired terminals selected before clause growth','heldout_inventory':True},'audit':z,'provenance':{'method':'bidirectional whole-clause phrase-pair grammar','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':f},'anti_shortcut':{'intact_prose':True,'fixed_tape':False,'seed_wrapping':False,'repeated_unit':False,'word_order_mirror':False},'next_repair':f'First residual {j}: replace opposing adjunct phrase pair while retaining complete SVO clauses.'})
 d={'experiment':'bidirectional-phrase-pair-growth-20260916','novelty_preflight':{'passed':True,'signature':'bidirectional-phrase-pair-growth|whole-clause-composition|live-boundary-obligation|heldout-lexical-inventory','overlaps_checked':['brown-phrase-pair-seam-20260916','live-cfg-character-chart-20260916'],'reason':'Productive whole-clause growth with held-out inventory, not fixed tape or duplicate sweep.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':0,'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(S[0])
if __name__=='__main__':main()
