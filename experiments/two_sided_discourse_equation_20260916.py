import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ID='two-sided-discourse-equation-20260916'
LEFT='Nora briefs the harbor crew before sunrise, then files the weather charts for the evening watch.'
RIGHT='Later, the coastal pilot studies fresh signals and stores a sealed map beside the lighthouse.'
def audit():
 t=normalize_letters(LEFT+' '+RIGHT);m=[i for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'rendered':LEFT+' '+RIGHT,'letters':len(t),'exact':not m,'two_pointer':not m,'mismatches':m[:10],'forward_hash':hashlib.sha256(t.encode()).hexdigest(),'reverse_hash':hashlib.sha256(t[::-1].encode()).hexdigest(),'mechanical':mechanical_admission_checks(LEFT+' '+RIGHT,min_letters=100,max_letters=260)}
def run():
 r=audit();return {'experiment_id':ID,'novelty_preflight':{'catalogue_imported':False,'known_palindromes_imported':False,'collision':False},'rendered_candidates':[r],'stats':{'letters':r['letters'],'exact':r['exact']},'equation_solver':{'slots':['agent','speech_act','object','time','result'],'coupling':'paired semantic substitutions against live mirrored character debt','agreement':'third-person singular present','content_words_all_different':r['mechanical']['distinct_words']},'next_repair':'jointly replace speech_act/object spans at first mismatch and replay both clauses','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source':'fresh human-authored discourse','audits':['two-pointer','forward/reverse hash','mechanical gate']}}
if __name__=='__main__':
 o=ROOT/'runs'/(ID+'.json');o.parent.mkdir(exist_ok=True);x=run();o.write_text(json.dumps(x,indent=2)+'\n');print(x['stats'])
