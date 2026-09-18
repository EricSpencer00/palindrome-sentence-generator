"""Typed imperative plans compiled through Astra's whole-text product."""
from __future__ import annotations
import argparse,json,sys
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.whole_text_palindrome_product_20260913 import compile_slots,construct,replay_path
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters,tokenize

PLANS=(
 {'verb':('carry',),'det':('a','the'),'object':('water',),'prep':('to',),'location':('camp',)},
 {'verb':('place',),'det':('a','the'),'object':('letter',),'prep':('in',),'location':('drawer',)},
 {'verb':('read',),'det':('a','the'),'object':('book',),'prep':('for',),'location':('class',)},
 {'verb':('wash',),'det':('a','the'),'object':('window',),'prep':('with',),'location':('water',)},
)
def slots(plan): return (plan['verb'],plan['det'],plan['object'],plan['prep'],plan['location'])
def parse(text,plan):
 w=tokenize(text); choices=slots(plan);return len(w)==len(choices) and all(x in choices[i] for i,x in enumerate(w))
def run(*,max_states=100000):
 rows=[];all_records=[];total=0;truncated=False
 for i,plan in enumerate(PLANS,1):
  grammar=compile_slots(slots(plan));result=construct(grammar,max_states);total+=result['states'];truncated|=result['truncated']
  plan_rows=[]
  for rec in result['records']:
   text=' '.join(rec['words']).capitalize()+'.';gate=mechanical_admission_checks(text,min_letters=30,max_letters=260);t=normalize_letters(text);codes=[k for k,v in gate.items() if not v]
   if not parse(text,plan):codes.append('independent_plan_parse_failed')
   row={**rec,'plan_id':f'I{i:02d}','rendered':text,'independent_plan_parse':parse(text,plan),'independent_exact_audit':{'exact':t==t[::-1],'letters':len(t),'normalized_sha256':sha256(t.encode()).hexdigest()},'central_admission':gate,'mechanically_admitted':not codes,'rejection_codes':codes,'reader_status':'unreviewed; programmatic checks do not certify readability'};plan_rows.append(row);all_records.append(row)
  rows.append({'plan_id':f'I{i:02d}','semantic_plan':plan,'slots':slots(plan),'states':result['states'],'states_exhausted':result['states_exhausted'],'truncated':result['truncated'],'deepest_matched_pairs':result['deepest_matched_pairs'],'records':plan_rows})
 exact=[r for r in all_records if r['independent_exact_audit']['exact']];admitted=[r for r in exact if r['independent_plan_parse'] and r['mechanically_admitted']]
 return {'status':'whole_text_typed_imperative_plans','operator':'Astra whole_text_palindrome_product over separate valency-typed imperative plans','config':{'plan_count':len(PLANS),'kernel':'experiments.whole_text_palindrome_product_20260913','kernel_center_inside_word':True,'finite_scope':True,'independent_semantic_parse':True,'central_admission_required':True,'no_catalogue_text':True,'no_repeated_content':True,'search_status':'truncated' if truncated else 'exhausted'},'plan_runs':rows,'records':all_records,'exact_survivors':exact,'admitted_exact_survivors':admitted,'stats':{'states':total,'records':len(all_records)},'provenance':{'generator_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'kernel_material':'Astra whole-text product','plan_material':'authored ordinary imperative valency frames'},'reader_status':'unreviewed; no programmatic result certifies readability'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():p.error(f'refusing to overwrite {a.out}')
 r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'out':str(a.out),'states':r['stats']['states'],'exact':len(r['exact_survivors'])},indent=2))
if __name__=='__main__':main()
