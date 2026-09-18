"""Broad typed semantic-span bridge: authored clause inventory -> reverse lattice."""
from __future__ import annotations
import argparse,json,sys
from hashlib import sha256
from pathlib import Path
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters,tokenize

ADJ=('careful','patient','skilled','quiet'); SUBJ=('baker','teacher','farmer','editor'); VERB=('makes','writes','carries','reads'); OBJ=('bread','notes','water','letters')
VOCAB=tuple(sorted(set(ADJ+SUBJ+VERB+OBJ+('a','the','and','clear','fresh','warm','today','our','new'))))
SOURCE=[(a,s,v,o) for a in ADJ for s in SUBJ for v in VERB for o in OBJ if (v,o) in {('makes','bread'),('writes','notes'),('carries','water'),('reads','letters')}]
def render_source(row): return f"A {row[0]} {row[1]} {row[2]} {row[3]} today."
def parse_source(text):
 w=tokenize(text);return len(w)==6 and w[0]=='a' and w[1] in ADJ and w[2] in SUBJ and w[3] in VERB and (w[3],w[4]) in {('makes','bread'),('writes','notes'),('carries','water'),('reads','letters')} and w[5]=='today'
def parse_right(words):
 return len(words)==6 and words[0] in {'a','the'} and words[1] in ADJ and words[2] in SUBJ and words[3] in VERB and (words[3],words[4]) in {('makes','bread'),('writes','notes'),('carries','water'),('reads','letters')} and words[5] in {'today','new'}
def lattice(tape,limit=256):
 @lru_cache(None)
 def go(pos):
  if pos==len(tape): return ((),)
  out=[]
  for w in VOCAB:
   if tape.startswith(w,pos):
    for tail in go(pos+len(w)):
     out.append((w,)+tail)
     if len(out)>=limit:return tuple(out)
  return tuple(out)
 return go(0), go.cache_info().currsize
def audit(source,right):
 text=source.rstrip('.').replace(' today',' today')+' '+' '.join(right).capitalize()+'.';tape=normalize_letters(text);g=mechanical_admission_checks(text,min_letters=30,max_letters=260);codes=[k for k,v in g.items() if not v]
 if not parse_source(source):codes.append('independent_source_parse_failed')
 if not parse_right(right):codes.append('independent_right_parse_failed')
 return {'rendered':text,'source':source,'right_words':right,'letters':len(tape),'independent_exact_audit':{'exact':bool(tape) and tape==tape[::-1],'normalized_sha256':sha256(tape.encode()).hexdigest()},'independent_source_parse':parse_source(source),'independent_right_parse':parse_right(right),'central_admission':g,'mechanically_admitted':not codes,'rejection_codes':codes,'reader_status':'unreviewed; programmatic checks do not certify readability'}
def run(*,segmentation_limit=256):
 records=[];sources=[];dead=[];truncated=False
 for i,row in enumerate(SOURCE,1):
  source=render_source(row);tape=normalize_letters(source);rev=tape[::-1];segs,cache=lattice(rev,segmentation_limit);rowdata={'source_id':f'S{i:03d}','source':source,'source_letters':len(tape),'source_parse':parse_source(source),'segmentation_count':len(segs),'lattice_states':cache,'search_status':'exhausted' if len(segs)<segmentation_limit else 'truncated','reversed_tape':rev}
  sources.append(rowdata)
  if len(segs)>=segmentation_limit:truncated=True
  if not segs: dead.append({'source_id':rowdata['source_id'],'position':next((p for p in range(len(rev)) if not any(rev.startswith(w,p) for w in VOCAB)),len(rev)),'prefix':rev[:12],'reason':'no_lexical_lattice_path'})
  records.extend(audit(source,seg) for seg in segs)
 exact=[r for r in records if r['independent_exact_audit']['exact']];parsed=[r for r in exact if r['independent_source_parse'] and r['independent_right_parse']];admitted=[r for r in parsed if r['mechanically_admitted']]
 return {'status':'semantic_span_bridge_frequency_bounded_lattice','operator':'typed_clause_inventory_reverse_tape_frequency_bounded_lattice','config':{'inventory_generated_from_typed_lexicons':True,'source_count':len(SOURCE),'cross_source_word_boundaries':True,'both_independent_syntax_witnesses':True,'semantic_relation_required':True,'catalogue_text':False,'word_mirror_forbidden':True,'repeated_span_gate':True,'segmentation_limit':segmentation_limit,'search_status':'truncated' if truncated else 'exhausted'},'source_records':sources,'records':records,'exact_survivors':exact,'parsed_exact_survivors':parsed,'admitted_exact_survivors':admitted,'dead_frontiers':dead,'provenance':{'generator_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'source_material':'typed authored lexicon inventory; no catalogue text'},'reader_facing_test':{'status':'not triggered because no admitted exact survivor','required':'randomized blinded intact-prose and shuffled controls'},'reader_status':'unreviewed; no programmatic result certifies readability'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--segmentation-limit',type=int,default=256);a=p.parse_args()
 if a.out.exists():p.error(f'refusing to overwrite {a.out}')
 r=run(segmentation_limit=a.segmentation_limit);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'out':str(a.out),'sources':len(r['source_records']),'records':len(r['records']),'exact':len(r['exact_survivors']),'status':r['config']['search_status']},indent=2))
if __name__=='__main__':main()
