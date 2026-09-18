"""Reverse word-boundary shifting over a generated typed clause inventory."""
from __future__ import annotations
import argparse,json,re,sys
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lexical_bridge_fresh_short_sentences_20260913 import segment_reversed_tape,first_lexical_dead_frontier,VOCABULARY
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters,tokenize
ADJ=('careful','patient','skilled','quiet');SUB=('baker','teacher','nurse','artist');PAIRS=(('makes','bread'),('writes','notes'),('reads','letter'),('paints','walls'))
SOURCES=tuple(f"A {a} {s} {v} {o} today." for a in ADJ for s,(v,o) in zip(SUB,PAIRS))
def parse_source(t):
 w=tokenize(t);return len(w)==6 and w[0]=='a' and w[1] in ADJ and w[2] in SUB and any(w[3:5]==p for p in PAIRS) and w[5]=='today'
def parse_right(w): return len(w)>=3 and all(x in VOCABULARY for x in w)
def run(*,limit=256):
 rows=[];records=[];dead=[];truncated=False
 for i,source in enumerate(SOURCES,1):
  tape=normalize_letters(source);rev=tape[::-1];segs=segment_reversed_tape(rev,VOCABULARY,limit=limit);status='truncated' if len(segs)>=limit else 'exhausted';truncated|=status=='truncated';rows.append({'id':i,'source':source,'source_parse':parse_source(source),'reversed_tape':rev,'segmentation_count':len(segs),'status':status,'dead_frontier':first_lexical_dead_frontier(rev,VOCABULARY) if not segs else None})
  for words in segs:
   text=source.rstrip('.')+' '+' '.join(words).capitalize()+'.';t=normalize_letters(text);g=mechanical_admission_checks(text,min_letters=30,max_letters=260);codes=[k for k,v in g.items() if not v]
   if not parse_right(words):codes.append('independent_target_parse_failed')
   records.append({'rendered':text,'source':source,'target_words':words,'independent_exact_audit':{'exact':t==t[::-1],'letters':len(t),'normalized_sha256':sha256(t.encode()).hexdigest()},'independent_source_parse':parse_source(source),'independent_target_parse':parse_right(words),'central_admission':g,'mechanically_admitted':not codes,'rejection_codes':codes,'reader_status':'unreviewed'})
 exact=[r for r in records if r['independent_exact_audit']['exact'] and r['independent_source_parse'] and r['independent_target_parse']];admitted=[r for r in exact if r['mechanically_admitted']]
 return {'status':'reverse_boundary_shift_typed_clause','operator':'generated_typed_clause_immutable_reverse_cross_word_boundary_segmentation','config':{'generated_source_inventory':True,'source_count':len(SOURCES),'boundary_shifting':True,'cross_source_word_boundaries':True,'search_status':'truncated' if truncated else 'exhausted','catalogue_text':False,'human_readability_required_after_admission':True},'source_records':rows,'records':records,'exact_candidates':exact,'admitted_candidates':admitted,'dead_frontiers':dead,'provenance':{'generator_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'material':'authored typed clause inventory; no catalogue text'},'reader_status':'unreviewed; no programmatic result certifies readability'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();
 if a.out.exists():p.error('refusing to overwrite')
 r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'out':str(a.out),'sources':len(r['source_records']),'exact':len(r['exact_candidates']),'admitted':len(r['admitted_candidates'])},indent=2))
if __name__=='__main__':main()
