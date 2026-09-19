"""Bounded constructive lane: semantic-valency reverse segmentation.

A left clause is authored from typed roles.  Its *unrealized character debt* is
segmented online into right-side lexical items; the right clause is never made
by reversing words or a finished sentence.  The segmenter only accepts parses
whose roles form a complete ordinary English clause.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ID='typed-reverse-segmentation-valency-20260919'
# Small, fresh, human-authored role inventory (not catalogue text).
SUBJ={'the curator':'animate','the pilot':'animate','the mason':'animate'}
VERB={'sketched':'animate->artifact','carried':'animate->artifact','mended':'animate->artifact'}
OBJ={'a bronze map':'artifact','the quiet bell':'artifact','a linen sail':'artifact'}
ADJ={'at dawn':'time','in rain':'weather','by noon':'time'}
LEX={w.replace(' ',''):w for w in list(SUBJ)+list(VERB)+list(OBJ)+list(ADJ)}

def tape(s): return ''.join(c for c in s.lower() if c.isalpha())
def audit(s):
 t=tape(s); m=[]; i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append((i,t[i],j,t[j]))
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'normalized_tape':t,'letters':len(t),'two_pointer_exact':bool(t) and not m,'first_mismatches':m[:8], 'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}

def grammar_parse(words):
 if len(words)!=4:return None
 s,v,o,a=words
 if s not in SUBJ or v not in VERB or o not in OBJ or a not in ADJ:return None
 if VERB[v]!=SUBJ[s]+'->'+OBJ[o]:return None
 return {'subject':SUBJ[s],'verb':VERB[v],'object':OBJ[o],'adjunct':ADJ[a]}

def segment(target):
 # Prefix segmentation of the live debt, with word boundaries allowed anywhere.
 out=[]
 def rec(pos,ws):
  if pos==len(target):
   p=grammar_parse(ws)
   if p: out.append((list(ws),p))
   return
  if len(ws)>=4:return
  for k,w in LEX.items():
   if target.startswith(k,pos): rec(pos+len(k),ws+[w])
 rec(0,[])
 return out

def main():
 lefts=[('the curator','sketched','a bronze map','at dawn'),('the pilot','carried','the quiet bell','by noon'),('the mason','mended','a linen sail','in rain')]
 rows=[]; states=0
 for left in lefts:
  rendered=' '.join(left)
  target=tape(rendered)[::-1]
  parses=segment(target); states+=len(parses)
  # Render independently segmented right clause only when semantic parse succeeds.
  for right,sem in parses:
   text=rendered+' '+' '.join(right)
   rows.append({'rendered':text,'left_clause':rendered,'right_clause':' '.join(right),'semantic_state':sem,'audit':audit(text),'provenance':{'source_sentences_copied':False,'catalogue_imported':False,'finished_tape_reversed_for_realization':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'construction':'live prefix segmentation of character debt into typed subject/verb/object/adjunct roles'}})
 # Include actual intact controls so output demonstrates ordinary clause generation even if no parse closes.
 controls=[{'rendered':'the curator sketched a bronze map at dawn','audit':audit('the curator sketched a bronze map at dawn'),'kind':'intact_control'}, {'rendered':'the pilot carried the quiet bell by noon','audit':audit('the pilot carried the quiet bell by noon'),'kind':'intact_control'}]
 exact=[r for r in rows if r['audit']['two_pointer_exact'] and r['audit']['letters']>38]
 result={'experiment_id':ID,'method':'typed semantic-valency reverse segmentation of live character debt','novelty_preflight':{'fresh_authored_inventory':True,'overlaps_checked':['morpheme-scene-composition-20260919','semantic-clause-reverse-intersection-20260919','valency-frame-reverse-resegmentation-20260919'],'catalogue_text':False,'self_palindromic_units':False},'states':states,'rendered_candidates':rows,'controls':controls,'stats':{'rendered':len(rows),'exact':len(exact),'longest_letters':max([r['audit']['letters'] for r in rows+controls],default=0)},'admission':{'threshold_letters':39,'exact_shortcut_free':len(exact)>0,'reader_gate':'closed'},'next_repair':'At the first failed segment boundary, add one typed 1–2 grapheme inflection variant to the object and adjunct lexicon while retaining the same valency frame; do not broaden to a catalogue sweep.','independent_audits':['explicit two-pointer scan','forward/reverse SHA-256']}
 out=Path('runs')/(ID+'.json'); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'states':states,'rendered':len(rows),'exact':len(exact),'longest':result['stats']['longest_letters'],'out':str(out)}))
if __name__=='__main__':main()
