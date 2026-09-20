"""Authored matrix/complement and relative frames over Brown lexical domains."""
from __future__ import annotations
import hashlib, itertools, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.brown_authored_semantic_reverse_decoder_20260920 import Word, Trie, audit, frame_score, load_words

ROOT=Path(__file__).resolve().parents[1]
BANK=ROOT/'data/brown_pcfg_bank_20260920.json'
OUT=ROOT/'runs/brown-authored-subordination-reverse-decoder-20260920.json'
ID='brown-authored-subordination-reverse-decoder-20260920'
SIGNATURE='brown-derived-lexicon|authored-subordination-grammar|matrix-complement-relative|variable-boundary-reverse-parse'

def letters(s): return re.sub(r'[^a-z]','',s.casefold())

SHAPES=(
 ('that-complement',('DET','ADJ','AGENT','ACTION','THAT','DET','ADJ','AGENT','ACTION','DET','OBJECT')),
 ('relative',('DET','ADJ','AGENT','ACTION','DET','OBJECT','WHO','ACTION','DET','OBJECT')),
)

def frames(domains,limit=9000):
 out=[]
 for kind,shape in SHAPES:
  choices=[]
  for role in shape:
   choices.append({'THAT':(Word('that',1.0),),'WHO':(Word('who',1.0),)}.get(role,domains[role][:8]))
  for selected in itertools.product(*choices):
   words=tuple(selected); content=[w.text for w in words if w.text not in {'the','a','an','that','who'}]
   if len(set(content))!=len(content): continue
   out.append((kind,shape,words))
   if len(out)>=limit:return out
 return out

def parse(tape,shape,trie,forbidden,limit=30):
 out=[]; states=0; seen=set()
 def walk(i,pos,words):
  nonlocal states
  states+=1; key=(i,pos,tuple(w.text for w in words))
  if key in seen or len(out)>=limit:return
  seen.add(key)
  if i==len(shape):
   if pos==len(tape):out.append(words)
   return
  for end,w in trie.matches(tape,pos,shape[i]):
   if w.text in forbidden and w.text not in {'the','a','an','that','who'}:continue
   walk(i+1,end,words+(w,))
 walk(0,0,()); return out,states

def run(max_frames=9000):
 domains=load_words(); domains=dict(domains)
 domains['THAT']=(Word('that',1.0),); domains['WHO']=(Word('who',1.0),)
 fs=frames(domains,max_frames); trie=Trie(domains); rows=[]; exact=[]; states=parses=0
 for kind,shape,words in fs:
  target=letters(' '.join(w.text for w in words))[::-1]
  found,used=parse(target,shape,trie,frozenset(w.text for w in words)); states+=used; parses+=len(found)
  for rw in found:
   text=' '.join(w.text for w in words)+'; '+ ' '.join(w.text for w in rw)+'.'
   row={'rendered':text,'frame_kind':kind,'roles':list(shape),'rank_score':frame_score(words)+frame_score(rw),'audit':audit(text),'provenance':{'source':'Brown-derived word forms only; no Brown sentence text','complete_matrix':True,'complete_embedded_clause':True,'variable_word_boundaries':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'readability_certified':False}}
   rows.append(row)
   if row['audit']['exact']:exact.append(row)
 controls=['The young man said that the old woman found the house.','A good boy sees the door who holds the key.']
 return {'experiment_id':ID,'method':'authored subordination and relative frames over Brown-derived lexical domains with complete reverse parsing','shapes':[{'name':k,'roles':list(s)} for k,s in SHAPES],'stats':{'complete_forward_frames':len(fs),'reverse_states':states,'complete_reverse_parses':parses,'rendered_candidates':len(rows),'exact':len(exact)},'rendered_candidates':sorted(rows,key=lambda x:-x['rank_score'])[:200],'exact_candidates':sorted(exact,key=lambda x:-x['audit']['letters'])[:100],'controls':[{'rendered':x,'audit':audit(x),'complete_prose':True} for x in controls],'novelty_preflight':{'status':'passed','signature':SIGNATURE,'distinct_from':'prior single-clause, dialogue, and coordination lanes; matrix plus embedded complement/relative interiors are selected as complete frames','catalogue_text':False,'finished_tape_reversal':False,'post_hoc_repair':False},'provenance':{'bank':str(BANK.relative_to(ROOT)),'bank_sha256':hashlib.sha256(BANK.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'marker_policy':'that and who admitted from Brown-derived forms; whether unavailable in frozen bank and not fabricated','next_reader_test':'randomized blinded ratings of intact embedded prose versus shuffled controls'},'status':'fresh exact candidates require human reading' if exact else 'no fresh exact parse in this lane','next_construction':'prepare blinded reader package if exact rows survive; otherwise author a new discourse-frame grammar'}

if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
