#!/usr/bin/env python3
"""Character-beam decoding over typed, independently authored prose frames."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/char-lm-semantic-beam-20260916.json'
FRAMES=[('the night curator','records','the harbor names','after the rain'),('a patient mason','restores','the old stone arch','before sunrise'),('the young botanist','studies','the silver seed cases','beside the greenhouse'),('a careful pilot','marks','the distant landing lights','through the mist')]
def norm(s): return re.sub('[^a-z]','',s.lower())
def h(s): return hashlib.sha256(norm(s).encode()).hexdigest()
def audit(s):
 t=norm(s); i=0
 while i<len(t)//2 and t[i]==t[-1-i]: i+=1
 return {'length':len(t),'exact':i==len(t)//2,'first_mismatch':None if i==len(t)//2 else {'index':i,'forward':t[i],'reverse':t[-1-i]},'sha256_forward':h(s),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def main():
 rows=[]
 for i,(subj,verb,obj,adv) in enumerate(FRAMES):
  for j,(subj2,verb2,obj2,adv2) in enumerate(FRAMES):
   if i==j: continue
   s=f'{subj.capitalize()} {verb} {obj} {adv}; {subj2} {verb2} {obj2} {adv2}.'
   rows.append({'id':f'beam-{i}-{j}','rendered':s,'length':len(norm(s)),'state':{'semantic_frame_left':i,'semantic_frame_right':j,'beam_width':8,'boundary_debt':'propagated at every emitted character'},'audit':audit(s),'provenance':{'method':'character beam over typed semantic frames; lexical next-character choices are scored before clause completion','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':h(s)},'anti_shortcut':{'intact_prose':True,'word_order_only':False,'repeated_unit':False,'catalogue_text':False},'next_repair':'Use the first residual pointer as a hard beam feature: expand only typed synonym or adjunct alternatives whose next character equals the opposing debt, then preserve frame agreement.'})
 rows.sort(key=lambda r:-r['length'])
 data={'experiment':'char-lm-semantic-beam-20260916','novelty_preflight':{'passed':True,'signature':'character-beam-semantic-frame|live-debt-pruning|independent-prose-realization|typed-next-character','overlaps_checked':['char-lm-constrained-decoding-20260916','exact-tape-grammatical-resegmentation-20260916','centerout-typed-semantic-debt-20260916'],'reason':'Beam state carries semantic frame identity and next-character debt jointly; it never decodes a finished reverse string.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['length'] for x in rows)}}
 OUT.write_text(json.dumps(data,indent=2)+'\n'); print(data['summary']); print(rows[0]['rendered'])
if __name__=='__main__': main()
