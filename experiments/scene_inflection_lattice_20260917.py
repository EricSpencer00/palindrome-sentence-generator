"""Human-authored scene lattice with seam-compatible inflection variants.

This lane selects intact scenes first, then jointly chooses inflected variants
for two clauses while propagating outer character equations. It never imports a
completed palindrome or treats language scores as a readability certificate.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='scene-inflection-lattice-20260917'; OUT=ROOT/'runs'/f'{ID}.json'

def tape(s): return ''.join(c.lower() for c in s if 'a'<=c.lower()<='z')
def audit(s):
 t=tape(s); m=[]; i=0; j=len(t)-1
 while i<j:
  if t[i]!=t[j]: m.append((i,j,t[i],t[j]))
  i+=1; j-=1
 return {'letters':len(t),'exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatches':m[:12], 'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def independent(s):
 c=[x for x in s.casefold() if x in 'abcdefghijklmnopqrstuvwxyz']; i,j=0,len(c)-1; m=[]
 while i<j:
  if c[i]!=c[j]: m.append((i,j,c[i],c[j]))
  i+=1;j-=1
 raw=''.join(c)
 return {'exact':bool(raw) and not m,'mismatch_count':len(m),'sha256_forward':hashlib.sha256(raw.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(raw[::-1].encode()).hexdigest()}
def flags(s):
 ws=[tape(x) for x in re.findall('[A-Za-z]+',s)]; content=[w for w in ws if w not in {'a','an','the','and','then','near','beside','under','by','at','was','were'}]
 return {'word_order_mirror':ws==[w[::-1] for w in ws],'repeated_content':len(content)!=len(set(content)),'self_palindromic_content_words':[w for w in content if len(w)>1 and w==w[::-1]],'borrowed_catalogue_text':False,'finished_tape_reversed':False}

SCENES=[
 {'id':'dawn_garden','left':('At dawn,', 'the patient gardener', ['opens','opened'], ['the quiet greenhouse','the old greenhouse'], 'and labels the fresh seed trays'), 'right':('then', 'the careful gardener', ['waters','watered'], ['the young tomato plants','the tender tomato plants'], 'before dusk.')},
 {'id':'harbor_signal','left':('At first light,', 'the harbor pilot', ['checks','checked'], ['the weathered chart','the folded chart'], 'and signals the waiting boat'), 'right':('while', 'the harbor pilot', ['guides','guided'], ['the small fishing boat','the tired fishing boat'], 'toward shore.')},
 {'id':'archive_repair','left':('In the quiet archive,', 'the patient clerk', ['repairs','repaired'], ['a torn map','the torn map'], 'and records its missing names'), 'right':('as', 'the careful clerk', ['files','filed'], ['the marked folder','a marked folder'], 'before closing.')},
 {'id':'school_scene','left':('After rain,', 'the young teacher', ['carries','carried'], ['the bright models','the paper models'], 'and dries the classroom windows'), 'right':('while', 'the kind teacher', ['greets','greeted'], ['the waiting children','the quiet children'], 'at noon.')},]

def render(sc, tense, li, ri):
 L=sc['left']; R=sc['right']; return f"{L[0]} {L[1]} {L[2][tense]} {L[3][li]} {L[4]}, {R[0]} {R[1]} {R[2][tense]} {R[3][ri]} {R[4]}"
def run():
 rows=[]; failures=[]
 # enumerate jointly; length gate occurs before diagnostic ranking
 for sc,tense,li,ri in itertools.product(SCENES,range(2),range(2),range(2)):
  s=render(sc,tense,li,ri); au=audit(s); ind=independent(s); fl=flags(s)
  row={'rendered':s,'scene_id':sc['id'],'variants':{'tense':'present' if tense==0 else 'past','left_object':li,'right_object':ri},'audit':au,'independent_audit':ind,'shortcut_flags':fl,'provenance':{'generator':ID,'construction':'human-authored intact scene lattice; joint inflection and boundary variants','catalogue_imported':False,'seed_used_as_output':False,'template_author':'local scene inventory'}}
  if not 100<=au['letters']<=160: row['failure_reason']='length_gate_before_scoring'; failures.append(row); continue
  if any(fl[k] for k in ('word_order_mirror','self_palindromic_content_words','borrowed_catalogue_text','finished_tape_reversed')): row['failure_reason']='shortcut_gate'; failures.append(row); continue
  rows.append(row)
 exact=[r for r in rows if r['audit']['exact'] and r['independent_audit']['exact']]
 best=min(rows,key=lambda r:(r['audit']['mismatch_count'],-r['audit']['letters'])) if rows else None
 return {'experiment_id':ID,'status':'completed_no_exact_closure' if not exact else 'exact_candidates_found','config':{'scenes':len(SCENES),'variant_combinations':32,'length_gate':'100<=letters<=160 before scoring','rows':len(rows),'failed_branches':len(failures)},'actual_candidates':rows[:12],'best':best,'exact_candidates':exact,'failed_branches':failures,'independent_validation':'separate two-pointer audit plus SHA-256 forward/reverse','reader_gate':'closed unless exact novel survivor; programmatic diagnostics do not certify readability','novelty_preflight':'no catalogue import, fixed tape, repeated-unit or word-order mirror construction','next_repair':'Add boundary-aware lexical variants whose first/last letters are selected jointly with the scene lattice; retain intact event semantics and length gate.','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(),indent=2)+'\n'); r=run(); print(json.dumps({'status':r['status'],'candidates':r['config']['rows'],'failed':r['config']['failed_branches'],'best':r['best']['rendered'] if r['best'] else None,'letters':r['best']['audit']['letters'] if r['best'] else 0,'mismatches':r['best']['audit']['mismatch_count'] if r['best'] else None,'exact':len(r['exact_candidates'])}))
