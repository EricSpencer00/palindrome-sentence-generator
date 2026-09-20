"""Contemporary two-clause discourse with pre-bound anaphoric references."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/anaphoric-two-clause-discourse-20260920.json';ID='anaphoric-two-clause-discourse-20260920';SIG='contemporary-two-clause|shared-referent-binding|anaphoric-pronoun-realization|live-residual-equations'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
@dataclass(frozen=True)
class Referent:name:str;pronoun:str;determiner:str
REFERENTS=(Referent('the analyst','they','the'),Referent('the teacher','she','the'),Referent('the engineer','they','the'),Referent('the doctor','he','the'),Referent('the writer','she','the'),Referent('the pilot','he','the'),Referent('the designer','they','the'),Referent('the student','they','the'))
VERBS=(('reviews','the report'),('opens','the file'),('checks','the schedule'),('writes','a note'),('tests','the system'),('calls','the team'),('reads','the message'),('plans','the meeting'))
LINKERS=('and','so','because','then')
PLURAL_VERBS={'reviews':'review','opens':'open','checks':'check','writes':'write','tests':'test','calls':'call','reads':'read','plans':'plan'}
def bound_frames():
 for ref in REFERENTS:
  for verb,obj in VERBS:
   for link in LINKERS:
    # Binding is explicit: the second subject is realized from this referent,
    # never selected independently after character constraints are known.
    left=f'{ref.name} {verb} {obj}'
    right_verb=PLURAL_VERBS[verb] if ref.pronoun=='they' else verb
    right=f'{ref.pronoun} {right_verb} {obj}'
    yield {'referent':ref,'linker':link,'left':left,'right':right,'text':f'{left} {link} {right}.'}
def compatible(left,right):
 a=letters(left);b=letters(right)[::-1]
 return all(x==y for x,y in zip(a,b))
def run():
 rows=[];states=0
 for frame in bound_frames():
  # The relation and shared referent are fixed before this residual check.
  states+=1; ok=compatible(frame['left']+' '+frame['linker'],frame['right'])
  if ok:
   row={'rendered':frame['text'],'audit':audit(frame['text']),'binding':{'referent':frame['referent'].name,'pronoun':frame['referent'].pronoun,'definite_reference':frame['referent'].determiner},'relation':frame['linker'],'complete_left_clause':True,'complete_right_clause':True,'provenance':{'prebound_anaphora':True,'live_residual_equations':True,'contemporary_english':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}}
   rows.append(row)
 exact=[x for x in rows if x['audit']['exact']]
 controls=[x['text'] for x in bound_frames()][:24]
 registry=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text())
 collision=any(x.get('id') != ID and x.get('signature') == SIG for x in registry.get('entries',[])+registry.get('excluded',[]))
 return {'experiment_id':ID,'method':'plain contemporary two-clause discourse with pre-bound anaphoric pronouns','stats':{'referents':len(REFERENTS),'verb_object_pairs':len(VERBS),'linkers':len(LINKERS),'bound_frames':states,'residual_compatible_frames':len(rows),'exact':len(exact),'controls':len(controls)},'rendered_candidates':rows,'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x),'complete_prose':True} for x in controls],'novelty_preflight':{'status':'collision' if collision else 'passed','signature':SIG,'registry_entries_checked':len(registry.get('entries',[])),'distinct_from':'prior Shakespearean, phrase-pair, envelope, and event-graph lanes; shared contemporary referents bind pronouns across two discourse clauses','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'source':'new authored contemporary clause inventory','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed; programmatic exactness does not certify readability','next_reader_test':'blinded intact discourse versus shuffled controls'},'status':'fresh exact candidates require human reading' if exact else 'no exact anaphoric discourse closure','next_construction':'add definite NP versus pronoun alternation with discourse relation constraints'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
