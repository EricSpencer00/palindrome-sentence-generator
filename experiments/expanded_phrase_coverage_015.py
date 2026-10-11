"""Substantive finite typed vocabulary search, no mirrored sentence chain."""
import gzip,hashlib,json,math,time,zipfile
from pathlib import Path
from dataclasses import asdict
from llm_palindrome.bidirectional_lexical import Frame,Slot,GrammarDAG,exact_grammar_palindromes,SearchBudgetExceeded
from llm_palindrome.admission import normalize_letters as norm
OUT=Path('research/block-seams/expanded-phrase-015')
def slot(role,*words):return Slot(role,tuple(words))
def frames():
 fs=[]
 def add(name,slots):fs.append(Frame(name,tuple(slots),provenance='authored typed phrase grammar; no completed sentence bank or sentence mirroring'))
 people=slot('human_theme','Mom','Dad');agent=slot('human_agent','Mom','Dad')
 counts=slot('observed_count_theme','car','cat','rat','wart','cart','art','drawer','diaper','door','gate','dog','ward','pot','pet','boat','civic','kayak','racecar','level','star','nurse','rider','doctor','teacher','sailor','tar')
 mods=slot('adjective','red','raw','live','evil','stressed','tired','sad','able','level','selfless','odd','mild','old','new','small','large','white','black','calm','warm','cold','hot','good','bad','clear','dark')
 opening=[slot('question_copula','was'),slot('question_subject','it'),slot('determiner','a')]
 ending=[slot('relative_agent','I'),slot('past_observation','saw')]
 add('observed_count_question',opening+[counts]+ending)
 add('modified_observation_question',opening+[mods,counts]+ending)
 add('double_modified_question',opening+[mods,mods,counts]+ending)
 add('alternative_observation_question',opening+[counts,slot('coordinator','or'),slot('determiner','a'),counts]+ending)
 add('modified_alternative_question',opening+[mods,counts,slot('coordinator','or'),slot('determiner','a'),mods,counts]+ending)
 add('plural_observation_question',[slot('question_copula','was'),slot('question_subject','it'),slot('plural_theme','desserts','pets','pots','rats','stars'),slot('relative_agent','I'),slot('past_observation','saw')])
 # Same agent throughout each coordinated/cause frame. Role-licensed objects
 # and tense agreement are authored; mixed plans are not actor-name swaps.
 predicates=[('caregiving',slot('past_verb','repaid','diapered','helped','stopped','spotted','reviled'),people),('household',slot('past_verb','opened','closed','delivered','cleaned'),slot('deliverable_theme','mail','desserts','pots','pets')),('drawing',slot('past_verb','drew','rewarded'),slot('human_or_drawing_theme','Mom','Dad','drawer'))]
 for name,verb,obj in predicates:
  add(name+'_past',[agent,verb,obj])
  add(name+'_coordinated',[agent,verb,obj,slot('coordinator','and'),verb,obj])
  add(name+'_causal',[agent,verb,obj,slot('causal_connector','because'),agent,verb,obj])
  add(name+'_relative',[agent,verb,obj,slot('relative_pronoun','who'),slot('past_verb','helped','stopped','spotted','reviled'),people]) if name=='caregiving' else None
 add('caregiving_present',[agent,slot('verb','stops','spots','helps','diapers','repays'),people])
 add('caregiving_present_causal',[agent,slot('verb','stops','spots','helps','diapers','repays'),people,slot('causal_connector','because'),agent,slot('verb','stops','spots','helps','diapers','repays'),people])
 add('firstperson_observation_cause',[slot('human_agent','I'),slot('past_verb','saw'),counts,slot('causal_connector','because'),agent,slot('past_verb','helped','stopped','spotted'),people])
 add('contrastive_state',[slot('human_agent','I'),slot('copula','am'),mods,slot('contrast','but'),slot('human_agent','Mom','Dad'),slot('copula','is'),mods])
 add('single_comparative',[slot('human_agent','Ma'),slot('copula','is'),slot('degree','as'),mods,slot('degree','as'),slot('comparison_agent','I'),slot('copula','am')])
 add('requested_reward',[slot('imperative','reward'),slot('determiner','a'),slot('human_theme','drawer')])
 add('requested_drawing',[slot('imperative','draw'),slot('determiner','a'),slot('drawing_theme','ward','cart','car','cat','rat')])
 return fs

def run():
 OUT.mkdir(parents=True,exist_ok=True);t=time.monotonic();specs=frames();lex=set(Path('data/lexicon.txt').read_text().splitlines());words=sorted({w for f in specs for s in f.slots for w in s.words});reversepairs=[(w,w[::-1]) for w in sorted(lex) if w!=w[::-1] and w[::-1] in lex and w<w[::-1]]
 def save(n,o):(OUT/n).write_text(json.dumps(o,indent=2)+'\n')
 save('plan.json',dict(typed_frames=len(specs),max_work=2000000,max_seconds=8,max_paths=5000,one_grammar_sentence_only=True,no_sentence_mirroring=True,local_models=0,paid_apis=0,stop='One complete expanded grammar product, no adaptive follow-up',sampling='ceil10%percompound_vs_simpleframe/copied_to010stratum',quality='sole-owner selected reading only; no independent or human claim'))
 save('frames.json',[asdict(f) for f in specs]);save('lexicon-audit.json',dict(wordlist_sha256=hashlib.sha256(Path('data/lexicon.txt').read_bytes()).hexdigest(),lexicon_headwords=len(lex),reverse_lexical_pairs=len(reversepairs),pairs=reversepairs,typed_words=words,absent_headwords=[w for w in words if w.lower() not in lex],licensing='typed domains authored; past/plural inflections and ordinary role words explicitly listed, not inferred grammatical from spelling reversal',notable_new_pairs=['diaper/repaid','spots/stops','draw/ward','drawer/reward'],semantic_limit='drawing ward and reward drawer are known controls; comparative modifiers and compound-role grammar still require readability judgment'))
 g=GrammarDAG(specs,1);raw=[]
 with gzip.open(OUT/'states.jsonl.gz','wt') as trace:
  try:paths,receipt=exact_grammar_palindromes(g,max_work=2000000,max_paths=5000,seconds=8,trace=lambda r:trace.write(json.dumps(r)+'\n'))
  except SearchBudgetExceeded as e:paths=[];receipt=e.receipt
 for path in paths:
  item=g.materialize(path);item['id']='expanded015-'+hashlib.sha256(item['tape'].encode()).hexdigest()[:16];item['character_arc_ids']=path;raw.append(item)
 save('receipt.json',receipt);save('raw-outputs.json',raw)
 old={json.loads(x)['tape'] for x in gzip.open('research/block-seams/semantic-assembly-010/unique-outputs.jsonl.gz','rt')};unique={r['tape']:r for r in raw}
 for r in unique.values():r['new_to010']=r['tape'] not in old;r['quality_unrated']=True
 save('unique-outputs.json',list(unique.values()))
 language=sum(math.prod(len(s.words) for s in f.slots) for f in specs);framecounts={f.name:sum(r['sentences'][0]['frame']==f.name for r in raw) for f in specs}
 from collections import defaultdict
 strata=defaultdict(list)
 for r in raw:strata[(any(c in r['sentences'][0]['frame'] for c in ('causal','coordinated','relative','contrast')),r['tape'] in old)].append(r)
 sample=[];manifest=[]
 for st,rs in sorted(strata.items()):
  k=math.ceil(len(rs)/10);chosen=sorted(rs,key=lambda r:hashlib.sha256(('expanded015'+r['id']).encode()).hexdigest())[:k];sample+=chosen;manifest.append(dict(stratum=st,population=len(rs),count=k))
 save('sample.json',sample);save('sample-manifest.json',manifest)
 save('graph.json',dict(start=g.start,accept=g.accept,arcs=[asdict(a) for a in g.arcs],epsilon={str(k):sorted(v) for k,v in g.eps.items() if v}))
 maxlen=max(sum(max(len(norm(w)) for w in s.words) for s in f.slots) for f in specs)
 summary=dict(frames=len(specs),typed_words=len(words),finite_grammatical_derivations=language,all_derivations_decided=receipt['complete'],exact_raw_derivations=len(raw),unique_outputs=len(unique),failed_exact_derivations=language-len(raw) if receipt['complete'] else None,new_unique_to010=sum(r['new_to010'] for r in unique.values()),frame_yield=framecounts,grammar_max_length=maxlen,actual_max_length=max((r['letters'] for r in raw),default=0),outputs_ge60=sum(r['letters']>=60 for r in raw),operations=receipt['work'],elapsed_seconds=time.monotonic()-t,sample_occurrences=len(sample),sample_denominator=len(raw),sample_percent=100*len(sample)/len(raw) if raw else None,independent_scores=0,human_labels=0)
 save('summary.json',summary);save('checkpoint.json',dict(status='complete' if receipt['complete'] else 'bounded_incomplete',summary=summary,next='Judge selected exact shortlist against010; compound phrase grammar lexical closure remains limiting, not a universal impossibility.'))
 print(json.dumps(summary,indent=2));print('\n'.join(r['text'] for r in unique.values()))
if __name__=='__main__':run()
