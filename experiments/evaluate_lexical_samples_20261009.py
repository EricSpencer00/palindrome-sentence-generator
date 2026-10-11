"""Individual transparent automatic diagnostics; never human/model ratings.

These role/context/repetition scores are auditable selection aids. A model
or human reading the rendered text is a separate review, not inferred here.
"""
import gzip,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
from llm_palindrome.admission import normalize_letters as norm
from experiments.bidirectional_lexical_completion_20261009 import frames,enumerate_sentences

ROOT=Path('research/block-seams')
OUT=ROOT/'lexical-completion-review-007'
RUNS=['bidirectional-lexical-005','food-argument-lexical-006']
NAMES={'Anna','Eve','Noel','Leon','Liam','Ma','Otto'}

def assess(row):
    ss=row['sentences'];texts=[s['text'] for s in ss];n=len(ss)
    repeated=n-len(set(texts));frame_names=[s['frame'] for s in ss]
    entities=[];context=[];self_events=[];food=[];stressed=[];events=[]
    for i,s in enumerate(ss):
        words=s['words'];roles=s['roles'];roles_words=defaultdict(list)
        for word,role in zip(words,roles):roles_words[role].append(word)
        entities.append(set(words)&NAMES)
        actor=(roles_words['human_agent'] or roles_words['agent'] or roles_words['addressee'] or roles_words['human_plural_agent'])
        theme=(roles_words['human_theme'] or roles_words['deliverable_theme'] or roles_words['food_theme'] or roles_words['animate_theme'] or roles_words['theme'])
        predicate=(roles_words['imperative'] or roles_words['past_verb'] or roles_words['verb'] or roles_words['copula'])
        events.append(dict(sentence=i,actor=actor,predicate=predicate,theme=theme,recipient=roles_words['recipient'],human_modifier=roles_words['human_adjective']))
        if s['frame'] in ('person_delivery_request','reference_request'):
            context.append(dict(sentence=i,issue='transport destination omitted' if s['frame']=='person_delivery_request' else 'referral destination omitted'))
        if s['frame']=='recipient_delivery_request':context.append(dict(sentence=i,issue='deliver-recipient-object double-object wording is grammatical but relatively marked'))
        if s['frame']=='celebrity_revile':context.append(dict(sentence=i,issue='stars has intended celebrity sense; rendered text alone is ambiguous'))
        if s['frame'] in ('past_revile','stressed_criticism','person_delivery_request','reference_request') and actor and theme and actor[0]==theme[-1]:
            self_events.append(dict(sentence=i,person=actor[0],predicate=predicate[0] if predicate else None))
        if s['frame']=='food_delivery_request':food.extend(roles_words['recipient'])
        if s['frame']=='stressed_criticism':stressed.extend(roles_words['human_agent'])
    shared_adjacent=sum(bool(a&b) for a,b in zip(entities,entities[1:]));food_links=sorted(set(food)&set(stressed))
    tokens=re.findall(r'[a-z]+',row['text'].lower());bigram_count=Counter(zip(tokens,tokens[1:]));repeat_bigrams=sum(c-1 for c in bigram_count.values() if c>1)
    structural_repeat=n-len(set(frame_names))
    # Syntax score reflects licensed slot paths only, not a human assessment.
    grammar=4
    readability=4 if n==1 and not context and not self_events else 3
    if self_events or len(context)>=max(2,n):readability=2
    if n==1:coherence=3 if not self_events else 1
    elif repeated:coherence=1
    elif row['known_scaffold'] and len(set(frame_names))>1:coherence=0
    elif food_links:coherence=2
    elif shared_adjacent==n-1 and not self_events:coherence=2
    else:coherence=1
    if repeated>=max(1,n//2):repetition=0 if n>2 else 1
    elif repeated:repetition=1
    elif repeat_bigrams>=max(4,len(tokens)//3):repetition=2
    elif structural_repeat or repeat_bigrams:repetition=3
    else:repetition=4
    provenance=0 if row['known_scaffold'] else 2
    explanation=[f'{n} complete licensed clause(s); {len(set(frame_names))} structural family/families; {repeated} identical repeated sentence(s).',f'{shared_adjacent}/{max(0,n-1)} adjacent pairs share a named participant.']
    if context:explanation.append('Context dependencies: '+', '.join(f"sentence {c['sentence']+1}: {c['issue']}" for c in context)+'.')
    if self_events:explanation.append('Self-directed actions require interpretation: '+', '.join(f"{x['person']} {x['predicate']} themself" for x in self_events)+'.')
    if food_links:explanation.append('Food recipient and later stressed subject match: '+', '.join(food_links)+'. The text leaves any causal relation implicit.')
    if n>1 and not food_links:explanation.append('Shared names alone do not establish a causal or narrative connection.')
    explanation.append('Known scaffold/control.' if row['known_scaffold'] else 'Authored grammatical construction using previously available reversed lexical material; no broad novelty or human approval claim.')
    return dict(grammar=grammar,readability=readability,coherence=coherence,repetition=repetition,provenance=provenance,explanation=' '.join(explanation),events=events,features=dict(shared_adjacent=shared_adjacent,food_stress_link_people=food_links,identical_sentence_repeats=repeated,repeated_bigrams=repeat_bigrams,structural_family_repeats=structural_repeat,context_dependencies=context,self_directed_actions=self_events),rating_kind='automatic individual role/context/repetition diagnostic; not independent model or human quality',human_label=None)

def structural_key(row):
    return tuple((s['frame'],tuple('<human>' if w in NAMES else w.lower() for w in s['words'])) for s in row['sentences'])

def verify_raw(run,arm,specs):
    count=0;seen=Counter();letters=0
    with gzip.open(ROOT/run/(arm['method']+'-outputs.jsonl.gz'),'rt') as f:
        for line in f:
            row=json.loads(line);t=norm(row['text']);assert t==t[::-1] and t==row['tape'];assert row['letters']==len(t)
            for s in row['sentences']:
                spec=specs[s['frame_index']]
                assert spec.name==s['frame'] and len(s['words'])==len(spec.slots)
                assert all(w in slot.words for w,slot in zip(s['words'],spec.slots))
                assert spec.render(s['words'])==s['text']
            assert ' '.join(s['text'] for s in row['sentences'])==row['text']
            count+=1;seen[t]+=1;letters+=len(t)
    assert count==arm['exact_derivations']
    assert seen==Counter({r['tape']:r['derivation_count'] for r in arm['outputs']})
    return dict(verified_raw_derivations=count,verified_unique_texts=len(seen),verified_letter_comparisons=letters,exactness=1.0)

def run():
    OUT.mkdir(parents=True,exist_ok=True);ratings=[];exports={};summary=[];all_pool={};checks=[]
    for name in RUNS:
        result=json.loads((ROOT/name/'results.json').read_text());specs=frames(include_food=name.startswith('food-'))
        for arm in result['arms']:
            # Original-bank arm uses only the initial frames, which share the
            # same frame indexes in the independently built larger grammar.
            check=verify_raw(name,arm,specs);checks.append(dict(run=name,method=arm['method'],**check))
            sample_ids={r['id'] for r in arm['sample']};assert len(sample_ids)==len(arm['sample'])
            assert sum(s['sample_count'] for s in arm['sampling_manifest'])==len(arm['sample'])
            for manifest in arm['sampling_manifest']:assert manifest['sample_count']==__import__('math').ceil(manifest['denominator']/10)
            distribution={key:Counter() for key in ('grammar','readability','coherence','repetition','provenance')}
            for row in arm['sample']:
                assessment=assess(row);rating=dict(run=name,method=arm['method'],id=row['id'],text=row['text'],letters=row['letters'],clause_count=row['clause_count'],exact=True,**assessment)
                ratings.append(rating)
                for key in distribution:distribution[key][assessment[key]]+=1
                aid=hashlib.sha256(row['tape'].encode()).hexdigest()[:16]
                if aid not in exports:exports[aid]=dict(assessment_id=aid,text=row['text'],letters=row['letters'],clauses=row['clause_count'],occurrences=[],human_label=None)
                exports[aid]['occurrences'].append(dict(run=name,method=arm['method'],id=row['id']))
            output_names=Counter(tuple(r['frames']) for r in arm['outputs']);structures={structural_key(r) for r in arm['outputs']}
            summary.append(dict(run=name,method=arm['method'],unique_output_count=len(arm['outputs']),sample_count=len(arm['sample']),sample_percent=100*len(arm['sample'])/len(arm['outputs']),sample_score_distribution={k:dict(sorted(v.items())) for k,v in distribution.items()},independent_model_ratings=0,new_human_ratings=0,frame_set_distribution={str(k):v for k,v in output_names.items()},structural_patterns_after_masking_human_names=len(structures),identical_sentence_repeat_distribution=dict(sorted(Counter(r['repeated_sentences'] for r in arm['outputs']).items())),exact_closure_frame_coverage=sorted({s['frame'] for r in arm['outputs'] for s in r['sentences']})))
            if arm['method']=='bidirectional_expanded_grammar':
                for row in arm['outputs']:all_pool[row['tape']]=dict(run=name,**row,automatic_diagnostic=assess(row))
    (OUT/'individual-sample-diagnostics.json').write_text(json.dumps(dict(scale='0 worst to 4 best; grammar=role-path syntax, readability/context and coherence/shared-participant heuristics, repetition 4=least repeated, provenance 0=copied classic scaffold 2=authored frame using existing reversal material; scores are NOT human or independent model ratings',ratings=ratings,missing_diagnostics=0),indent=2)+'\n')
    (OUT/'independent-review-export.json').write_text(json.dumps(dict(instructions='Read every text individually for grammar, readability, coherence and repetition; do not infer quality from exactness, known source, or automatic scores. Return per-assessment-id scores 0-4 with reasons. This export is an unreviewed sample, not an approval request. Earlier five BAD labels and readability-positive/repetition-concern on sees example remain separate.',items=list(exports.values()),unique_texts=len(exports),method_sample_occurrences=len(ratings),independent_model_review_completed=False),indent=2)+'\n')
    overall=dict(sampled_occurrences=len(ratings),unique_sampled_texts=len(exports),unique_method_output_denominator=sum(s['unique_output_count'] for s in summary),overall_percent=100*len(ratings)/sum(s['unique_output_count'] for s in summary),union_unique_exact_pool=len(all_pool),independent_model_ratings=0,human_new_output_approvals=0)
    (OUT/'quality-distribution.json').write_text(json.dumps(dict(overall=overall,arms=summary,verification=checks),indent=2)+'\n')
    # Rank the entire union as a selection aid; preserve all outputs and do
    # not confuse this posthoc selection with the fixed random sample.
    ranked=sorted(all_pool.values(),key=lambda x:(-x['automatic_diagnostic']['coherence'],x['known_scaffold'],x['repeated_sentences'],len(x['automatic_diagnostic']['features']['self_directed_actions']),-x['automatic_diagnostic']['repetition'],-len(x['frames']),abs(x['letters']-70),x['text']))
    with gzip.open(OUT/'ranked-pool.jsonl.gz','wt') as f:
        for row in ranked:f.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps(overall))
    for s in summary:print(s['run'],s['method'],'structural patterns',s['structural_patterns_after_masking_human_names'],'scores',s['sample_score_distribution'])

if __name__=='__main__':run()
