"""Vocabulary-controlled, complete four-sentence CPU grammar comparison."""
import gzip,hashlib,itertools,json,math,time
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
from llm_palindrome.admission import normalize_letters as norm
from llm_palindrome.bidirectional_lexical import Slot,Frame,GrammarDAG,exact_grammar_palindromes,SearchBudgetExceeded
from llm_palindrome.paragraph_residual_join import exact_residual_pairs,JoinBudgetExceeded
from llm_palindrome.typed_constituents import NAMES

OUT=Path('research/block-seams/bidirectional-lexical-005')
SEED='bidirectional-lexical-005'
WORK=20000000
SECONDS=30
PATHS=20000

def slot(role,*words):return Slot(role,tuple(words))

def frames(include_food=False):
    out=[]
    def frame(name,slots,style='statement',provenance='authored role grammar',known=False):
        out.append(Frame(name,tuple(slots),style,provenance,known))
    # Same finite language as diagnostic 004, independently factored by slots.
    for subjects,verb in [(['I','we'],'open'),(['Anna','Eve','Otto'],'opens')]:
        frame('open',[slot('agent',*subjects),slot('verb',verb),slot('theme','mail')])
        frame('open',[slot('agent',*subjects),slot('verb',verb),slot('determiner','a'),slot('theme','door','gate')])
    frame('pets',[slot('agent','Anna','Eve','Otto','Noel'),slot('verb','pets'),slot('determiner','a'),slot('animate_theme','cat','dog')])
    frame('pets',[slot('agent','Anna','Eve','Otto','Noel'),slot('verb','pets'),slot('animate_theme','pets')])
    for subjects,verb in [(['I','we'],'say'),(['Anna','Eve'],'says')]:
        frame('say',[slot('speaker',*subjects),slot('verb',verb),slot('quoted_content','yes','no','wow')],style='quote')
    frame('let_in',[slot('imperative','let'),slot('animate_theme','Anna','Eve','Otto','Noel'),slot('particle','in')])
    for subjects,copula in [(['I'],'am'),(['Ma','Anna','Eve','Otto'],'is')]:
        adjectives=slot('predicate_adjective','red','tired','sad','able','selfless','evil','live')
        frame('copula',[slot('subject',*subjects),slot('copula',copula),adjectives])
        frame('comparative',[slot('subject',*subjects),slot('copula',copula),slot('degree','as'),adjectives,slot('degree','as'),slot('comparator','I'),slot('copula','am')],provenance='catalogue v3#526 scaffold; copied control',known=True)
    frame('step',[slot('imperative','step'),slot('preposition','on'),slot('determiner','no'),slot('theme','pets')],provenance='known Step on no pets control',known=True)
    frame('step',[slot('imperative','step'),slot('preposition','on'),slot('determiner','the'),slot('theme','mat')])
    frame('step',[slot('imperative','step'),slot('preposition','on'),slot('determiner','a'),slot('theme','rug')])
    # New grammatical families. These are not palindromic whole-sentence
    # scaffolds. deliver/reviled is an existing reversed lexical pair; keep
    # that provenance separate from full-sentence or paragraph novelty.
    names=slot('addressee','Noel','Leon','Anna','Eve')
    people=slot('human_theme','Noel','Leon','Anna','Eve')
    theme=slot('deliverable_theme','mail','pots','pets','desserts')
    lexical_provenance='authored grammar; existing deliver/reviled lexical pair appeared in earlier rejected material; no preserved word-salad body'
    frame('delivery_request',[names,slot('imperative','deliver'),theme],style='vocative',provenance=lexical_provenance)
    frame('delivery_request',[names,slot('imperative','deliver'),slot('determiner','a'),slot('deliverable_theme','drawer')],style='vocative',provenance=lexical_provenance)
    frame('person_delivery_request',[names,slot('imperative','deliver'),slot('human_adjective','reviled'),people],style='vocative',provenance=lexical_provenance)
    frame('person_delivery_request',[names,slot('imperative','deliver'),people],style='vocative',provenance=lexical_provenance)
    frame('recipient_delivery_request',[names,slot('imperative','deliver'),slot('recipient','Noel','Leon','Liam','Anna','Eve'),slot('deliverable_theme','mail')],style='vocative',provenance=lexical_provenance)
    frame('past_revile',[slot('human_agent','Noel','Leon','Liam','Anna','Eve'),slot('past_verb','reviled'),people],provenance='authored human-agent past-transitive frame; revile+d explicit morphology')
    frame('celebrity_revile',[slot('human_plural_agent','stars'),slot('past_verb','reviled'),people],provenance='authored frame; stars explicitly denotes celebrities, not celestial objects')
    frame('stopping_request',[slot('imperative','stop'),slot('human_adjective','reviled'),people],provenance='authored imperative stop + adjective + human object')
    frame('reference_request',[names,slot('imperative','refer'),people],style='vocative',provenance='authored refer-human frame; destination is context-dependent optional argument')
    frame('delivery_statement',[slot('human_agent','Noel','Leon','Anna','Eve'),slot('verb','delivers'),theme],provenance='authored frame; deliver+s explicit third-person agreement')
    if include_food:
        frame('food_delivery_request',[names,slot('imperative','deliver'),slot('recipient','Noel','Leon','Anna','Eve'),slot('food_theme','desserts')],style='vocative',provenance='new ditransitive deliver-recipient-food frame; desserts/stressed existing lexical reversal; no copied whole body')
        frame('stressed_criticism',[slot('human_adjective','stressed'),slot('human_agent','Noel','Leon','Anna','Eve'),slot('past_verb','reviled'),people],provenance='new affect-adjective human-subject past-transitive frame; stress+ed morphology; source lexical pair desserts/stressed')
    return tuple(out)

def enumerate_sentences(specs):
    rows=[]
    for fi,f in enumerate(specs):
        for words in itertools.product(*(s.words for s in f.slots)):
            rows.append(dict(text=f.render(words),frame=f.name,frame_index=fi,words=words,roles=[s.role for s in f.slots],known_scaffold=f.known_scaffold,provenance=f.provenance))
    return rows

def vocabulary_receipt(specs):
    lexicon=set(Path('data/lexicon.txt').read_text().splitlines());v3=json.loads(Path('data/v3_bank.json').read_text())
    v3words=defaultdict(list)
    for i,row in enumerate(v3):
        for word in row['text'].split():v3words[norm(word)].append(i)
    inflections={'opens':('open','third_singular_s'),'says':('say','third_singular_s'),'pets':('pet','plural_or_third_singular_s'),'pots':('pot','plural_s'),'desserts':('dessert','plural_s'),'reviled':('revile','past_or_adjective_d'),'delivers':('deliver','third_singular_s'),'stars':('star','plural_s'),'stressed':('stress','past_or_adjective_ed')}
    records=[]
    for word in sorted({w for f in specs for s in f.slots for w in s.words}):
        lower=norm(word)
        if lower in inflections:
            lemma,rule=inflections[lower];assert lemma in lexicon
            source=dict(kind='explicit_morphology',lemma=lemma,rule=rule,lemma_in_shipped_lexicon=True)
        elif lower in lexicon:source=dict(kind='shipped_lexicon',headword=lower)
        elif lower in v3words:source=dict(kind='existing_v3_token',indices=v3words[lower])
        elif word in NAMES:source=dict(kind='existing_typed_name_vocabulary',source='llm_palindrome/typed_constituents.py:NAMES',semantic_type='human')
        else:raise AssertionError('unlicensed vocabulary '+word)
        roles=sorted({s.role for f in specs for s in f.slots if word in s.words})
        records.append(dict(word=word,roles=roles,source=source))
    old=json.loads(Path('research/block-seams/nonsees-fragment-004/bank.json').read_text());oldwords={w for r in old for w in r['tokens']}
    return dict(words=records,total_words=len(records),words_absent_from_original_fragment_bank=[r['word'] for r in records if r['word'] not in oldwords],coverage='Explicit finite typed subset of existing lexicon/v3 with listed morphology; omitted words are vocabulary limits, not search pruning',semantic_assumptions={'human_names':'Noel Leon Anna Eve Liam','animals':'pets cats dogs; pet themes exclude human names','deliverable':'mail pots pets desserts drawer; deliver-human separately means escort or transport a person','revile':'human or explicitly celebrity agent; human theme','live_copula':'inherited control vocabulary: live may require broadcast context','reference':'human theme, destination context-dependent'})

def output_row(method,item,derivation):
    text=item['text'];t=norm(text);assert t and t==t[::-1]
    sentences=item['sentences'];texts=[s['text'] for s in sentences]
    return dict(id=method+'-'+hashlib.sha256(t.encode()).hexdigest()[:16],text=text,tape=t,letters=len(t),clause_count=len(sentences),sentences=sentences,frames=sorted({s['frame'] for s in sentences}),known_scaffold=any(s['known_scaffold'] for s in sentences),repeated_sentences=len(texts)-len(set(texts)),derivation=derivation,exact=True,human_label=None)

def enumerated_arm(method,bank):
    start=time.monotonic();deadline=start+SECONDS;raw=[];receipts=[];work=0;status='complete'
    tapes=[norm(r['text']) for r in bank]
    for n in (1,2,3,4):
        if n==1:
            pairs=[(i,None) for i,t in enumerate(tapes) if t==t[::-1]];left=[(i,) for i in range(len(bank))];right=[];receipt=dict(work=len(bank),complete=True)
        else:
            left=list(itertools.product(range(len(bank)),repeat=n//2));right=list(itertools.product(range(len(bank)),repeat=n-n//2))
            try:
                pairs,receipt=exact_residual_pairs([''.join(tapes[i] for i in s) for s in left],[''.join(tapes[i] for i in s) for s in right],max_work=WORK-work,deadline=deadline)
            except JoinBudgetExceeded as exc:
                status='truncated';receipts.append(dict(clauses=n,denominator=len(bank)**n,complete=False,reason=str(exc),outputs_not_claimed=True));break
        work+=receipt['work'];receipts.append(dict(clauses=n,denominator=len(bank)**n,**receipt))
        for a,b in pairs:
            seq=left[a] if n==1 else left[a]+right[b]
            item=dict(text=' '.join(bank[i]['text'] for i in seq),sentences=[bank[i] for i in seq])
            raw.append(output_row(method,item,dict(sentence_ids=seq)))
            if len(raw)>PATHS:
                status='truncated_output_bound';break
        if status!='complete':break
    return dict(method=method,status=status,elapsed_seconds=time.monotonic()-start,sentence_bank_occurrences=len(bank),receipts=receipts,raw_outputs=raw)

def grammar_arm(specs):
    method='bidirectional_expanded_grammar';start=time.monotonic();raw=[];receipts=[];work=0;status='complete'
    for n in (1,2,3,4):
        graph=GrammarDAG(specs,n)
        (OUT/('grammar-'+str(n)+'.json')).write_text(json.dumps(dict(start=graph.start,accept=graph.accept,nodes=graph.nodes,arcs=[asdict(a) for a in graph.arcs],epsilon={str(k):sorted(v) for k,v in graph.eps.items() if v}),separators=(',',':')))
        with gzip.open(OUT/('product-states-'+str(n)+'.jsonl.gz'),'wt') as trace:
            def emit(row):trace.write(json.dumps(row,separators=(',',':'))+'\n')
            try:paths,receipt=exact_grammar_palindromes(graph,max_work=WORK-work,max_paths=PATHS-len(raw),seconds=max(.001,SECONDS-(time.monotonic()-start)),trace=emit)
            except SearchBudgetExceeded as exc:
                status='truncated';receipts.append(dict(clauses=n,outputs_not_claimed=True,**exc.receipt));break
        work+=receipt['work'];receipts.append(dict(clauses=n,**receipt))
        for path in paths:
            raw.append(output_row(method,graph.materialize(path),dict(character_arc_ids=path,graph_file='grammar-'+str(n)+'.json')))
    return dict(method=method,status=status,elapsed_seconds=time.monotonic()-start,receipts=receipts,raw_outputs=raw)

def dedup_and_sample(arm):
    grouped=defaultdict(list)
    for row in arm['raw_outputs']:grouped[row['tape']].append(row)
    outputs=[]
    for tape,rows in sorted(grouped.items()):
        row=rows[0].copy();row.pop('derivation');row['derivation_count']=len(rows)
        row['renderings']=sorted({r['text'] for r in rows});outputs.append(row)
    strata=defaultdict(list)
    for row in outputs:
        band='under60' if row['letters']<60 else ('60-119' if row['letters']<120 else '120plus')
        strata[(row['clause_count'],band,row['known_scaffold'])].append(row)
    sample=[];manifest=[]
    for key,rows in sorted(strata.items()):
        count=math.ceil(len(rows)/10);chosen=sorted(rows,key=lambda r:hashlib.sha256((SEED+'|'+r['id']).encode()).hexdigest())[:count]
        sample.extend(chosen);manifest.append(dict(clauses=key[0],length_band=key[1],contains_known_scaffold=key[2],denominator=len(rows),sample_count=count,ids=[r['id'] for r in chosen]))
    with gzip.open(OUT/(arm['method']+'-outputs.jsonl.gz'),'wt') as f:
        for row in arm.pop('raw_outputs'):f.write(json.dumps(row,separators=(',',':'))+'\n')
    arm.update(outputs=outputs,sample=sample,sampling_manifest=manifest,exact_derivations=sum(r['derivation_count'] for r in outputs),unique_exact_outputs=len(outputs),unique_long_outputs=sum(r['letters']>=60 for r in outputs),unique_without_classic_scaffold=sum(not r['known_scaffold'] for r in outputs))

def run():
    OUT.mkdir(parents=True,exist_ok=True);specs=frames();sentences=enumerate_sentences(specs)
    oldrows=json.loads(Path('research/block-seams/nonsees-fragment-004/bank.json').read_text())
    core=enumerate_sentences(tuple(f for f in specs if f.name in ('open','pets','say','let_in','copula','comparative','step')))
    assert {r['text'] for r in core}=={r['text'] for r in oldrows}
    vocab=vocabulary_receipt(specs)
    plan=dict(seed=SEED,methods=['enumerated_original_bank','enumerated_expanded_grammar','bidirectional_expanded_grammar'],max_work_per_arm=WORK,max_seconds_per_arm=SECONDS,max_accepted_derivations_per_arm=PATHS,clause_counts=[1,2,3,4],cpu_only=True,model_calls=0,fragment_bank_sentences=len(core),expanded_sentence_language_count=len(sentences),frames=sorted({f.name for f in specs}),sampling='ceil10% per method × clause count × letter band × classic-scaffold feature; sample deduplicated exact texts; report derivation distribution separately',comparison='original vs expanded enumerated arm measures vocabulary/grammar coverage; expanded enumerated vs bidirectional arm isolates representation on identical language',search_pruning='character inequality and exact graph reachability only; no semantic/repetition/quality pruning',human_calibration={'five_prior_outputs':'BAD; distinct earlier feedback','pending_v3_fragment_salvage-0332715fe9d2f50e':'18:16UTC readability-positive, repetition concern; no numeric rating or unqualified novel-paragraph approval','selection':'broad finite diverse pool then select; repetition is ranking feature'},stopping='complete finite matched-budget comparison or recorded resource truncation')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(OUT/'vocabulary.json').write_text(json.dumps(vocab,indent=2)+'\n');(OUT/'frames.json').write_text(json.dumps([asdict(f) for f in specs],indent=2)+'\n');(OUT/'enumerated-expanded-sentences.json').write_text(json.dumps(sentences,indent=2)+'\n')
    arms=[enumerated_arm('enumerated_original_bank',core),enumerated_arm('enumerated_expanded_grammar',sentences),grammar_arm(specs)]
    for arm in arms:
        dedup_and_sample(arm)
        print(json.dumps({k:v for k,v in arm.items() if k not in ('outputs','sample','sampling_manifest','receipts')}),flush=True)
    sets=[{r['tape'] for r in a['outputs']} for a in arms]
    matched=arms[1]['status']=='complete' and arms[2]['status']=='complete'
    if matched:assert sets[1]==sets[2]
    record=dict(plan=plan,arms=arms,all_outputs_exact=True,identical_expanded_language_output_sets=sets[1]==sets[2] if matched else None,all_arms_complete=all(a['status']=='complete' for a in arms),new_unique_output_tapes=sorted(sets[2]-sets[0]),human_review_queue=[])
    record['sample_count']=sum(len(a['sample']) for a in arms);record['unique_method_output_count']=sum(len(a['outputs']) for a in arms)
    record['overall_sample_percent']=100*record['sample_count']/record['unique_method_output_count'] if record['unique_method_output_count'] else 0
    (OUT/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    print('new texts',len(record['new_unique_output_tapes']),'sample',record['sample_count'],'percent',record['overall_sample_percent'])
    for row in arms[-1]['outputs']:
        if not row['known_scaffold'] and not row['repeated_sentences'] and row['clause_count']<=2:print(row['text'])
    return record

if __name__=='__main__':run()
