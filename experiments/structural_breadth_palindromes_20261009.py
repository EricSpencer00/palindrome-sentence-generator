"""Version009: six disjoint first-frame partitions, frozen prior review.

All prior run sources/outputs are read-only. Each partition shares its bound
across1..4 sentences. Search remains the tested lexical role/DAG product.
"""
import gzip,hashlib,json,math,time
from collections import Counter,defaultdict,deque
from dataclasses import asdict
from pathlib import Path
from llm_palindrome.admission import normalize_letters as norm
from llm_palindrome.bidirectional_lexical import Frame,GrammarDAG,exact_grammar_palindromes,SearchBudgetExceeded
from llm_palindrome.typed_constituents import NAMES
from experiments.bidirectional_lexical_completion_20261009 import frames as old_frames,slot,enumerate_sentences,output_row

ROOT=Path('research/block-seams');OUT=ROOT/'structural-breadth-009'
WORK=20000000;SECONDS=30;PATHS=20000;SEED='structural-breadth-009'

class BreadthFrame(Frame):
    def render(self,words):
        if self.style=='tail_vocative':
            text=' '.join(words[:-1]);return text[0].upper()+text[1:]+', '+words[-1]+'.'
        if self.style=='relative_delivery':
            # Human proper-name recipient takes a nonrestrictive relative.
            text=' '.join(words[:5])+', '+' '.join(words[5:])
            # addressee,deliver,desserts,to,Anna,who,reviled,Leon
            return text[0].upper()+text[1:].replace(' deliver',', deliver',1)+'.'
        return super().render(words)

def breadth_frames():
    fs=list(old_frames(include_food=True))
    def add(name,slots,style='statement',provenance='new authored licensed role frame',known=False):fs.append(BreadthFrame(name,tuple(slots),style,provenance,known))
    names=slot('addressee','Noel','Leon','Anna','Eve');people=slot('human_theme','Noel','Leon','Anna','Eve','Iris');subjects=slot('human_agent','Noel','Leon','Liam','Anna','Eve')
    honorific=[slot('addressee','Sir'),slot('agent','I')]
    add('addressed_delivery_report',honorific+[slot('verb','deliver'),slot('deliverable_theme','mail','pots','pets','desserts')],'vocative','first-person report to an honorific addressee; deliverable object; no copied whole scaffold')
    add('addressed_person_delivery_report',honorific+[slot('verb','deliver'),people],'vocative','first-person person-transport report; destination implicit')
    add('addressed_person_delivery_report',honorific+[slot('verb','deliver'),slot('human_adjective','reviled'),people],'vocative','first-person report; reviled adjective licenses a human theme')
    add('addressed_food_delivery_report',honorific+[slot('verb','deliver'),slot('recipient','Noel','Leon','Anna','Eve'),slot('food_theme','desserts')],'vocative','first-person recipient-food double-object frame; desserts/stressed lexical reversal retained as existing material')
    add('addressed_reference_report',honorific+[slot('verb','refer'),people],'vocative','first-person referral report; destination context-dependent')
    add('iris_criticism',[subjects,slot('past_verb','reviled'),slot('human_theme','Iris')],provenance='same human-agent criticism grammar with an existing human-name sense of Iris')
    add('iris_stressed_criticism',[slot('human_adjective','stressed'),slot('human_agent','Noel','Leon','Anna','Eve'),slot('past_verb','reviled'),slot('human_theme','Iris')],provenance='affected-human subject and human theme; no original body retained')
    add('iris_celebrity_criticism',[slot('human_plural_agent','stars'),slot('past_verb','reviled'),slot('human_theme','Iris')],provenance='stars has explicitly intended celebrity meaning, not celestial-body agent')
    add('material_delivery_request',[names,slot('imperative','deliver'),slot('material_adjective','red'),slot('material_theme','iron')],'vocative','red iron constituent previously present in accepted-readable sees material; new non-seeing delivery role')
    add('addressed_material_delivery_report',honorific+[slot('verb','deliver'),slot('material_adjective','red'),slot('material_theme','iron')],'vocative','first-person material-delivery frame')
    add('no_rider_criticism',[slot('negative_quantifier','no'),slot('human_agent','rider'),slot('past_verb','reviled'),people],provenance='No rider constituent, human negative subject; red iron/no rider lexical seam retained as existing material')
    add('animal_delivery_statement',[slot('human_agent','Noel','Leon','Anna','Eve'),slot('verb','delivers'),slot('animal_theme','rats')],provenance='deliver+s agrees with human singular; rats is rat+s; rats/stars licenses animal-theme versus celebrity-agent overhang')
    add('drawing_request',[slot('imperative','draw'),slot('determiner','a'),slot('location_theme','ward')],provenance='draw depicts a hospital ward; draw/ward is an existing reversed lexical pair; generated slot path, not original-bank text')
    add('reward_request',[slot('imperative','reward'),slot('determiner','a'),slot('intended_human_theme','drawer')],provenance='intended person-who-draws meaning; drawer is ambiguous with furniture, so semantic interpretation remains a quality concern')
    add('existential_request',[slot('imperative','live')],provenance='complete intransitive imperative; existential command requires discourse context')
    add('evil_criticism',[slot('human_adjective','evil'),slot('human_agent','Noel','Leon','Anna','Eve'),slot('past_verb','reviled'),people],provenance='moral adjective modifies a human subject; live/evil existing lexical reversal')
    add('animal_pet_statement',[slot('human_agent','Noel','Leon','Anna','Eve'),slot('verb','pets'),slot('determiner','a'),slot('animal_theme','rat')],provenance='animal object, human agent; no inanimate seeing subject')
    add('tail_stepping_request',[slot('imperative','step'),slot('addressee','Noel','Leon')],'tail_vocative','intransitive physical stepping imperative with final vocative; direction context-dependent')
    add('coordinated_delivery_request',[names,slot('imperative','deliver'),slot('deliverable_theme','mail','desserts'),slot('coordinator','and'),slot('deliverable_theme','mail','desserts')],'vocative','two overt coordinated deliverable objects; tests conjunction coverage rather than word-debt filler')
    add('causal_delivery_report',honorific+[slot('verb','deliver'),slot('deliverable_theme','mail','desserts'),slot('causal_connector','because'),slot('human_agent','Anna','Eve'),slot('past_verb','reviled'),slot('human_theme','Leon','Iris')],'vocative','explicit causal relation in a grammatical report; truth/plausibility not inferred')
    add('relative_delivery_request',[names,slot('imperative','deliver'),slot('food_theme','desserts'),slot('preposition','to'),slot('recipient','Anna','Eve'),slot('relative_pronoun','who'),slot('past_verb','reviled'),slot('human_theme','Leon','Iris')],'relative_delivery','food delivery to a named recipient with nonrestrictive human relative clause')
    add('purpose_delivery_report',honorific+[slot('verb','deliver'),slot('food_theme','desserts'),slot('preposition','to'),people],'vocative','explicit recipient preposition; named report theme')
    add('linked_past_scene',[slot('human_agent','Anna','Eve'),slot('past_verb','delivered'),slot('food_theme','desserts'),slot('causal_connector','because'),slot('human_agent','Anna','Eve'),slot('past_verb','reviled'),slot('human_theme','Leon','Iris')],provenance='two finite past clauses with overt because link; deliver+ed morphology')
    return tuple(fs)

GROUPS={
 '01-person-transport':{'person_delivery_request','addressed_person_delivery_report'},
 '02-criticism':{'past_revile','iris_criticism','evil_criticism'},
 '03-referral-movement':{'reference_request','stopping_request','addressed_reference_report','tail_stepping_request','existential_request'},
 '04-food-affect':{'food_delivery_request','stressed_criticism','addressed_food_delivery_report','iris_stressed_criticism'},
 '05-delivery-material':{'delivery_request','recipient_delivery_request','delivery_statement','celebrity_revile','addressed_delivery_report','material_delivery_request','addressed_material_delivery_report','no_rider_criticism','animal_delivery_statement','iris_celebrity_criticism'},
 '06-other-structure':{'open','pets','say','let_in','copula','comparative','step','drawing_request','reward_request','animal_pet_statement','coordinated_delivery_request','causal_delivery_report','relative_delivery_request','purpose_delivery_report','linked_past_scene'},
}

def restricted_graph(specs,n,first_families):
    g=GrammarDAG(specs,n)
    keep=set()
    for root in g.eps[g.start]:
        fi=g.arcs[g.out[root][0]].frame
        if g.frames[fi].name in first_families:keep.add(root)
        else:g.reverse_eps[root].discard(g.start)
    g.eps[g.start]=keep
    # Recompute exact graph reachability after removing first-frame starts.
    successors={x:set(g.eps[x])|{g.arcs[a].target for a in g.out[x]} for x in range(g.nodes)}
    indegree=[0]*g.nodes
    for dests in successors.values():
        for d in dests:indegree[d]+=1
    queue=deque(x for x,d in enumerate(indegree) if d==0);order=[]
    while queue:
        x=queue.popleft();order.append(x)
        for d in successors[x]:
            indegree[d]-=1
            if indegree[d]==0:queue.append(d)
    assert len(order)==g.nodes
    for x in reversed(order):
        mask=1<<x
        for d in successors[x]:mask|=g.reachable[d]
        g.reachable[x]=mask
    return g

def frozen_receipt():
    paths=[ROOT/'food-argument-lexical-006/results.json',ROOT/'bidirectional-lexical-005/results.json']
    paths+=[ROOT/'blind-readability-export-008'/name for name in ('blind-sampled-occurrences-4412.jsonl','blind-unique-texts-3949.jsonl','occurrence-rating-map-4412.jsonl')]
    paths+=sorted((ROOT/'blind-readability-export-008/chunks').glob('*.jsonl'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def vocabulary(specs):
    lex=set(Path('data/lexicon.txt').read_text().splitlines());old=json.loads((ROOT/'food-argument-lexical-006/vocabulary.json').read_text());prior={r['word']:r for r in old['words']};rows=[]
    morphology={'rats':('rat','plural_s'),'delivered':('deliver','past_ed')}
    for w in sorted({w for f in specs for s in f.slots for w in s.words}):
        if w in prior:source=prior[w]['source']
        elif norm(w) in morphology:
            lemma,rule=morphology[norm(w)];assert lemma in lex;source=dict(kind='explicit_morphology',lemma=lemma,rule=rule)
        elif norm(w) in lex:source=dict(kind='shipped_lexicon',headword=norm(w))
        elif w in NAMES:source=dict(kind='existing_typed_name_vocabulary',source='llm_palindrome/typed_constituents.py:NAMES')
        else:raise AssertionError('unlicensed vocabulary '+w)
        rows.append(dict(word=w,roles=sorted({s.role for f in specs for s in f.slots if w in s.words}),source=source,new_relative_to006=w not in prior))
    return rows

def run():
    OUT.mkdir(parents=True,exist_ok=True);frozen=frozen_receipt();specs=breadth_frames();vocab=vocabulary(specs)
    fams={f.name for f in specs};assert set.union(*GROUPS.values())==fams;assert sum(len(x) for x in GROUPS.values())==len(fams)
    old=json.loads((ROOT/'food-argument-lexical-006/results.json').read_text());baseline={r['tape'] for r in old['arms'][-1]['outputs']}
    feedback=dict(message_id='Sentinel_083ad70b72f48191b8ab014f4eb45d23',quote='hm those are okay, theyre real sentences, lets get as much as we can',scope='qualitative sentence-readability acceptance of three displayed examples; no numeric score, novelty or full-cohesive-paragraph claim',actual_displayed_texts=['Noel, deliver Anna desserts. Noel, refer Eve. Eve, refer Leon. Stressed Anna reviled Leon.','Noel, deliver mail. Anna reviled Leon. Noel, deliver Anna. Liam reviled Leon.','Noel, deliver Anna desserts. Stressed Anna reviled Leon.'])
    # IDs are resolved from the frozen output itself, not assumed by order.
    oldlookup={r['text']:r['id'] for r in old['arms'][-1]['outputs']};feedback['candidate_ids']=[oldlookup[t] for t in feedback['actual_displayed_texts']]
    plan=dict(run=SEED,seed=SEED,cpu_only=True,model_calls=0,clause_counts=[1,2,3,4],partitions={k:sorted(v) for k,v in GROUPS.items()},max_work_per_partition=WORK,max_seconds_per_partition=SECONDS,max_accepted_derivations_per_partition=PATHS,aggregate_max_operations=WORK*len(GROUPS),aggregate_max_arm_seconds=SECONDS*len(GROUPS),aggregate_max_derivations=PATHS*len(GROUPS),expanded_sentence_language_count=len(enumerate_sentences(specs)),grammar_families=len(fams),sampling='ceil10% per first-family partition × clause count × under60/60-119/120plus × new-to-frozen006 feature; retain raw duplicate derivations; no quality/repetition pruning',baseline='frozen006 14,260-text set; no rerun of old search or old samples',stopping='stop after six partitions and complete saved yield/coverage checkpoint; truncation explicitly reported before any further batch')
    for name,obj in [('plan',plan),('human-feedback',feedback),('frozen-inputs',frozen),('vocabulary',vocab),('frames',[asdict(f) for f in specs])]: (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
    all_rows={};partitions=[]
    for pname,families in GROUPS.items():
        start=time.monotonic();work=0;accepted=0;receipts=[];local=defaultdict(list);status='complete'
        with gzip.open(OUT/(pname+'-raw-outputs.jsonl.gz'),'wt') as raw:
            for n in plan['clause_counts']:
                g=restricted_graph(specs,n,families)
                (OUT/(pname+'-graph-'+str(n)+'.json.gz')).write_bytes(gzip.compress(json.dumps(dict(start=g.start,accept=g.accept,nodes=g.nodes,arcs=[asdict(a) for a in g.arcs],epsilon={str(k):sorted(v) for k,v in g.eps.items() if v}),separators=(',',':')).encode(),mtime=0))
                with gzip.open(OUT/(pname+'-states-'+str(n)+'.jsonl.gz'),'wt') as trace:
                    def emit(row):trace.write(json.dumps(row,separators=(',',':'))+'\n')
                    try:paths,receipt=exact_grammar_palindromes(g,max_work=WORK-work,max_paths=PATHS-accepted,seconds=max(.001,SECONDS-(time.monotonic()-start)),trace=emit)
                    except SearchBudgetExceeded as exc:
                        status='truncated';receipts.append(dict(clauses=n,outputs_not_claimed=True,**exc.receipt));g.closure.cache_clear();g.transitions.cache_clear();break
                work+=receipt['work'];receipts.append(dict(clauses=n,**receipt))
                for path in paths:
                    item=g.materialize(path);assert item['sentences'][0]['frame'] in families
                    row=output_row(pname,item,dict(character_arc_ids=path,graph_file=pname+'-graph-'+str(n)+'.json.gz'))
                    row.update(run=SEED,partition=pname,new_to_frozen006=row['tape'] not in baseline)
                    raw.write(json.dumps(row,separators=(',',':'))+'\n');local[row['tape']].append(row);accepted+=1
                g.closure.cache_clear();g.transitions.cache_clear()
        local_rows=[]
        for t,rows in sorted(local.items()):
            row=rows[0].copy();row.pop('derivation');row['derivation_count']=len(rows);row['renderings']=sorted({r['text'] for r in rows});local_rows.append(row)
            if t not in all_rows:all_rows[t]=row.copy();all_rows[t]['partitions']=[pname]
            else:all_rows[t]['derivation_count']+=len(rows);all_rows[t]['partitions'].append(pname)
        strata=defaultdict(list)
        for row in local_rows:
            band='under60' if row['letters']<60 else ('60-119' if row['letters']<120 else '120plus')
            strata[(row['clause_count'],band,row['new_to_frozen006'])].append(row)
        sample=[];manifest=[]
        for st,rows in sorted(strata.items()):
            k=math.ceil(len(rows)/10);chosen=sorted(rows,key=lambda r:hashlib.sha256((SEED+'|'+r['id']).encode()).hexdigest())[:k];sample.extend(chosen);manifest.append(dict(clauses=st[0],length_band=st[1],new_to_frozen006=st[2],denominator=len(rows),sample_count=k,ids=[r['id'] for r in chosen]))
        record=dict(partition=pname,first_families=sorted(families),status=status,elapsed_seconds=time.monotonic()-start,work=sum(r['work'] for r in receipts),accepted_derivations=accepted,unique_exact_outputs=len(local_rows),new_unique_outputs=sum(r['new_to_frozen006'] for r in local_rows),long_unique_outputs=sum(r['letters']>=60 for r in local_rows),receipts=receipts,sampling_manifest=manifest,sample=sample)
        partitions.append(record);(OUT/(pname+'-receipt.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ('sample','sampling_manifest','receipts')}),flush=True)
    with gzip.open(OUT/'all-deduplicated-outputs.jsonl.gz','wt') as f:
        for t,row in sorted(all_rows.items()):f.write(json.dumps(row,separators=(',',':'))+'\n')
    assert frozen_receipt()==frozen,'frozen review inputs changed'
    union=set(all_rows);closed={s['frame'] for row in all_rows.values() for s in row['sentences']};coverage={f:sum(f in row['frames'] for row in all_rows.values()) for f in sorted(fams)}
    summary=dict(plan=plan,all_partitions_complete=all(r['status']=='complete' for r in partitions),unique_exact_outputs=len(union),new_unique_to_frozen006=len(union-baseline),baseline_outputs_recovered=len(union&baseline),raw_derivations=sum(r['accepted_derivations'] for r in partitions),long_unique_outputs=sum(r['letters']>=60 for r in all_rows.values()),new_long_unique_outputs=sum(r['letters']>=60 and r['new_to_frozen006'] for r in all_rows.values()),identical_sentence_repeat_distribution=dict(Counter(r['repeated_sentences'] for r in all_rows.values())),family_output_coverage=coverage,zero_closure_families=sorted(fams-closed),sampled_partition_occurrences=sum(len(r['sample']) for r in partitions),partition_population_denominator=sum(r['unique_exact_outputs'] for r in partitions),work=sum(r['work'] for r in partitions),measured_partition_seconds=sum(r['elapsed_seconds'] for r in partitions),frozen_review_inputs_unchanged=True,human_quality_approved_new_run_outputs=0,partitions=[{k:v for k,v in r.items() if k!='sample'} for r in partitions])
    summary.pop('human_quality_approved_new_run_outputs')
    summary.update(new_to006_outputs_with_human_quality_approval=0,retained_previous_sentence_readability_accepted_texts=sum(norm(t) in union for t in feedback['actual_displayed_texts']))
    summary['sample_percent']=100*summary['sampled_partition_occurrences']/summary['partition_population_denominator'] if summary['partition_population_denominator'] else 0
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('plan','partitions','family_output_coverage')}),flush=True)
    return summary

if __name__=='__main__':run()
