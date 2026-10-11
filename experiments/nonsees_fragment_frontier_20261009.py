"""Bounded complete finite splice diagnostic; no model calls or quality gate."""
import gzip, hashlib, itertools, json, math, time
from collections import Counter, defaultdict
from pathlib import Path
from llm_palindrome.admission import normalize_letters as norm
from llm_palindrome.fragment_frontier import FragmentFrontier
from llm_palindrome.paragraph_residual_join import exact_residual_pairs

OUT = Path('research/block-seams/nonsees-fragment-004')
SEED = 'nonsees-fragment-004'
MAX_SPLICES = 2000000
MAX_JOIN_WORK = 20000000
SECONDS = 30

def bank():
    rows=[]
    def add(frame,tokens,roles,source):
        text=' '.join(tokens)
        rows.append(dict(frame=frame,tokens=tokens,roles=roles,text=text[0].upper()+text[1:]+'.',provenance=source))
    for s,o in itertools.product(['I','we','Anna','Eve','Otto'],['mail','a door','a gate']):
        add('open',[s,'open' if s in ('I','we') else 'opens',*o.split()],['agent','verb',*(['theme']*len(o.split()))],'authored frame; open token v3#383')
    for s,o in itertools.product(['Anna','Eve','Otto','Noel'],['a cat','a dog','pets']):
        add('pets',[s,'pets',*o.split()],['agent','verb',*(['animate_theme']*len(o.split()))],'authored frame; pets token v3#453')
    for s,o in itertools.product(['I','we','Anna','Eve'],['yes','no','wow']):
        # Quoted content is syntactically distinct from an unmarked bare object.
        add('say',[s,'say' if s in ('I','we') else 'says',o],['speaker','verb','quoted_content'],'authored frame; say token v3#286')
        prefix=' '.join(rows[-1]['tokens'][:-1]);rows[-1]['text']=prefix[0].upper()+prefix[1:]+', "'+o+'."'
    for s in ['Anna','Eve','Otto','Noel']:
        add('let_in',['let',s,'in'],['imperative','animate_theme','particle'],'authored frame; let token v3#19')
    for s,a in itertools.product(['I','Ma','Anna','Eve','Otto'],['red','tired','sad','able','selfless','evil','live']):
        add('copula',[s,'am' if s=='I' else 'is',a],['subject','copula','predicate_adjective'],'authored frame; is token v3#526')
        add('comparative',[s,'am' if s=='I' else 'is','as',a,'as','I','am'],['subject','copula','degree','predicate_adjective','degree','comparator','copula'],'authored substitutions in catalogue v3#526; copied scaffold flagged')
    for o in ['no pets','the mat','a rug']:
        add('step',['step','on',*o.split()],['imperative','preposition',*(['theme']*len(o.split()))],'authored imperative; step token v3#265; no pets known control')
    return rows

def cuts(entry,method):
    tape=norm(entry['text']);at=0;out=[]
    for word,role in zip(entry['tokens'],entry['roles']):
        w=norm(word)
        offsets=range(1,len(w)) if method=='partial_word' else [0]
        for k in offsets:
            cut=at+k
            if 0<cut<len(tape):out.append((cut,role))
        at+=len(w)
    return out

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    entries=bank(); tapes=[norm(e['text']) for e in entries];lookup={t:i for i,t in enumerate(tapes)}
    assert len(lookup)==len(entries)
    plan=dict(seed=SEED,cpu_only=True,paid_calls=0,frames=sorted({e['frame'] for e in entries}),licensed_sentences=len(entries),max_splice_attempts_per_method=MAX_SPLICES,max_join_work_per_method=MAX_JOIN_WORK,wall_seconds_per_phase=SECONDS,methods=['whole_sentence_baseline','word_boundary','partial_word'],paragraph_clause_counts=[1,2,3],sampling='ceil(10% of exact outputs) independently per method/clause-count/length-band; zero yields zero samples; report aggregate rounding',quality='individual self-review scores 0-4; human labels null; repetition and copied controls are features, not vetoes',stopping='complete finite diagnostic or explicit truncation; no publication; no claim of novel readable paragraphs')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'bank.json').write_text(json.dumps(entries,indent=2)+'\n')
    f=FragmentFrontier(entries)
    for e in entries:
        assert f.complete(e['text'])
        for cut,role in cuts(e,'partial_word'):
            assert any(p.argument_role==role and not p.complete for p in f.positions(norm(e['text'])[:cut]))
    results=[]
    for method in plan['methods']:
        start=time.monotonic();deadline=start+SECONDS;selected=set();counts=Counter();representatives={};status='complete'
        if method=='whole_sentence_baseline':selected=set(range(len(entries)))
        else:
            groups=defaultdict(list)
            for i,e in enumerate(entries):
                for cut,role in cuts(e,method):groups[role].append((i,cut))
            denominator=sum(len(v)**2 for v in groups.values())
            with gzip.open(OUT/(method+'-attempts.jsonl.gz'),'wt') as raw:
                stop=False
                for role,positions in sorted(groups.items()):
                    for (i,a),(j,b) in itertools.product(positions,repeat=2):
                        if counts['attempts']>=MAX_SPLICES or time.monotonic()>deadline:status='truncated';stop=True;break
                        proposed=tapes[i][:a]+tapes[j][b:];licensed=lookup.get(proposed)
                        reason='licensed' if licensed is not None else 'no_complete_grammar_path'
                        counts['attempts']+=1;counts[reason]+=1
                        if licensed is not None:selected.add(licensed)
                        row=dict(left=i,left_offset=a,right=j,right_offset=b,role=role,result=licensed,reason=reason)
                        raw.write(json.dumps(row,separators=(',',':'))+'\n')
                        representatives.setdefault(reason,{**row,'proposed_tape':proposed})
                    if stop:break
            counts['declared_denominator']=denominator
        assert status=='complete', (method,dict(counts))
        splice_elapsed=time.monotonic()-start
        ids=sorted(selected);licensed_tapes=[tapes[i] for i in ids];outputs=[];receipts=[];work=0;deadline=time.monotonic()+SECONDS
        for n in (1,2,3):
            if n==1:pairs=[(i,None) for i,t in enumerate(licensed_tapes) if t==t[::-1]];left=[(i,) for i in ids];right=[];receipt={'work':len(ids),'complete':True}
            else:
                left=list(itertools.product(ids,repeat=n//2));right=list(itertools.product(ids,repeat=n-n//2))
                pairs,receipt=exact_residual_pairs([''.join(tapes[i] for i in seq) for seq in left],[''.join(tapes[i] for i in seq) for seq in right],max_work=MAX_JOIN_WORK-work,deadline=deadline)
            work+=receipt['work'];receipts.append({'clauses':n,'denominator':len(ids)**n,**receipt})
            for a,b in pairs:
                seq=left[a] if n==1 else left[a]+right[b];text=' '.join(entries[i]['text'] for i in seq);t=norm(text)
                assert t==t[::-1] and t
                outputs.append(dict(id=method+'-'+hashlib.sha256(text.encode()).hexdigest()[:16],text=text,letters=len(t),clauses=n,sentence_ids=seq,frames=[entries[i]['frame'] for i in seq],exact=True,repeated_clauses=n-len(set(seq)),known_scaffold=any(entries[i]['frame']=='comparative' or entries[i]['tokens']==['step','on','no','pets'] for i in seq),provenance=[entries[i]['provenance'] for i in seq],human_label=None))
        strata=defaultdict(list)
        for row in outputs:strata[(row['clauses'],'under60' if row['letters']<60 else '60plus')].append(row)
        sample=[];manifest=[]
        for key,rows in sorted(strata.items()):
            count=math.ceil(len(rows)*.1);chosen=sorted(rows,key=lambda r:hashlib.sha256((SEED+r['id']).encode()).hexdigest())[:count]
            sample.extend(chosen);manifest.append(dict(clauses=key[0],length_band=key[1],denominator=len(rows),sample_count=count,ids=[r['id'] for r in chosen]))
        results.append(dict(method=method,complete=True,splice_counts=dict(counts),splice_elapsed_seconds=splice_elapsed,representative_attempts=representatives,sentence_ids=ids,join_receipts=receipts,outputs=outputs,sample=sample,sampling_manifest=manifest))
    baseline={r['text'] for r in results[0]['outputs']}
    for r in results:assert {o['text'] for o in r['outputs']}==baseline
    record=dict(plan=plan,results=results,sample_count=sum(len(r['sample']) for r in results),exact_output_occurrences=sum(len(r['outputs']) for r in results),all_output_exact=True,all_methods_same_output_text_set=True,review_queue=[])
    record['overall_sample_percent']=100*record['sample_count']/record['exact_output_occurrences'] if record['exact_output_occurrences'] else 0
    (OUT/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('results','plan')}))
    for r in results:print(r['method'],r['splice_counts'],'outputs',len(r['outputs']),[o['text'] for o in r['outputs'][:5]])
    return record

if __name__=='__main__':run()
