"""Reverify009 without searches; rebuild frozen sample and blind exports."""
import argparse,gzip,hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
from experiments.structural_breadth_palindromes_20261009 import OUT,ROOT,SEED,GROUPS,breadth_frames,frozen_receipt

def norm(text):return ''.join(re.findall('[a-z]',text.lower()))
def key(text):return 'sha256:'+hashlib.sha256(norm(text).encode()).hexdigest()
def lines(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def write(path,rows):
    with path.open('w') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    specs=breadth_frames();summary=json.loads((OUT/'summary.json').read_text());frozen=json.loads((OUT/'frozen-inputs.json').read_text())
    assert frozen_receipt()==frozen
    pool=list(lines(OUT/'all-deduplicated-outputs.jsonl.gz'));lookup={r['tape']:r for r in pool};assert len(lookup)==len(pool)==26641
    derivations=Counter();occurrences=[];provenance=[];state_checks=[];raw_count=0
    for partition in summary['partitions']:
        pname=partition['partition'];receipt=json.loads((OUT/(pname+'-receipt.json')).read_text());assert receipt['status']=='complete';local=defaultdict(list)
        graphs={}
        for rec in receipt['receipts']:
            n=rec['clauses'];g=json.loads(gzip.decompress((OUT/(pname+'-graph-'+str(n)+'.json.gz')).read_bytes()));graphs[n]=g;arcs=g['arcs'];tot=Counter()
            for row in lines(OUT/(pname+'-states-'+str(n)+'.jsonl.gz')):
                tot['states']+=1
                if row['reason']=='no_grammar_path':tot['reachability_rejects']+=1
                else:
                    l=Counter(arcs[a]['char'] for a in row['left_arcs']);r=Counter(arcs[a]['char'] for a in row['right_arcs']);matched=sum(v*r[c] for c,v in l.items());mismatched=len(row['left_arcs'])*len(row['right_arcs'])-matched
                    assert matched==row['matched_arc_pairs'] and mismatched==row['char_mismatches']
                    tot['matched_arc_pairs']+=matched;tot['char_mismatch_pairs']+=mismatched
                    tot['partial_word_states']+=any(arcs[a]['offset']>0 for a in row['left_arcs']) or any(arcs[a]['offset']>0 for a in row['right_arcs'])
            for k in ('states','reachability_rejects','matched_arc_pairs','char_mismatch_pairs','partial_word_states'):assert tot[k]==rec[k]
            state_checks.append(dict(partition=pname,clauses=n,all_counters_replayed=True,**dict(tot)))
        local_count=0
        for row in lines(OUT/(pname+'-raw-outputs.jsonl.gz')):
            assert norm(row['text'])==row['tape']==row['tape'][::-1]
            assert row['letters']==len(row['tape']) and row['sentences'][0]['frame'] in GROUPS[pname]
            for sentence in row['sentences']:
                f=specs[sentence['frame_index']];assert sentence['frame']==f.name and len(sentence['words'])==len(f.slots)
                assert all(w in s.words for w,s in zip(sentence['words'],f.slots))
                assert f.render(sentence['words'])==sentence['text']
            assert ' '.join(s['text'] for s in row['sentences'])==row['text']
            arcs=graphs[row['clause_count']]['arcs'];path=row['derivation']['character_arc_ids']
            assert ''.join(arcs[a]['char'] for a in path)==row['tape']
            assert [arcs[a]['word'] for a in path if arcs[a]['offset']==0]==[w for s in row['sentences'] for w in s['words']]
            local[row['tape']].append(row);derivations[row['tape']]+=1;local_count+=1
        assert local_count==receipt['accepted_derivations'];raw_count+=local_count
        strata=defaultdict(list)
        for t,rows in sorted(local.items()):
            row=rows[0];band='under60' if row['letters']<60 else ('60-119' if row['letters']<120 else '120plus')
            strata[(row['clause_count'],band,row['new_to_frozen006'])].append(row)
        expected=[]
        for st,rows in sorted(strata.items()):
            k=math.ceil(len(rows)/10);chosen=sorted(rows,key=lambda r:hashlib.sha256((SEED+'|'+r['id']).encode()).hexdigest())[:k]
            opaque='stratum-'+hashlib.sha256(json.dumps([pname,*st],separators=(',',':')).encode()).hexdigest()[:16]
            provenance.append(dict(stratum_id=opaque,partition=pname,clauses=st[0],length_band=st[1],new_to_frozen006=st[2],denominator=len(rows),sample_count=k,ids=[r['id'] for r in chosen]))
            expected.extend(chosen)
            for row in chosen:occurrences.append(dict(occurrence_id=SEED+'#'+row['id'],method='bidirectional_grammar',partition=pname,seed=SEED,stratum_id=opaque,text=row['text'],normalized_text_key=key(row['text']),rating_id='text-'+key(row['text']).removeprefix('sha256:')))
        assert [r['id'] for r in expected]==[r['id'] for r in receipt['sample']]
        assert [(m['denominator'],m['sample_count'],m['ids']) for m in receipt['sampling_manifest']]==[(m['denominator'],m['sample_count'],m['ids']) for m in provenance[-len(strata):]]
        assert receipt['work']<=summary['plan']['max_work_per_partition'] and receipt['elapsed_seconds']<summary['plan']['max_seconds_per_partition'] and local_count<=summary['plan']['max_accepted_derivations_per_partition']
    assert derivations==Counter({r['tape']:r['derivation_count'] for r in pool});assert raw_count==26641
    assert len(occurrences)==2686 and len({r['normalized_text_key'] for r in occurrences})==2686
    unique=[{k:r[k] for k in ('rating_id','normalized_text_key','text')} for r in occurrences]
    unique.sort(key=lambda r:hashlib.sha256(('blind009|'+r['rating_id']).encode()).hexdigest())
    mapping=[{k:r[k] for k in ('occurrence_id','rating_id','normalized_text_key','partition','seed','stratum_id')} for r in occurrences]
    export=OUT/'blind-review';export.mkdir(exist_ok=True);(export/'chunks').mkdir(exist_ok=True)
    exports=[(export/'sample-occurrences-2686.jsonl',occurrences),(export/'unique-texts-2686.jsonl',unique),(export/'occurrence-map-2686.jsonl',mapping)]
    exports.extend((export/'chunks'/f'chunk-{i//64+1:03d}.jsonl',unique[i:i+64]) for i in range(0,len(unique),64))
    for path,rows in exports:
        if args.write:write(path,rows)
        else:assert list(lines(path))==rows
    old_unique={r['normalized_text_key']:r for r in lines(ROOT/'blind-readability-export-008/blind-unique-texts-3949.jsonl')}
    overlaps=[]
    for row in occurrences:
        if row['normalized_text_key'] in old_unique:
            old=old_unique[row['normalized_text_key']];assert old['text']==row['text']
            overlaps.append(dict(new_occurrence_id=row['occurrence_id'],new_rating_id=row['rating_id'],frozen008_rating_id=old['rating_id'],normalized_text_key=row['normalized_text_key'],judgment_reused=False))
    private=dict(strata=provenance,overlap_with_frozen008=overlaps,independent_ratings_completed=False)
    if args.write:(OUT/'review-provenance.json').write_text(json.dumps(private,indent=2)+'\n')
    else:assert json.loads((OUT/'review-provenance.json').read_text())==private
    receipt=dict(all_raw_outputs_exact_and_role_licensed=True,verified_raw_derivations=raw_count,verified_unique_outputs=len(pool),state_traces_replayed=len(state_checks),state_trace_checks=state_checks,sample_occurrences=len(occurrences),unique_sample_texts=len(unique),sample_percent=100*len(occurrences)/len(pool),all_sample_membership_reproduced=True,chunk_count=math.ceil(len(unique)/64),max_unique_per_chunk=64,last_chunk_size=len(unique)%64,overlap_with_frozen008=len(overlaps),old_ratings_inferred_or_reused=0,frozen_review_inputs_unchanged=True,independent_new_review_completed=False)
    if args.write:(OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='state_trace_checks'}))

if __name__=='__main__':main()
