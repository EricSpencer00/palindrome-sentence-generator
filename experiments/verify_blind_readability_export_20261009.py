"""Reproduce fixed sample membership, blind export, chunks and rating fanout.

Standard library only. No grammar search, scoring or model/API calls.
Run from the repository root, with --write once to create the blind export.
Without --write, recompute everything and compare every exported record.
Optional --ratings judgments.jsonl --mapped-output occurrences-rated.jsonl
requires one independent rating per unique text, then maps it to every
original sampled occurrence. Missing/duplicate ratings fail closed.
"""
import argparse,hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path('research/block-seams')
RUNS=('bidirectional-lexical-005','food-argument-lexical-006')
OUT=ROOT/'blind-readability-export-008'
CHUNK_SIZE=64

def dump(row):return json.dumps(row,ensure_ascii=True,separators=(',',':'))
def tape(text):
    if any(c.isalpha() and not c.isascii() for c in text):raise ValueError('non-ASCII letter')
    return ''.join(re.findall('[a-z]',text.casefold()))
def key(text):return 'sha256:'+hashlib.sha256(tape(text).encode()).hexdigest()
def readlines(path):
    with path.open() as f:return [json.loads(line) for line in f if line.strip()]
def write_lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for row in rows:f.write(dump(row)+'\n')

def fanout_ratings(occurrences,unique,incoming):
    index={row['rating_id']:row for row in incoming}
    assert len(index)==len(incoming),'duplicate rating IDs'
    assert set(index)=={row['rating_id'] for row in unique},'missing or unexpected ratings'
    for row in incoming:
        for dim in ('grammar','readability','coherence','repetition'):
            assert isinstance(row[dim],int) and not isinstance(row[dim],bool) and 0<=row[dim]<=4
        assert isinstance(row['explanation'],str) and row['explanation'].strip()
    return [dict(occurrence_id=row['occurrence_id'],rating_id=row['rating_id'],normalized_text_key=row['normalized_text_key'],independent_judgment=index[row['rating_id']]) for row in occurrences]

def build():
    occurrences=[];groups={};strata=[];budgets=[];method_outputs=0;union=set();source_hashes={}
    for run in RUNS:
        source=ROOT/run/'results.json';source_hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        result=json.loads(source.read_text());seed=result['plan']['seed'];poolsets={}
        for arm in result['arms']:
            assert arm['status']=='complete'
            method=arm['method'];all_rows=arm['outputs'];method_outputs+=len(all_rows)
            poolsets[method]={row['tape'] for row in all_rows}
            assert len(poolsets[method])==len(all_rows)
            if method=='bidirectional_expanded_grammar':union.update(poolsets[method])
            populations=defaultdict(list)
            for row in all_rows:
                assert tape(row['text'])==row['tape'] and row['tape']==row['tape'][::-1]
                band='under60' if row['letters']<60 else ('60-119' if row['letters']<120 else '120plus')
                populations[(row['clause_count'],band,row['known_scaffold'])].append(row)
            expected_sample=[];stratum_of={}
            for st,rows in sorted(populations.items()):
                count=math.ceil(len(rows)/10)
                chosen=sorted(rows,key=lambda row:hashlib.sha256((seed+'|'+row['id']).encode()).hexdigest())[:count]
                stratum_id='stratum-'+hashlib.sha256(dump([run,method,*st]).encode()).hexdigest()[:16]
                expected_sample.extend(chosen)
                for row in chosen:stratum_of[row['id']]=stratum_id
                strata.append(dict(stratum_id=stratum_id,run=run,method=method,seed=seed,clause_count=st[0],length_band=st[1],contains_known_scaffold=st[2],denominator=len(rows),sample_count=count,occurrence_ids=[run+'#'+row['id'] for row in chosen]))
            assert [row['id'] for row in expected_sample]==[row['id'] for row in arm['sample']]
            assert len(expected_sample)==len(arm['sample'])
            # Reproduce the stored manifest's membership and denominators.
            for stored,rebuilt in zip(arm['sampling_manifest'],strata[-len(populations):]):
                assert stored['denominator']==rebuilt['denominator']
                assert stored['sample_count']==rebuilt['sample_count']
                assert [run+'#'+sid for sid in stored['ids']]==rebuilt['occurrence_ids']
            assert len(arm['sampling_manifest'])==len(populations)
            for row in expected_sample:
                normalized_key=key(row['text']);rating_id='text-'+normalized_key.removeprefix('sha256:')
                occurrence=dict(occurrence_id=run+'#'+row['id'],method=method,seed=seed,stratum_id=stratum_of[row['id']],text=row['text'],normalized_text_key=normalized_key,rating_id=rating_id)
                occurrences.append(occurrence)
                if rating_id not in groups:groups[rating_id]=dict(rating_id=rating_id,normalized_text_key=normalized_key,text=row['text'],occurrence_ids=[])
                # Do not silently merge differently punctuated/segmented text
                # merely because the normalized tapes are equal.
                assert groups[rating_id]['text']==row['text'],'different rendering needs separate rating labor'
                groups[rating_id]['occurrence_ids'].append(occurrence['occurrence_id'])
            budgets.append(dict(run=run,method=method,work=sum(r['work'] for r in arm['receipts']),operation_limit=result['plan']['max_work_per_arm'],elapsed_seconds=arm['elapsed_seconds'],seconds_limit=result['plan']['max_seconds_per_arm'],accepted_derivations=arm['exact_derivations'],derivation_limit=result['plan']['max_accepted_derivations_per_arm']))
        assert poolsets['enumerated_expanded_grammar']==poolsets['bidirectional_expanded_grammar']
    unique=sorted(groups.values(),key=lambda row:hashlib.sha256(('blind-labor-008|'+row['rating_id']).encode()).hexdigest())
    mapping=[{field:row[field] for field in ('occurrence_id','rating_id','normalized_text_key','method','seed','stratum_id')} for row in occurrences]
    assert len({row['occurrence_id'] for row in occurrences})==len(occurrences)==4412
    assert len(unique)==3949 and method_outputs==43816 and len(union)==14260
    assert Counter(oid for row in unique for oid in row['occurrence_ids'])==Counter(row['occurrence_id'] for row in occurrences)
    assert all(b['work']<=b['operation_limit'] and b['elapsed_seconds']<b['seconds_limit'] and b['accepted_derivations']<=b['derivation_limit'] for b in budgets)
    return occurrences,unique,mapping,strata,budgets,source_hashes

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');parser.add_argument('--ratings',type=Path);parser.add_argument('--mapped-output',type=Path)
    args=parser.parse_args();occurrences,unique,mapping,strata,budgets,source_hashes=build()
    blind_labor=[{field:row[field] for field in ('rating_id','normalized_text_key','text')} for row in unique]
    chunks=[blind_labor[i:i+CHUNK_SIZE] for i in range(0,len(blind_labor),CHUNK_SIZE)]
    files=[(OUT/'blind-sampled-occurrences-4412.jsonl',occurrences),(OUT/'blind-unique-texts-3949.jsonl',unique),(OUT/'occurrence-rating-map-4412.jsonl',mapping)]
    files.extend((OUT/'chunks'/f'blind-unique-{i+1:03d}.jsonl',rows) for i,rows in enumerate(chunks))
    for path,rows in files:
        if args.write:write_lines(path,rows)
        else:assert readlines(path)==rows,str(path)
    # Stratum meaning is withheld from blind text; provenance is separate.
    provenance=ROOT/'blind-readability-provenance-008.json'
    private=dict(strata=strata,source_result_sha256=source_hashes)
    if args.write:provenance.write_text(json.dumps(private,indent=2)+'\n')
    else:assert json.loads(provenance.read_text())==private
    allowed_occurrence={'occurrence_id','method','seed','stratum_id','text','normalized_text_key','rating_id'}
    assert all(set(row)==allowed_occurrence for row in occurrences)
    assert all(set(row)=={'rating_id','normalized_text_key','text','occurrence_ids'} for row in unique)
    flattened=[row for rows in chunks for row in rows];assert flattened==blind_labor
    receipt=dict(sample_occurrences=len(occurrences),unique_rating_texts=len(unique),duplicate_labor_saved=len(occurrences)-len(unique),chunk_count=len(chunks),max_unique_texts_per_chunk=CHUNK_SIZE,last_chunk_size=len(chunks[-1]),method_output_occurrences=43816,distinct_output_texts=14260,sample_percent=100*len(occurrences)/43816,all_sample_membership_reproduced=True,all_occurrences_mapped_once=True,blind_fields_verified=True,all_expanded_output_sets_matched=True,unique_text_occurrence_count_distribution=dict(sorted(Counter(len(row['occurrence_ids']) for row in unique).items())),budgets=budgets,aggregate_actual_operations=sum(b['work'] for b in budgets),aggregate_operation_limits=sum(b['operation_limit'] for b in budgets),aggregate_measured_arm_seconds=sum(b['elapsed_seconds'] for b in budgets),aggregate_arm_seconds_limits=sum(b['seconds_limit'] for b in budgets),aggregate_accepted_derivations=sum(b['accepted_derivations'] for b in budgets),aggregate_derivation_limits=sum(b['derivation_limit'] for b in budgets),source_result_sha256=source_hashes,independent_ratings_completed=False)
    if args.ratings:
        if not args.mapped_output:parser.error('--ratings requires --mapped-output')
        incoming=readlines(args.ratings)
        write_lines(args.mapped_output,fanout_ratings(occurrences,unique,incoming))
        receipt.update(independent_ratings_completed=True,mapped_rated_occurrences=len(occurrences),unique_independent_ratings=len(incoming))
    if args.write:(OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
