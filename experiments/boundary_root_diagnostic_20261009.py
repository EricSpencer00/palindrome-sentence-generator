"""One predeclared boundary-filtered, root-quota diagnostic; no target whitelist."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

from experiments.block_seam_comparison_20261009 import WordAdditiveScorer, render_paragraph
from experiments.derived_paragraph_controls_20261009 import tapes
from llm_palindrome.admission import normalize_letters
from llm_palindrome.block_search import BlockUnit, compatible_actions
from llm_palindrome.grammar_boundaries import GrammarBoundaryIndex
from llm_palindrome.root_partition_search import root_partition_search
from llm_palindrome.typed_constituents import TypedGrammar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/block-seams'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial_result(result):
    data = {k: value for k, value in result.items() if k not in ('terminals','remaining_beam')}
    data['terminals'] = [{**{k:v for k,v in t.items() if k!='state'},
                          'raw_text':t['state'].text()} for t in result['terminals']]
    data['remaining_beam'] = [{'score':score,'left':[p.text for p in state.left],
        'right':[p.text for p in state.right],'debt':state.debt()} for score,state in result['remaining_beam']]
    return data


def run():
    start=time.monotonic()
    deadline=start+58.0  # Reserve two seconds for final artifact output within the 60-second outer guard.
    plan_path=OUT/'boundary-root-diagnostic-001-plan.json'
    if plan_path.exists():raise RuntimeError('immutable diagnostic already declared')
    inventory_path=OUT/'licensed-control-inventory-v2-001.json'
    plan={'schema_version':1,'base_commit':'d3d92dc132eeb9181e7f5f059b8aa572322ea090',
        'inventory_version':'licensed-control-components-v2','inventory_sha256':sha(inventory_path),
        'workers':1,'aggregate_outer_seconds':60,'search_deadline_seconds':58,
        'max_seconds_per_root':2,'beam_width_per_root':1,'max_actions_per_root':2000,
        'max_total_steps_including_root':64,'seeds_cycle':[921,922,923],
        'diversity':.4,'letter_band':[60,119],'max_words':64,'clause_count':[2,4],
        'distinct_clauses':True,'root_policy':'Every actual viable root action retained, lexicographic side/text/ID order, one bounded quota each.',
        'within_root_policy':'Unchanged word-additive scorer, native diversity ranking and pruning; roots are not compared by score.',
        'feasibility':'Necessary character prefix/suffix condition from compiled grammar; partial lexical seams are compared as letters.',
        'target_whitelist':False,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__),ROOT/'llm_palindrome/block_search.py',ROOT/'llm_palindrome/grammar_boundaries.py',
             ROOT/'llm_palindrome/root_partition_search.py',ROOT/'llm_palindrome/typed_constituents.py',
             ROOT/'llm_palindrome/scoring.py',ROOT/'llm_palindrome/search.py')}}
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    inventory=json.loads(inventory_path.read_text())
    grammar=TypedGrammar(inventory['words'])
    boundary=GrammarBoundaryIndex(grammar)
    units=tuple(BlockUnit('v2-unit-'+str(i),text,tuple(inventory['unit_provenance'].get(text,[])))
                for i,text in enumerate(inventory['blocks']))
    callbacks=[];closures=[]
    def frontier(state):
        left,right=tapes(state)
        return grammar.paragraph_frontier(left,right,4)
    def linguistic_gate(state):
        left,right=tapes(state)
        entry={'left':[p.text for p in state.left],'right':[p.text for p in state.right],
               'status':'in_progress','phase':'grammar_frontier'}
        callbacks.append(entry)
        syntax=frontier(state)
        options=[]
        if syntax:
            entry['phase']='boundary_filtered_lookahead'
            options=compatible_actions(state,units,grammar_accept=frontier,
                boundary_accept=boundary.allows_state,deadline=deadline,
                max_letters=119,max_words=64)
        complete=grammar.paragraph(left+right,4) is not None
        reason='unlicensed_grammar_frontier' if not syntax else 'no_grammar_safe_next_option' if not options and not complete else None
        entry.update(status='checked',reason=reason,grammar_frontier_feasible=syntax,
            next_options=[{'side':a.side,'unit':a.unit.text} for a in options])
        return reason is None
    def closed(state):
        left,right=tapes(state);raw=' '.join(left+right);n=normalize_letters(raw)
        entry={'raw_text':raw,'letters':len(n),'global_exact':n==n[::-1],
               'status':'in_progress','eligible':False}
        closures.append(entry)
        parsed=grammar.paragraph(left+right,4)
        rendered=render_paragraph(parsed) if parsed else None
        written=grammar.text_paragraph(rendered,4) if rendered else None
        distinct=parsed is not None and len({tuple(w) for w,_ in parsed})==len(parsed)
        count=len(parsed) if parsed else None
        excluded=n in inventory['excluded_normalized']
        eligible=(n==n[::-1] and 60<=len(n)<=119 and written is not None
                  and 2<=count<=4 and distinct and not excluded)
        if rendered:assert normalize_letters(rendered)==n
        entry.update(status='checked',rendered_text=rendered,clause_count=count,
            distinct_clauses=distinct,complete_as_rendered=written is not None,
            known_or_supplied_control=excluded,eligible=eligible,
            coherence='unreviewed',human_readability_verified=False,
            originality='unverified',text_sha256=hashlib.sha256((rendered or raw).encode()).hexdigest())
        return eligible
    prepared=time.monotonic()-start
    scheduled=root_partition_search(units,WordAdditiveScorer(inventory['words']),boundary,
        grammar_accept=linguistic_gate,allow_closed=closed,beam_width=1,max_steps=64,
        max_actions_per_root=2000,min_letters=60,max_letters=119,max_words=64,
        seeds=(921,922,923),diversity=.4,deadline=deadline,seconds_per_root=2,
        output_reserve_seconds=1)
    for entry in callbacks+closures:
        if entry['status']=='in_progress':entry.update(status='interrupted',reason='root_or_aggregate_deadline')
    lanes=[];rows=[]
    for lane in scheduled['lanes']:
        result=lane['result']
        data={k:v for k,v in lane.items() if k!='result'}
        if result is not None:
            data['result']=serial_result(result)
            actions=result['action_log']
            layers=Counter(len(a['parent_left'])+len(a['parent_right']) for a in actions)
            rows.append({k:lane[k] for k in ('root_number','root_side','root_unit','boundary_families','seed','status','elapsed_seconds')}|{
                'attempted_actions':result['attempted_actions'],'visited_states':result['visited_states'],
                'maximum_expanded_depth':max(layers,default=1),'fully_completed_maximum_expansion_depth':max(
                    (d for d,n in layers.items() if n==142 and (d<max(layers) or result['status']=='completed')),default=0),
                'max_visited_letters':max((len(normalize_letters(' '.join(a['parent_left']+a['parent_right']))) for a in actions),default=len(normalize_letters(lane['root_unit']))),
                'max_retained_proposal_letters':max((len(normalize_letters(' '.join(a['parent_left']+a['parent_right']+[a['text']]))) for a in actions if a['status'] in ('retained_proposal','beam_pruned')),default=0),
                'rejections':dict(Counter(a.get('reason') for a in actions if a.get('reason'))),
                'pruned_proposals':sum(a['status']=='beam_pruned' for a in actions),
                'raw_closures':len(result['terminals']),'eligible_closures':sum(t['eligible'] for t in result['terminals']),
                'truncated':result['truncated'],'reported_terminal_total_scores':[t['score']+lane['root_score_delta'] for t in result['terminals']]})
        else:
            data['result']=None
            rows.append(data)
        lanes.append(data)
    summary={k:v for k,v in scheduled.items() if k!='lanes'}
    summary.update(schema_version=1,plan_sha256=sha(plan_path),preparation_seconds=prepared,
        elapsed_before_artifact_write=time.monotonic()-start,boundary_index=boundary.receipt(),
        roots=rows,closures=closures,raw_closure_presentations=len(closures),
        unique_rendered_complete_exact_texts=len({e['rendered_text'] for e in closures if e.get('complete_as_rendered') and e['global_exact']}),
        unique_eligible_texts=len({e['rendered_text'] for e in closures if e.get('eligible')}),
        independent_readability_acceptances=0,originality_claim=False)
    (OUT/'boundary-root-diagnostic-001-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    raw={'plan':plan,'boundary':boundary.receipt(),'lanes':lanes,'grammar_callbacks':callbacks,'closures':closures}
    with (OUT/'boundary-root-diagnostic-001-results.json.gz').open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:
            g.write((json.dumps(raw,separators=(',',':'))+'\n').encode())
    print(json.dumps({k:summary[k] for k in ('root_count','family_count',
        'families_with_started_root','families_without_inventory_root',
        'raw_closure_presentations','unique_rendered_complete_exact_texts','unique_eligible_texts')},indent=2))
    print('Elapsed including artifact output:',time.monotonic()-start)


if __name__=='__main__':
    run()
