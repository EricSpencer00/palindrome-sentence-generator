"""Deterministic one-step feasibility profile; never runs a paragraph search."""
from collections import Counter
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time

from experiments.derived_paragraph_controls_20261009 import PATH
from llm_palindrome.block_search import BlockUnit, compatible_actions
from llm_palindrome.block_seams import Seam
from llm_palindrome.grammar_boundaries import GrammarBoundaryIndex
from llm_palindrome.typed_constituents import TypedGrammar, words

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/block-seams'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(mode):
    if mode not in ('baseline','cached'):raise ValueError('baseline or cached mode required')
    out=OUT/f'cold-position-profile-001-{mode}.json'
    if out.exists():raise RuntimeError('preserve immutable profile receipts')
    inventory_path=OUT/'licensed-control-inventory-v2-001.json'
    inventory=json.loads(inventory_path.read_text())
    grammar=TypedGrammar(inventory['words'])
    index=GrammarBoundaryIndex(grammar)
    units=tuple(BlockUnit('profile-'+str(i),text) for i,text in enumerate(inventory['blocks']))
    bytext={u.text:u for u in units}
    fixtures=[]
    for side,text in [('left','no rider'),('left','liam'),('left','no evil'),('right','red iron')]:
        fixtures.append(Seam().add(side,__import__('llm_palindrome.block_seams',fromlist=['Piece']).Piece(bytext[text].id,0,text)))
    state=Seam()
    for depth,(side,text) in enumerate(PATH,1):
        seq=state.left if side=='left' else state.right
        state=state.add(side,__import__('llm_palindrome.block_seams',fromlist=['Piece']).Piece(bytext[text].id,len(seq),text))
        assert state is not None
        if depth in (6,11):fixtures.append(state)
    trace_path=OUT/'boundary-root-diagnostic-001-results.json.gz'
    calls=[]
    def frontier(state):
        left=words(' '.join(p.text for p in state.left))
        right=words(' '.join(p.text for p in state.right))
        calls.append((left,right,4))
        return grammar.paragraph_frontier(left,right,4)
    def evaluate():
        return [{'left':[p.text for p in state.left],'right':[p.text for p in state.right],
            'frontier':frontier(state),'menu':[{'side':a.side,'unit':a.unit.text,
                'debt':a.child.debt()} for a in compatible_actions(state,units,
                    grammar_accept=frontier,boundary_accept=index.allows_state,
                    max_letters=119,max_words=64)]} for state in fixtures]
    profiler=cProfile.Profile()
    times=[];signatures=[]
    profiler.enable()
    for repeat in range(4):
        start=time.perf_counter();result=evaluate();times.append(time.perf_counter()-start)
        signatures.append(hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest())
    profiler.disable()
    assert len(set(signatures))==1
    stats=pstats.Stats(profiler)
    top=[]
    for (filename,line,name),(primitive,total,self_seconds,cumulative,callers) in sorted(
            stats.stats.items(),key=lambda item:-item[1][3])[:20]:
        top.append({'function':name,'module':Path(filename).name,'line':line,
                    'primitive_calls':primitive,'total_calls':total,
                    'self_seconds':self_seconds,'cumulative_seconds':cumulative})
    cache_info=(grammar.frontier_cache_info()._asdict()
                if hasattr(grammar,'frontier_cache_info') else None)
    record={'schema_version':1,'mode':mode,'scope':'Six fixed seam states; one-step menus, four identical passes. No beam search, scorer, model, or candidate search invoked.',
        'fixture_sources':'Four declared root actions and parent-provided construction states at depths 6 and 11.',
        'inventory_sha256':sha(inventory_path),'source_trace_sha256':sha(trace_path),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__),ROOT/'llm_palindrome/typed_constituents.py',
             ROOT/'llm_palindrome/grammar_boundaries.py',ROOT/'llm_palindrome/block_search.py')},
        'fixtures':[{'left':[p.text for p in s.left],'right':[p.text for p in s.right]}
                    for s in fixtures],
        'passes':4,'pass_seconds':times,'total_seconds':sum(times),
        'frontier_invocations':len(calls),'unique_frontier_keys':len(set(calls)),
        'repeated_frontier_invocations':len(calls)-len(set(calls)),
        'menu_signatures':signatures,'last_fixture_result':result,
        'frontier_cache_info':cache_info,'position_cache_info':(grammar.positions_cache_info()._asdict() if hasattr(grammar,'positions_cache_info') else None),'profile_top':top,
        'search_invocations':0,'model_calls':0,'paid_resources':0}
    out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('mode','pass_seconds','total_seconds',
        'frontier_invocations','unique_frontier_keys','repeated_frontier_invocations',
        'frontier_cache_info','menu_signatures')},indent=2))


if __name__=='__main__':profile(sys.argv[1])
