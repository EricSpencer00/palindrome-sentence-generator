"""Bounded local pilot; every viable root receives the same time/action quota."""
import gzip,hashlib,json,time
from pathlib import Path
from collections import Counter
from experiments.block_seam_comparison_20261009 import WordAdditiveScorer,render_paragraph
from experiments.derived_paragraph_controls_20261009 import tapes
from llm_palindrome.admission import normalize_letters
from llm_palindrome.block_search import BlockUnit,compatible_actions,block_beam_search
from llm_palindrome.block_seams import Seam
from llm_palindrome.grammar_boundaries import GrammarBoundaryIndex
from llm_palindrome.typed_constituents import TypedGrammar
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/block-seams'
def run():
 planpath=OUT/'controlled-seam-ablation-003-plan.json'
 if planpath.exists():raise RuntimeError('immutable run exists')
 inv=json.loads((OUT/'licensed-control-inventory-v2-001.json').read_text());g=TypedGrammar(inv['words']);idx=GrammarBoundaryIndex(g)
 fullparts=set()
 # Complete licensed slot alternatives; excludes unfinished lexical fragments.
 for slots,label in g.paths:
  for slot in slots:
   fullparts.update(' '.join(t) for t in slot)
 definitions=[('full_constituents',False,True,True),('full_plus_partial',True,True,True),('without_boundary',True,False,True),('without_composition',True,True,False),('cache_disabled',True,True,True)]
 settings=[]
 for name,partial,boundary,composition in definitions:
  us=tuple(BlockUnit('ab-'+str(i),t,tuple(inv['unit_provenance'].get(t,[]))) for i,t in enumerate(inv['blocks']) if partial or t in fullparts)
  def accept(s,composition=composition,cache=name!='cache_disabled'):
   l,r=tapes(s)
   return (g.paragraph_frontier(l,r,4) if cache else g._paragraph_frontier_uncached(l,r,4)) if composition else bool(g.compatible(l,r))
  boundaryfn=idx.allows_state if boundary else None
  roots=compatible_actions(Seam(),us,grammar_accept=accept,boundary_accept=boundaryfn,max_letters=119,max_words=64)
  roots.sort(key=lambda a:(a.side,a.unit.text,a.unit.id));settings.append((name,us,accept,boundaryfn,roots))
 total_lanes=sum(len(x[4]) for x in settings)*3
 quota=min(.5,60/total_lanes)
 plan={'seeds':[921,922,923],'band':[60,119],'workers':1,'aggregate_search_quota_seconds':60,'same_seconds_per_arm_seed':4,'same_actions_per_arm_seed':20000,'same_steps':16,'same_beam':1,'arms':[{'name':name,'units':len(us),'roots':len(roots)} for name,us,a,b,roots in settings],'inventory_sha256':hashlib.sha256((OUT/'licensed-control-inventory-v2-001.json').read_bytes()).hexdigest(),'scorer':'unchanged WordAdditiveScorer','partial_definition':'Full means complete licensed grammar-slot alternatives in existing inventory; plus-partial adds all historical blocks including partial source seams. This is not a general full-sentence vs fragment comparison.','comparison_limit':'Equal arm/seed time budget 4 seconds and action allowance 20000, divided across every viable root. Root-level quotas differ with root count. Inventory sizes mean action denominators differ; no linguistic conclusion from time-limited censored outputs. Wall-time truncation limits causal quality inference; caching equivalence separately tested on action-limited fixtures.','root_policy':'All viable roots per arm retained; same seed applied to every root within each seed cell; no target whitelist.'}
 planpath.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)
 scorer=WordAdditiveScorer(inv['words']);lanes=[];outputs=[];start=time.monotonic()
 for seed in plan['seeds']:
  for name,us,accept,boundary,roots in settings:
   for n,a in enumerate(roots):
    quota=4/len(roots)
    action_quota=20000//len(roots)+int(n<20000%len(roots))
    def closed(s):
     l,r=tapes(s);raw=s.text();norm=normalize_letters(raw);parsed=g.paragraph(l+r,4)
     rendered=render_paragraph(parsed) if parsed else None
     distinct=parsed is not None and len({w for w,_ in parsed})==len(parsed)
     eligible=bool(parsed and 2<=len(parsed)<=4 and distinct and norm not in inv['excluded_normalized'])
     if 60<=len(norm)<=119:
      outputs.append({'arm':name,'seed':seed,'root':n,'text':rendered or raw,'exact':norm==norm[::-1],'letters':len(norm),'finite_grammar_complete':bool(parsed),'clause_count':len(parsed) if parsed else None,'distinct_clauses':distinct,'known_or_supplied':norm in inv['excluded_normalized'],'eligible_mechanical':eligible,'coherence':'unreviewed','human_score':None})
     return eligible
    r=block_beam_search(us,scorer,initial_state=a.child,grammar_accept=accept,boundary_accept=boundary,allow_closed=closed,beam_width=1,max_steps=15,max_actions=action_quota,min_letters=60,max_letters=119,max_words=64,seed=seed,diversity=.4,deadline=time.monotonic()+quota)
    lanes.append({'arm':name,'seed':seed,'root':n,'root_side':a.side,'root_text':a.unit.text,'status':r['status'],'actions':r['attempted_actions'],'visited':r['visited_states'],'truncated':r['truncated'],'time_quota':quota,'action_quota':action_quota,'action_log':r['action_log']})
   print(name,seed,len(roots),'roots done',flush=True)
 summary={'elapsed_seconds':time.monotonic()-start,'outputs':len(outputs),'eligible_mechanical':sum(o['eligible_mechanical'] for o in outputs),'human_verified_novel_paragraphs':0,'arms':[{'name':name,'lanes':sum(l['arm']==name for l in lanes),'statuses':dict(Counter(l['status'] for l in lanes if l['arm']==name)),'actions':sum(l['actions'] for l in lanes if l['arm']==name),'visited':sum(l['visited'] for l in lanes if l['arm']==name),'outputs':sum(o['arm']==name for o in outputs)} for name,*_ in settings]}
 (OUT/'controlled-seam-ablation-003-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with gzip.open(OUT/'controlled-seam-ablation-003-results.json.gz','wt') as f:json.dump({'plan':plan,'lanes':lanes,'outputs':outputs},f)
 print(json.dumps(summary),flush=True)
if __name__=='__main__':run()
