"""Finite phrase-slot DAG; exact reversal debt is solved at character offsets.
A phrase is an annotated partial grammatical unit, never a sentence bank.
"""
from collections import defaultdict,deque
from dataclasses import dataclass
from .bidirectional_lexical import GrammarDAG,Arc
from .admission import normalize_letters,ALLOWED_RENDERING
@dataclass(frozen=True)
class Phrase:
 id:str
 text:str
 entry:str
 exit:str
 role:str
 source:str
 bindings:tuple=()

class PhraseInventoryDAG(GrammarDAG):
 def __init__(self,inventory,plans):
  if len(inventory)>48 or len(plans)>8:raise ValueError('inventory/plan input budget')
  self.inventory={p.id:p for p in inventory};self.plans=tuple(plans);self.frames=();self.sentences=1
  if len(self.inventory)!=len(inventory):raise ValueError('duplicate fragment ID')
  for p in inventory:
   if not p.id or not p.role or not p.source or not normalize_letters(p.text):raise ValueError('incomplete fragment')
   if len(normalize_letters(p.text))>160:raise ValueError('phrase character budget')
   if any(not isinstance(k,str) or not isinstance(v,str) or not k or not v for k,v in p.bindings):raise ValueError('invalid actor binding')
   if not all(c.isascii() and (c.isalpha() or c.isspace() or c in "'-.,;:!?") for c in p.text):raise ValueError('unsupported phrase rendering')
  self.out=defaultdict(list);self.inc=defaultdict(list);self.eps=defaultdict(set);self.reverse_eps=defaultdict(set);self.arcs=[];self.nodes=0
  def node():n=self.nodes;self.nodes+=1;return n
  def eps(a,b):self.eps[a].add(b);self.reverse_eps[b].add(a)
  self.start=node();self.accept=node()
  for fi,plan in enumerate(self.plans):
   states=plan['states'];slots=plan['slots']
   if not 1<=len(slots)<=12:raise ValueError('phrase slot budget')
   if len(states)!=len(slots)+1 or states[0]!='START' or states[-1]!='END':raise ValueError('plan state topology')
   at=node();eps(self.start,at)
   for si,options in enumerate(slots):
    end=node()
    if not options:raise ValueError('empty phrase slot')
    for pid in options:
     p=self.inventory[pid]
     if (p.entry,p.exit)!=(states[si],states[si+1]):raise ValueError('unlicensed phrase seam:'+pid)
     current=at
     for offset,ch in enumerate(normalize_letters(p.text)):
      if len(self.arcs)>=20000:raise ValueError('compiled character arc budget')
      target=node();aid=len(self.arcs);self.arcs.append(Arc(current,target,ch,0,fi,si,pid,p.role,offset));self.out[current].append(aid);self.inc[target].append(aid);current=target
     eps(current,end)
    at=end
   eps(at,self.accept)
  successors={n:set(self.eps[n])|{self.arcs[a].target for a in self.out[n]} for n in range(self.nodes)};ind=[0]*self.nodes
  for ds in successors.values():
   for d in ds:ind[d]+=1
  q=deque(n for n,d in enumerate(ind) if d==0);order=[]
  while q:
   n=q.popleft();order.append(n)
   for d in successors[n]:
    ind[d]-=1
    if ind[d]==0:q.append(d)
  if len(order)!=self.nodes:raise ValueError('cycle')
  self.reachable=[0]*self.nodes
  for n in reversed(order):
   mask=1<<n
   for d in successors[n]:mask|=self.reachable[d]
   self.reachable[n]=mask

 def materialize(self,path):
  current=self.start
  for aid in path:
   arc=self.arcs[aid];assert arc.source in self.closure(current);current=arc.target
  assert self.accept in self.closure(current)
  arcs=[self.arcs[a] for a in path];starts=[a for a in arcs if a.offset==0];fi=starts[0].frame;plan=self.plans[fi]
  assert all(a.frame==fi for a in arcs);assert [a.slot for a in starts]==list(range(len(plan['slots'])))
  phrases=[self.inventory[a.word] for a in starts];bindings={};conflicts=[]
  for i,p in enumerate(phrases):
   assert p.id in plan['slots'][i];assert (p.entry,p.exit)==tuple(plan['states'][i:i+2])
   for k,v in p.bindings:
    if k in bindings and bindings[k]!=v:conflicts.append(dict(variable=k,previous=bindings[k],new=v))
    bindings[k]=v
  text=''.join(p.text for p in phrases);t=normalize_letters(text);assert t==t[::-1];assert ALLOWED_RENDERING.fullmatch(text)
  return dict(text=text,tape=t,letters=len(t),plan=plan['id'],method=plan['method'],phrase_ids=[p.id for p in phrases],phrases=[dict(id=p.id,text=p.text,role=p.role,entry=p.entry,exit=p.exit,source=p.source,bindings=p.bindings) for p in phrases],actor_bindings=bindings,binding_conflicts=conflicts,role_consistent=not conflicts,source_claim='finite authored grammar states; no independent readability certificate',center=dict(phrase_id=arcs[len(arcs)//2].word,offset=arcs[len(arcs)//2].offset),character_arc_ids=list(path))

 @staticmethod
 def from_json_api(payload):
  """Validate finite parent/native fragment JSON before compiling search."""
  if not isinstance(payload,dict) or not isinstance(payload.get('fragments'),list) or not isinstance(payload.get('plans'),list):raise ValueError('fragments and plans lists required')
  ps=[]
  for row in payload['fragments']:
   required=('id','text','entry','exit','role','source')
   if not isinstance(row,dict) or any(not isinstance(row.get(k),str) or not row[k] for k in required):raise ValueError('invalid fragment fields')
   b=row.get('bindings',{})
   if not isinstance(b,dict):raise ValueError('bindings object required')
   ps.append(Phrase(*(row[k] for k in required),tuple(sorted(b.items()))))
  return PhraseInventoryDAG(ps,payload['plans'])

def search_fragment_api(payload):
 """Bounded native fragment input -> exact paths/role conflicts/receipt."""
 from .bidirectional_lexical import exact_grammar_palindromes,SearchBudgetExceeded
 grammar=PhraseInventoryDAG.from_json_api(payload)
 try:
  paths,receipt=exact_grammar_palindromes(grammar,max_work=100000,max_paths=2000,seconds=5)
  rows=[grammar.materialize(p) for p in paths]
 except SearchBudgetExceeded as exc:rows=[];receipt=exc.receipt
 finally:grammar.closure.cache_clear();grammar.transitions.cache_clear()
 return dict(outputs=rows,receipt=receipt,model_quality_scores=0,grammar_claim='annotated finite state grammar only',role_conflicts=sum(not r['role_consistent'] for r in rows),quality_unrated=True)

def fragment_frontier(payload,plan_id,left_ids=(),right_ids=()):
 """Actionable local constraints, not a promise of full grammatical closure.
 Right IDs are suffix fragments in normal reading order. Alternatives are
 restricted to the next open grammar slots and tested against mirrored tape
 and actor bindings. Root endpoints can be rejected without product search.
 """
 from .phrase_api_loop import mirror_debt
 g=PhraseInventoryDAG.from_json_api(payload)
 plan=next((p for p in g.plans if p['id']==plan_id),None)
 if plan is None:raise ValueError('unknown plan')
 slots=plan['slots'];n=len(slots);l=list(left_ids);r=list(right_ids)
 if len(l)+len(r)>n:raise ValueError('frontier overlaps')
 for i,pid in enumerate(l):
  if pid not in slots[i]:raise ValueError('left fragment outside grammar slot')
 for i,pid in enumerate(r,n-len(r)):
  if pid not in slots[i]:raise ValueError('right fragment outside grammar slot')
 def inspect(a,b):
  bindings={};conflicts=[]
  for pid in a+b:
   for k,v in g.inventory[pid].bindings:
    if k in bindings and bindings[k]!=v:conflicts.append(dict(variable=k,previous=bindings[k],proposed=v))
    bindings[k]=v
  left=''.join(g.inventory[x].text for x in a);right=''.join(g.inventory[x].text for x in b)
  return dict(left_text=left,right_text=right,letter_debt=mirror_debt(left,right),actor_bindings=bindings,binding_conflicts=conflicts)
 current=inspect(l,r);remaining=n-len(l)-len(r)
 def choice(pid,side):
  state=inspect(l+[pid],r) if side=='left' else inspect(l,[pid]+r)
  p=g.inventory[pid];full=normalize_letters(state['left_text']+state['right_text']);complete_exact=remaining==1 and full==full[::-1]
  return dict(fragment_id=pid,text=p.text,entry=p.entry,exit=p.exit,role=p.role,source=p.source,compatible=state['letter_debt']['compatible'] and not state['binding_conflicts'] and (remaining!=1 or complete_exact),complete_exact=complete_exact,**state)
 alternatives={s:[] for s in ['left','right']}
 if remaining:
  alternatives['left']=[choice(pid,'left') for pid in slots[len(l)]]
  alternatives['right']=[choice(pid,'right') for pid in slots[n-len(r)-1]]
 pairs=[]
 if remaining>=2:
  for a in slots[len(l)]:
   for b in slots[n-len(r)-1]:
    state=inspect(l+[a],[b]+r);pairs.append(dict(left_fragment=a,right_fragment=b,compatible=state['letter_debt']['compatible'] and not state['binding_conflicts'],**state))
 completed=remaining==0;exact=completed and normalize_letters(current['left_text']+current['right_text'])==normalize_letters(current['left_text']+current['right_text'])[::-1]
 result=dict(plan_id=plan_id,selected_left=l,selected_right=r,remaining_slots=remaining,left_grammar_state=plan['states'][len(l)],right_grammar_state=plan['states'][n-len(r)],current=current,alternatives=alternatives,paired_endpoint_alternatives=pairs,immediate_rejection=not current['letter_debt']['compatible'] or bool(current['binding_conflicts']) or (bool(pairs) and not any(p['compatible'] for p in pairs)),complete_exact=exact,viability_scope='finite supplied phrase choices, local reverse-debt and binding checks; full product search still verifies closure',generation_instruction='Propose alternative partial grammatical phrases for these open states and exact prefix/suffix debt. Preserve actor identity and a coherent premise. Do not append filler or an independently mirrored sentence; grammar/meaning still require review.')
 g.closure.cache_clear();g.transitions.cache_clear();return result

def conditioned_lattice(payload,plan_id,left_ids=(),right_ids=()):
 """Return only finite continuations with a verified exact full-path witness."""
 import copy
 p=copy.deepcopy(payload);plan=next((x for x in p['plans'] if x['id']==plan_id),None)
 if plan is None:raise ValueError('unknown plan')
 frontier=fragment_frontier(payload,plan_id,left_ids,right_ids)
 if frontier['immediate_rejection']:return dict(frontier=frontier,outputs=[],continuation_slots=[],exact_witnesses=0,status='endpoint_rejected')
 n=len(plan['slots'])
 for i,pid in enumerate(left_ids):plan['slots'][i]=[pid]
 for i,pid in enumerate(right_ids,n-len(right_ids)):plan['slots'][i]=[pid]
 p['plans']=[plan];used={i for slot in plan['slots'] for i in slot};p['fragments']=[r for r in p['fragments'] if r['id'] in used]
 result=search_fragment_api(p);valid=[r for r in result['outputs'] if r['role_consistent']]
 slots=[]
 for i in range(len(left_ids),n-len(right_ids)):
  ids=sorted({r['phrase_ids'][i] for r in valid});slots.append(dict(slot=i,entry=plan['states'][i],exit=plan['states'][i+1],fragments=[dict(fragment=next(f for f in p['fragments'] if f['id']==pid),witness_count=sum(r['phrase_ids'][i]==pid for r in valid)) for pid in ids]))
 return dict(frontier=frontier,outputs=valid,continuation_slots=slots,exact_witnesses=len(valid),status='complete' if result['receipt']['complete'] else 'incomplete',receipt=result['receipt'],semantic_selection='Luna chooses among exact witness paths; no independent grammar/coherence claim')

def letter_admit_alternative(payload,plan_id,left_ids,right_ids,side,fragment):
 """No inventory mutation: certify alternative via debt then full witness."""
 import copy
 from .phrase_api_loop import mirror_debt
 if side not in ('left','right'):raise ValueError('side required')
 frontier=fragment_frontier(payload,plan_id,left_ids,right_ids);n=frontier['remaining_slots']
 if n==0:return dict(admitted=False,reason='no open slot')
 plan=next(p for p in payload['plans'] if p['id']==plan_id);slot=len(left_ids) if side=='left' else len(plan['slots'])-len(right_ids)-1
 required=(plan['states'][slot],plan['states'][slot+1])
 if (fragment.get('entry'),fragment.get('exit'))!=required:return dict(admitted=False,reason='grammar_state_mismatch',required_states=required)
 if any(fragment.get('id')==p['id'] for p in payload['fragments']):return dict(admitted=False,reason='duplicate fragment ID')
 left=frontier['current']['left_text'];right=frontier['current']['right_text'];text=fragment.get('text','')
 debt=mirror_debt(left+text,right) if side=='left' else mirror_debt(left,text+right)
 if not debt['compatible']:return dict(admitted=False,reason='letter_mismatch',mismatch=debt,inventory_mutated=False)
 p=copy.deepcopy(payload);selected=copy.deepcopy(plan);selected['slots'][slot]=[fragment['id']];p['plans']=[selected];used={i for s in selected['slots'] for i in s};p['fragments']=[f for f in p['fragments'] if f['id'] in used]+[fragment]
 l=list(left_ids)+[fragment['id']] if side=='left' else list(left_ids);r=[fragment['id']]+list(right_ids) if side=='right' else list(right_ids)
 lattice=conditioned_lattice(p,plan_id,l,r)
 return dict(admitted=bool(lattice['exact_witnesses']) and lattice['status']=='complete',reason='incomplete_search_no_admission' if lattice['status']=='incomplete' else ('exact_closure_witness' if lattice['exact_witnesses'] else 'no_exact_closure_in_finite_inventory'),letter_debt=debt,witnesses=lattice['outputs'],completion_status=lattice['status'],inventory_mutated=False)
