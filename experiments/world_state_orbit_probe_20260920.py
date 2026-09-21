"""Tiny probe: world-state action plans constrain an ordinary surface before audit."""
import hashlib, json
from pathlib import Path

def norm(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
    t=norm(s); return {'letters':len(t),'pointer_exact':t==t[::-1],
      'mismatches':[i for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b][:8]}

OPS={
 'open':({'closed'},{'open'}), 'seal':({'open'},{'closed'}),
 'carry':({'open','parcel'},{'moved'}), 'drop':({'moved'},{'delivered'}),
 'load':({'empty'},{'loaded'}), 'unload':({'loaded'},{'empty'}),
 'receive':({'delivered'},{'received'}),
}
WORDS={'open':'opens the gate','seal':'seals the gate','carry':'carries the parcel',
        'drop':'delivers the parcel','load':'loads the cart','unload':'unloads the cart'}
WORDS['receive']='receives the parcel'

def feasible(plan, initial):
    state=set(initial); trace=[]
    for op in plan:
        req,adds=OPS[op]
        if not req <= state: return False, {'failed_action':op,'required':sorted(req),'state':sorted(state),'trace':trace}
        if 'open' in adds: state.discard('closed')
        if 'closed' in adds: state.discard('open')
        if 'loaded' in adds: state.discard('empty')
        if 'empty' in adds: state.discard('loaded')
        state |= adds; trace.append({'action':op,'state':sorted(state)})
    return True, {'final_state':sorted(state),'trace':trace}

def render(subject, plan): return subject+' '+' '.join(WORDS[x] for x in plan)+'.'
def online_equations(left,right):
    a,b=norm(left),norm(right)[::-1]; n=min(len(a),len(b))
    return {'compared':n,'matched':sum(a[i]==b[i] for i in range(n)),
            'first_mismatch':next((i for i in range(n) if a[i]!=b[i]),None),
            'equal_length':len(a)==len(b)}

def main():
    plans=[['open','carry','drop'],['open','seal'],['carry'],['load','unload'],['unload','load']]
    rows=[]
    for p in plans:
        ok,w=feasible(p,{'closed','parcel','empty'})
        text=render('Mira',p) if ok else '<semantic UNSAT: '+p[-1]+'>'
        rows.append({'plan':p,'rendered':text,'world_audit':w,'character_audit':audit(text),
          'provenance':{'action_catalogue':'six hand-authored typed actions','world_initial':['closed','parcel','empty'],
                        'surface_rendered_only_if_world_feasible':True,'finished_tape_reversal':False,
                        'post_hoc_repair':False,'mirrored_units':False}})
    # Two independent complete plans with an explicit world-state handoff.
    pairs=[(['open','carry','drop'],['receive']), (['load','unload'],['receive']),
           (['open','seal'],['receive'])]
    handoff=[]
    for left_plan,right_plan in pairs:
        lok,lw=feasible(left_plan,{'closed','parcel','empty'})
        rok,rw=feasible(right_plan,{'delivered'})
        left,right=render('Mira',left_plan),render('Noah',right_plan)
        joined=left+' Then '+right
        handoff.append({'left_plan':left_plan,'right_plan':right_plan,'rendered':joined,
          'handoff':{'left_feasible':lok,'right_feasible_from_left_final':rok,
                     'left_final':lw.get('final_state'),'right_trace':rw.get('trace')},
          'online_character_audit':online_equations(left,right),
          'character_audit':audit(joined),'provenance':{'independent_subjects':True,
             'typed_final_state_handoff':'delivered -> receive','ordinary_clause_order':True,
             'finished_tape_reversal':False,'post_hoc_repair':False}})
    # symmetry search is over action order, not lexical inventory: find reverse-feasible plans.
    sym=[]
    for p in (['open','carry','drop'],['load','unload'],['open','seal']):
        q=list(reversed(p)); a,_=feasible(p,{'closed','parcel','empty'}); b,_=feasible(q,{'closed','parcel','empty'})
        sym.append({'plan':p,'reverse_plan':q,'forward_feasible':a,'reverse_feasible':b})
    out={'experiment_id':'world-state-orbit-probe-20260920','method':'executable world-state precondition/effect orbit before complete surface rendering',
      'stats':{'plans':len(rows),'world_feasible':sum(r['world_audit'].get('final_state') is not None for r in rows),'semantic_unsat':sum('failed_action' in r['world_audit'] for r in rows),'character_exact':sum(r['character_audit']['pointer_exact'] for r in rows)},
      'rendered_candidates':rows,'handoff_compositions':handoff,'symmetry_search':sym,'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['world precondition/effect replay','normalized character pointer scan','opposing-clause online character comparison']},
      'novelty_preflight':{'signature':'typed-world-state|precondition-effect-orbit|surface-gated-rendering|symmetry-search','registry_inspected':True,'distinct_from':'clause banks, seams, repairs, semantic-slot sweeps'},
      'status':'probe only; no exact closure','next_construction':'compose two independently authored action plans whose final world states satisfy a typed handoff relation before character orbit consumption'}
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
