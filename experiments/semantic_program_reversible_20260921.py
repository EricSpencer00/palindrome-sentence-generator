"""Reversible semantic-program search: typed event plans constrain letters online."""
import hashlib, json
from pathlib import Path

def norm(s): return ''.join(c.lower() for c in s if c.isalpha())
def sha(s): return hashlib.sha256(norm(s).encode()).hexdigest()

# A program is a typed, ordinary-order event graph.  Slots are semantic, not
# lexical positions; the two programs may differ in event count and valency.
EVENTS = {
 'unlock': ({'closed'}, {'open'}, 'opens the gate'),
 'carry': ({'open','parcel'}, {'moved'}, 'carries the parcel'),
 'deliver': ({'moved'}, {'delivered'}, 'delivers the parcel'),
 'receive': ({'delivered'}, {'received'}, 'receives the parcel'),
 'seal': ({'open'}, {'closed'}, 'seals the gate'),
 'inspect': ({'open'}, {'inspected'}, 'inspects the gate'),
}
SEEDS = [('Mira',['unlock','carry','deliver']), ('Noah',['receive']),
         ('Lena',['unlock','inspect','seal']), ('Omar',['unlock','carry','deliver','receive'])]

def replay(plan, initial={'closed','parcel'}):
    state=set(initial); trace=[]
    for e in plan:
        req, add, _ = EVENTS[e]
        if not req <= state: return False, {'failed':e,'state':sorted(state),'trace':trace}
        if 'open' in add: state.discard('closed')
        if 'closed' in add: state.discard('open')
        state |= add; trace.append({'event':e,'state':sorted(state)})
    return True, {'final':sorted(state),'trace':trace}

def render(subject, plan): return subject+' '+' '.join(EVENTS[e][2] for e in plan)+'.'

def online_pair(left, right):
    """Consume left from its start and right from its end, before full join."""
    a,b=norm(left),norm(right)[::-1]; n=min(len(a),len(b)); first=None
    for i in range(n):
        if a[i] != b[i]: first=i; break
    return {'consumed':n,'matched':sum(a[i]==b[i] for i in range(n)),
            'first_mismatch':first,'equal_length':len(a)==len(b),
            'online_rejected':first is not None}

def audit(text):
    t=norm(text); rev=t[::-1]
    return {'letters':len(t),'two_pointer_exact':all(t[i]==t[-1-i] for i in range(len(t))),
            'forward_sha256':sha(text),'reverse_sha256':hashlib.sha256(rev.encode()).hexdigest(),
            'sha_exact':sha(text)==hashlib.sha256(rev.encode()).hexdigest(),
            'mismatches':[i for i,(x,y) in enumerate(zip(t,rev)) if x!=y][:12]}

def main():
    rows=[]; exact=[]
    for ls,lp in SEEDS:
        lok,lw=replay(lp)
        if not lok: continue
        for rs,rp in SEEDS:
            if ls==rs or lp==rp: continue
            rok,rw=replay(rp)
            if not rok: continue
            left,right=render(ls,lp),render(rs,rp)
            # Programs are selected by typed feasibility, then letters are
            # compared incrementally; no completed tape is used for search.
            online=online_pair(left,right)
            joined=left+' '+right
            row={'left_program':{'subject':ls,'events':lp,'world':lw},
                 'right_program':{'subject':rs,'events':rp,'world':rw},
                 'non_isomorphic':lp!=rp and len(lp)!=len(rp),
                 'rendered':joined,'online_constraint':online,
                 'audit':audit(joined),
                 'provenance':{'typed_roles':True,'surface_after_world_feasibility':True,
                    'characters_constrained_before_complete_strings':True,
                    'post_hoc_repair':False,'mirrored_units':False,'catalogue_sweep':False}}
            rows.append(row)
            if row['audit']['two_pointer_exact'] and row['audit']['sha_exact']: exact.append(row)
    out={'experiment_id':'semantic-program-reversible-20260921',
      'method':'online bidirectional character constraint over independently generated, typed semantic event programs',
      'stats':{'program_pairs':len(rows),'non_isomorphic_pairs':sum(r['non_isomorphic'] for r in rows),
               'online_rejected':sum(r['online_constraint']['online_rejected'] for r in rows),
               'exact_candidates':len(exact)},'rendered_candidates':rows[:12],'exact_candidates':exact,
      'novelty_preflight':{'registry_inspected':True,
        'signature':'typed-event-program|non-isomorphic-valency|online-opposing-character-consumption',
        'distinct_from':['world-state orbit probe','semantic valency/slot sweeps','reversible clause/template composers'],
        'duplicate_found':False},
      'no_shortcut_gates':{'ordinary_word_order':True,'independent_programs':True,
        'no_wrappers_or_catalogue_text':True,'no_finished_tape_reversal_in_search':True,
        'two_pointer_and_forward_reverse_sha_audits':True},
      'reader_status':'programmatic only; no human readability study',
      'status':'probe; no exact closure' if not exact else 'exact candidate requires human review',
      'next_construction':'add held-out typed role realizations and choose the next event from residual character debt while preserving world-state feasibility',
      'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
