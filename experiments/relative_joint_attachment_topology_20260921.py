"""Joint relative-clause event topology with shared participant state.

The pair is formed only when both clauses share an attachment/participant
state (agent-object, agent-path, etc.).  Internal head/relative/predicate
features are checked first; outer character obligations are queried last.
"""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.relative_arc_twochar_trie_20260921 import (LEFT,RIGHT,audit,tape,online,exposed,render,object_agreement)
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs/relative-joint-attachment-topology-20260921.json'
ID='relative-joint-attachment-topology-20260921'

def shared_state(a,b):
    # The shared participant is the semantic role after the first dash; this
    # is an attachment constraint, not lexical mirroring.
    ar=a.role.split('-',1)[-1]; br=b.role.split('-',1)[-1]
    return ar if ar==br else None
def head_class(a):
    i=a.words.index('who'); return tape(a.words[i-1])[-1]
def internal_compatible(a,b):
    return shared_state(a,b) is not None and (object_agreement(a) in {object_agreement(b),'definite'} or object_agreement(b)=='definite')
def outer_obligation(a,b):
    x=tape(' '.join(a.words)); y=tape(' '.join(b.words)); return x[:2]==y[-2:][::-1]

def main():
    internal=[]
    for a in LEFT:
        for b in RIGHT:
            if internal_compatible(a,b): internal.append((a,b))
    rows=[]
    for a,b in internal:
        text=render(a,b); z=online(text)
        rows.append({'rendered':text,'left_arc':a.name,'right_arc':b.name,
          'shared_attachment_state':shared_state(a,b),
          'internal_fill':{'head_class_left':head_class(a),'head_class_right':head_class(b),
                           'agreement_left':object_agreement(a),'agreement_right':object_agreement(b),
                           'internally_compatible':True},
          'outer_query':{'two_char_obligation_satisfied':outer_obligation(a,b),
                         'left_first2':exposed(a)[0],'right_last2':exposed(b)[1]},
          'audit':audit(text),'live_trace':z,'exact_admitted':z['obligation'] is None,
          'reader_status':'unreviewed; exactness does not certify readability',
          'provenance':{'construction':'joint participant/attachment relative topology',
            'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False,
            'word_order_symmetry':False}})
    exact=[r for r in rows if r['exact_admitted']]
    out={'experiment_id':ID,'status':'completed_exact' if exact else 'completed_no_exact_closure',
      'method':'joint relative-clause attachment topology: internal semantic fill before outer query',
      'candidate_count':len(rows),'internal_pairs':len(internal),'exact_count':len(exact),
      'reader_eligible':False,'rendered_candidates':rows,
      'stats':{'longest_letters':max((r['audit']['letters'] for r in rows),default=0),
               'outer_obligation_hits':sum(r['outer_query']['two_char_obligation_satisfied'] for r in rows),
               'shared_states':sorted(set(r['shared_attachment_state'] for r in rows))},
      'novelty_preflight':{'outer_first_rejected':True,'local_trie_key_reuse':False,
                           'shared_participant_state':True,'cartesian_widening':False},
      'failure_and_repair':{'failure':'internal attachment pairs remain outer-incompatible' if not exact else 'none',
        'next_construction':'pivot to paired relative clauses sharing a locative attachment state, with prepositional valency carried through the seam'},
      'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'independent_audits':['two-pointer normalized comparison','forward/reverse SHA-256'],'shortcuts_excluded':True}}
    RUN.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'status':out['status'],'internal_pairs':len(internal),'outer_hits':out['stats']['outer_obligation_hits'],'exact':len(exact),'longest_letters':out['stats']['longest_letters']}))
if __name__=='__main__': main()
