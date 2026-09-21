"""Outer-obligation-first topology for paired relative clauses.

This pivot reverses the previous indexing order: two-character outer
obligations select arc pairs first; only then are head-who, predicate, and
agreement states inspected.  It is bounded and retains complete grammatical
relative clauses rather than constructing fragments.
"""
import hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.relative_arc_twochar_trie_20260921 import LEFT,RIGHT,audit,tape,online,exposed,relative_subject_class,predicate_class,object_agreement,agreement_compatible,render
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs/relative-arc-outer-topology-20260921.json'
ID='relative-arc-outer-topology-20260921'

def outer_compatible(a,b):
    x=tape(' '.join(a.words)); y=tape(' '.join(b.words))
    return x[:2]==y[-2:][::-1]

def main():
    # Topology pass is deliberately before any internal grammatical key.
    outer=[]
    for a in LEFT:
        for b in RIGHT:
            if outer_compatible(a,b): outer.append((a,b))
    rows=[]
    for a,b in outer:
        text=render(a,b); internal={
          'head_who_class':relative_subject_class(a)==relative_subject_class(b),
          'predicate_class':predicate_class(a)==predicate_class(b),
          'agreement_compatible':agreement_compatible(a,b)}
        z=online(text)
        rows.append({'rendered':text,'left_arc':a.name,'right_arc':b.name,
          'outer_boundary_obligation':{'left_first2':exposed(a)[0],'right_last2':exposed(b)[1],
                                       'reversed_match':True},
          'internal_classes':internal,'semantic_valency':{'left':a.role,'right':b.role},
          'audit':audit(text),'live_trace':z,'exact_admitted':z['obligation'] is None,
          'reader_status':'unreviewed; exactness is not readability evidence',
          'provenance':{'construction':'outer-boundary topology then relative-arc fill',
            'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False,
            'word_order_symmetry':False}})
    exact=[r for r in rows if r['exact_admitted']]
    out={'experiment_id':ID,'status':'completed_exact' if exact else 'completed_no_exact_closure',
      'method':'outer-boundary obligation topology followed by semantic relative-arc filling',
      'candidate_count':len(rows),'outer_pairs':len(outer),'exact_count':len(exact),
      'reader_eligible':False,'rendered_candidates':rows,
      'stats':{'longest_letters':max((r['audit']['letters'] for r in rows),default=0),
               'internal_head_who_matches':sum(r['internal_classes']['head_who_class'] for r in rows),
               'internal_predicate_matches':sum(r['internal_classes']['predicate_class'] for r in rows),
               'agreement_matches':sum(r['internal_classes']['agreement_compatible'] for r in rows)},
      'novelty_preflight':{'local_trie_key_reuse':False,'outer_topology_first':True,
                           'cartesian_widening':False,'repair':False},
      'failure_and_repair':{'failure':'outer-compatible pairs still mismatch after internal fill' if not exact else 'none',
        'next_construction':'use event-frame topology with relative-clause attachment as the next bounded grammar family'},
      'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'independent_audits':['two-pointer normalized comparison','forward/reverse SHA-256'],'shortcuts_excluded':True}}
    RUN.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'status':out['status'],'outer_pairs':len(outer),'exact':len(exact),'longest_letters':out['stats']['longest_letters']}))
if __name__=='__main__': main()
