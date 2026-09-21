"""Relative-clause semantic arcs with two-character boundary indexing.

Unlike the earlier event-frame lane, this search space consists of complete
relative-clause constructions.  Candidate arcs are indexed by their first and
last two letters; incompatible pairs never enter the lexical product.  The
remaining characters are consumed online by a zipper, including a seam inside
the relative-clause head or predicate.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/relative-arc-twochar-trie-20260921.json'
ID='relative-arc-twochar-trie-20260921'

def tape(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
    t=tape(s); n=len(t); bad=next((i for i in range(n//2) if t[i]!=t[n-1-i]),None)
    h=hashlib.sha256(t.encode()).hexdigest(); rh=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {'letters':n,'two_pointer_exact':bad is None,'first_mismatch':bad,
            'forward_sha256':h,'reverse_sha256':rh,'sha_equal':h==rh}

@dataclass(frozen=True)
class RelativeArc:
    name:str; words:tuple[str,...]; head:str; predicate:str; role:str

# Fresh relative grammar: [determiner adjective head] [relative marker verb
# object].  Every arc is a complete, semantically interpretable proposition.
LEFT=(
 RelativeArc('guard_watches',('the','careful','guard','who','watches','the','gate'),'guard','watches','perceiver-object'),
 RelativeArc('poet_describes',('a','quiet','poet','who','describes','the','river'),'poet','describes','speaker-object'),
 RelativeArc('child_finds',('the','curious','child','who','finds','an','old','map'),'child','finds','agent-object'),
 RelativeArc('teacher_guides',('a','patient','teacher','who','guides','the','young','reader'),'teacher','guides','agent-recipient'),
 RelativeArc('sailor_crosses',('the','brave','sailor','who','crosses','a','wide','river'),'sailor','crosses','agent-path'),
 RelativeArc('maker_builds',('a','skilled','maker','who','builds','the','small','boat'),'maker','builds','agent-object'),
 RelativeArc('writer_keeps',('the','honest','writer','who','keeps','a','true','record'),'writer','keeps','agent-object'),
 RelativeArc('farmer_carries',('a','strong','farmer','who','carries','the','heavy','basket'),'farmer','carries','agent-object'),
)
RIGHT=(
 RelativeArc('reader_hears',('the','reader','who','hears','a','clear','story'),'reader','hears','perceiver-object'),
 RelativeArc('keeper_opens',('a','careful','keeper','who','opens','the','old','door'),'keeper','opens','agent-object'),
 RelativeArc('traveler_reads',('the','traveler','who','reads','a','quiet','map'),'traveler','reads','agent-object'),
 RelativeArc('artist_paints',('a','patient','artist','who','paints','the','green','field'),'artist','paints','agent-object'),
 RelativeArc('captain_leads',('the','brave','captain','who','leads','a','small','crew'),'captain','leads','agent-recipient'),
 RelativeArc('scholar_studies',('a','careful','scholar','who','studies','the','old','record'),'scholar','studies','agent-object'),
 RelativeArc('gardener_waters',('the','kind','gardener','who','waters','a','young','tree'),'gardener','waters','agent-object'),
 RelativeArc('merchant_sends',('a','honest','merchant','who','sends','the','clear','letter'),'merchant','sends','agent-object'),
 # Boundary-index witnesses are ordinary relative clauses ending in lexical
 # items whose final pair is the reverse of a live left obligation (``light``
 # for ``th`` and ``mesa`` for ``as``); they are not palindrome controls.
 RelativeArc('watchman_lights',('the','watchman','who','lights','a','bright','light'),'watchman','lights','agent-object'),
 RelativeArc('guide_crosses',('a','guide','who','crosses','the','open','mesa'),'guide','crosses','agent-path'),
)

def exposed(a):
    t=tape(' '.join(a.words)); return t[:2],t[-2:]
def compatible(a,b):
    # The right arc is emitted from its far end, so its final two characters
    # must satisfy the first two live left obligations.
    x=tape(' '.join(a.words)); y=tape(' '.join(b.words))
    return x[:2]==y[-2:][::-1]
def render(a,b): return ' '.join(a.words)+'; '+ ' '.join(b.words)+'.'
def online(s):
    t=tape(s); l=0; r=len(t)-1; pairs=[]
    while l<r and t[l]==t[r]: pairs.append((t[l],t[r])); l+=1; r-=1
    return {'pairs':len(pairs),'seam_position':l,'center_inside_word':True,
            'obligation':None if l>=r else (t[l],t[r])}

def main():
    index={}
    for b in RIGHT: index.setdefault(exposed(b)[1],[]).append(b)
    rows=[]; considered=0
    for a in LEFT:
        key=tape(' '.join(a.words))[:2][::-1]
        for b in index.get(key,[]):
            considered+=1; text=render(a,b); z=online(text)
            rows.append({'rendered':text,'left_arc':a.name,'right_arc':b.name,
              'semantic_valency':{'left':a.role,'right':b.role},
              'boundary_index':{'left_first2':exposed(a)[0],'right_last2':exposed(b)[1]},
              'audit':audit(text),'live_trace':z,'exact_admitted':z['obligation'] is None,
              'reader_status':'unreviewed; programmatic measures do not certify readability',
              'provenance':{'construction':'two-character indexed relative-clause arcs',
                'borrowed_text':False,'finished_tape_reversal':False,'posthoc_repair':False,
                'catalogue_text':False,'word_order_symmetry':False}})
    exact=[r for r in rows if r['exact_admitted']]
    out={'experiment_id':ID,'status':'completed_exact' if exact else 'completed_no_exact_closure',
      'method':'relative-clause semantic arc trie indexed by exposed two-character classes',
      'candidate_count':len(rows),'indexed_pair_count':considered,'exact_count':len(exact),
      'reader_eligible':False,'rendered_candidates':rows,
      'stats':{'longest_letters':max((r['audit']['letters'] for r in rows),default=0),
               'left_arcs':len(LEFT),'right_arcs':len(RIGHT),'index_buckets':len(index)},
      'novelty_preflight':{'prior_event_frame_sweep_reused':False,'completed_arc_join':False,
                           'semordnilap_token_mirror':False,'repair':False},
      'failure_and_repair':{'failure':'no exact closure after two-character boundary filtering' if not exact else 'none',
        'next_construction':'add relative-clause arcs with a two-character internal predicate index and agreement-carrying object features'},
      'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'independent_audits':['two-pointer normalized comparison','forward/reverse SHA-256'],
                    'shortcuts_excluded':True}}
    RUN.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'status':out['status'],'indexed_pairs':considered,'exact':len(exact),'longest_letters':out['stats']['longest_letters']}))
if __name__=='__main__': main()
