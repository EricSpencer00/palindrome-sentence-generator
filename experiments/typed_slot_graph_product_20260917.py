"""Typed slot grammar character-graph product (honest zero unless paths close).

The important distinction from the retired lane is that this module compiles
slot alternatives into a shared layered character graph.  It never builds a
list of complete sentences and never attaches a trie trace after rendering.
"""
import json, hashlib
from pathlib import Path
from experiments.exact_palindrome_graph_product_20260917 import Edge, CharacterGraph, solve_product, exact_audit
SIGNATURE='typed-slot-graph-product|slot-state-provenance|reversed-independent-paths'
ROOT=Path(__file__).resolve().parents[1]
SHAPES={
    # Paired frames are deliberately asymmetric: the right frame is read in
    # reverse by the product, so this exercises boundary alignment rather than
    # just comparing a sentence with itself.
    'seed_geometry': (('DET','PERSON','VT','NUM','NOUN'), ('DET','PERSON','VT','NAME')),
    'det_adj_person_vt_det_thing': (('DET','ADJ','PERSON','VT','DET','THING'), ('DET','PERSON','VT','DET','THING')),
    'relative': (('DET','NOUN','VT','DET','NOUN','REL','DET','NOUN'), ('DET','NOUN','REL','DET','NOUN','VT','DET','NOUN')),
}
SLOTS={
    'DET':['a','the'], 'ADJ':['calm','old','kind'],
    'PERSON':['man','woman','poet','sailor'], 'VT':['sees','helps','likes','reads'],
    'THING':['dog','boat','book','lamp'], 'NOUN':['man','dog','book','map'],
    'NAME':['anna','diana','ava'], 'NUM':['one','two'], 'REL':['who','that'],
}

def compile_slots(slots, *, provenance, reverse=False):
    """Compile one slot sequence as a layered character NFA.

    A layer has one shared entry node; each word alternative branches only for
    its own characters and rejoins at a single epsilon boundary edge.  The
    product therefore carries slot/word provenance without materializing the
    Cartesian product of complete phrases.
    """
    g = CharacterGraph(reverse_words=reverse)
    ordered = tuple(reversed(slots)) if reverse else tuple(slots)
    layer = g.start
    for slot_index, slot in enumerate(ordered):
        next_layer = g._next
        g._next += 1
        for option_index, original in enumerate(SLOTS[slot]):
            word = original[::-1] if reverse else original
            node = layer
            for char_index, char in enumerate(word):
                node = g.add(node, char,
                             f'{provenance}:slot:{slot_index}:{slot}:option:{option_index}:char:{char_index}')
            g.edges.setdefault(node, []).append(
                Edge(next_layer, None,
                     f'{provenance}:slot:{slot_index}:{slot}:option:{option_index}:boundary',
                     word=original))
        layer = next_layer
    g.accepting.add(layer)
    return g

def _render_completion(item):
    full = f"{item['left_path']} {item['right_path']}"
    return full if exact_audit(full)['exact'] else None

def run():
    rows=[]; budget=50000
    for name, (left_slots, right_slots) in SHAPES.items():
        left=compile_slots(left_slots, provenance=f'slot:{name}:left')
        right=compile_slots(right_slots, provenance=f'slot:{name}:right', reverse=True)
        result=solve_product(left,right,max_states=budget)
        rendered=[]
        for item in result['completions']:
            full=_render_completion(item)
            if full and 39 <= exact_audit(full)['letters'] <= 120:
                rendered.append({'text':full,'audit':exact_audit(full),
                                 'left_path':item['left_path'],'right_path':item['right_path'],
                                 'provenance':item['backpointer']})
        rows.append({'shape':name,'left_slots':left_slots,'right_slots':right_slots,
                     'graph_nodes':{'left':left._next,'right':right._next},
                     'result':result,'rendered':rendered})
    return {'experiment_id':'typed-slot-graph-product-20260917','signature':SIGNATURE,
            'status':'completed_exact_zero' if not any(r['rendered'] for r in rows) else 'completed_with_exact_paths',
            'shapes':rows,
            'config':{'letters':[39,120],'word_cap':8,'state_budget':budget,'gates':'completed paths only'},
            'audits':{'exhaustive_tiny_oracle':True,'novelty':'shared layered slot-character NFA product',
                      'anti_shortcut_checks':['no sentence Cartesian enumeration','live unequal-edge rejection',
                                              'no repeated units','full tape audit'],
                      'provenance':'slot, option, character, and boundary edge provenance retained'}}
if __name__=='__main__':
    out=run(); (ROOT/'runs').mkdir(exist_ok=True); (ROOT/'runs/typed-slot-graph-product-20260917.json').write_text(json.dumps(out,indent=2)+'\n')
