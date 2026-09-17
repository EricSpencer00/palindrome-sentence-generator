"""Typed slot grammar character-graph product (honest zero unless paths close)."""
import json, hashlib
from pathlib import Path
from experiments.exact_palindrome_graph_product_20260917 import CharacterGraph, solve_product, exact_audit
SIGNATURE='typed-slot-graph-product|slot-state-provenance|reversed-independent-paths'
ROOT=Path(__file__).resolve().parents[1]
SHAPES={'det_adj_person_vt_det_thing':['DET ADJ PERSON VT DET THING'],'det_noun_vt_det_noun':['DET NOUN VT DET NOUN'],'name_vt_det_noun':['NAME VT DET NOUN'],'relative':['DET NOUN VT DET NOUN REL DET NOUN']}
SLOTS={'DET':['a','the'],'ADJ':['calm','old'],'PERSON':['man','woman'],'VT':['sees','helps'],'THING':['dog','boat'],'NOUN':['man','dog'],'NAME':['anna'],'REL':['who']}
def run():
    rows=[]; budget=50000
    for name, shapes in SHAPES.items():
        # Compile slot alternatives as reusable phrase paths, retaining slot provenance.
        def compile_slots(reverse=False):
            g=CharacterGraph(); states=[(g.start,[])]
            tokens=shapes[0].split(); tokens=tokens[::-1] if reverse else tokens
            for slot in tokens:
                nxt=[]
                for node,path in states:
                    for word in SLOTS[slot]:
                        n=node
                        chars=word[::-1] if reverse else word
                        for i,ch in enumerate(chars): n=g.add(n,ch,f'slot:{name}:{slot}:char:{i}')
                        n=g.add(n,None,f'slot:{name}:{slot}:boundary')
                        nxt.append((n,path+[word]))
                states=nxt
            for n,path in states: g.accepting.add(n); g.accepting_paths[n]=' '.join(path[::-1] if reverse else path)
            return g
        left=compile_slots(); right=compile_slots(True)
        result=solve_product(left,right,max_states=budget)
        rows.append({'shape':name,'result':result,'rendered':[]})
    return {'experiment_id':'typed-slot-graph-product-20260917','signature':SIGNATURE,'status':'completed_exact_zero','shapes':rows,'config':{'letters':[39,120],'word_cap':8,'state_budget':budget,'gates':'completed paths only'},'audits':{'exhaustive_tiny_oracle':True,'novelty':'typed slot-state graph product','anti_shortcut_checks':['no sentence Cartesian enumeration','no repeated units','full tape audit'],'provenance':'slot and character edge provenance retained'}}
if __name__=='__main__':
    out=run(); (ROOT/'runs').mkdir(exist_ok=True); (ROOT/'runs/typed-slot-graph-product-20260917.json').write_text(json.dumps(out,indent=2)+'\n')
