"""One lexical repair conditioned on the recursive lane's first residual."""
from __future__ import annotations
import argparse,json,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve(); spec=importlib.util.spec_from_file_location('base',HERE.with_name('recursive_discourse_frame_product_20260917.py'));base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

# Recorded frontier was left 't' versus right 'd'.  Replace only the teacher
# object with a role-compatible ordinary noun ending in t; no broad sweep.
base.OBJECT['teacher']='the chart'
def run():
 r=base.run();r['config']['targeted_first_obligation_repair']=True
 r['provenance']={'repair':'teacher SVO object changed to the chart because recorded first residual was t versus d','condition':'only this role-compatible lexical edge; no other inventory expansion'}
 r['novelty_preflight']={'distinction':'single first-obligation-conditioned semantic lexical edge in the recursive frame product','excluded':{'seed':True,'catalogue':True,'word_order_mirror':True,'broad_lexical_sweep':True}}
 r['reader_gate']['next_repair']='if no closure, condition the next edge on the surviving residual pair and add only a role-compatible synonym'
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.write_text(json.dumps(run(),indent=2)+'\n')
