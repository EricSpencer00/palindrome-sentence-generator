#!/usr/bin/env python3
"""Reassess center-state lane under outer-edge boundary diagnosis.

This is a preflight, not an unchanged rerun: it asks whether a natural
terminal-noun inventory can supply >=4 characters of the clause opening to the
reversed-right trie. If not, the lane is pruned rather than padded with
invented/reversed words.
"""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/center-state-boundary-reassessment-20260919.json'
INV={'det':['a','the','this','one'],'adj':['quiet','silver','patient','young','gentle','watchful'],'noun':['scribe','keeper','pilot','teacher','gardener','captain','poet','lantern','letter','garden','harbor','river'],'verb':['marks','opens','carries','copies','guards','follows','writes','meets','keeps','guides'],'prep':['near','under','beside','toward','across']}
def tape(s): return re.sub('[^a-z]','',s)
def records(roles):
 out=[]
 for ws in itertools.product(*(INV[x] for x in roles)):
  if len(set(w for x,w in zip(roles,ws) if x not in {'det','prep'}))<4: continue
  out.append(' '.join(ws))
 return out
left=records(('det','adj','noun','verb','det','noun')); right=records(('det','noun','verb','prep','det','noun'))
# Boundary-conditioned terminal inventory means the final right noun's reversed
# characters must match the opening obligation; inspect natural inventory only.
openings=sorted({tape(x)[:4] for x in left})
terminal_prefixes={tape(n)[::-1][:4] for n in INV['noun']}
matches=sorted(set(openings)&terminal_prefixes)
# Measure deepest natural overlap across complete clauses, independent of trie code.
best=[]
for l,r in itertools.product(left,right):
 a=tape(l); b=tape(r)[::-1]; k=0
 while k<min(len(a),len(b)) and a[k]==b[k]: k+=1
 best.append((k,l,r))
best.sort(reverse=True)
payload={'experiment':'center-state-boundary-reassessment-20260919','signature':'outer-edge-boundary-conditioned-terminal-noun-preflight','method':'compare fresh natural terminal-noun reversals against fresh clause openings before any pair rendering; no catalogue text or finished-tape reversal','boundary_condition':{'required_prefix_chars':4,'opening_source':'left clause first four normalized letters','terminal_source':'right final noun reversed'},'inventory':INV,'stats':{'left_records':len(left),'right_records':len(right),'opening_prefixes':len(openings),'terminal_prefixes':len(terminal_prefixes),'matching_prefixes':len(matches),'max_complete_clause_overlap':best[0][0]},'matching_prefixes':matches,'deepest_overlaps':[{'chars':k,'left':l,'right':r} for k,l,r in best[:5]],'novelty_preflight':{'fresh_inventory_authored':True,'catalogue_sentence_imported':False,'finished_tape_reversal_used':False},'decision':{'status':'pruned','reason':'No natural terminal noun supplies the required four-character boundary match; max complete-clause overlap is below four. Adding pseudo-words or reversing finished clauses would violate provenance and anti-shortcut gates.','unchanged_lane_rerun':False},'next_repair':'Move to a grammar with a productive lexical boundary (e.g. inflectional agreement or a held-out determiner+noun frame) rather than enlarging this terminal noun list.','provenance':{'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_check':'direct normalized prefix comparison'}}
OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload['stats'],sort_keys=True))
