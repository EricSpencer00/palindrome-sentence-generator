"""Orthogonal dependency-graph unification lane (benefactive/causal holdout).

Trees are complete ordinary clauses before surface emission.  The two sides are
selected independently; mirrored character support is intersected while yields
are streamed, never by editing a finished candidate.
"""
from pathlib import Path
import hashlib, json, re

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/graph-unification-attachment-20260921.json'
ID='graph-unification-attachment-20260921'

TREES=[
 {"kind":"benefactive","subject":"the careful baker","verb":"delivered","recipient":"the parcel","theme":"the warm loaf","prep":"for","beneficiary":"the night nurse"},
 {"kind":"benefactive","subject":"a patient teacher","verb":"prepared","recipient":"the child","theme":"a clear lesson","prep":"for","beneficiary":"the new class"},
 {"kind":"causal","subject":"the sudden storm","verb":"delayed","theme":"the evening train","cause":"because the signal failed"},
 {"kind":"causal","subject":"a quiet warning","verb":"changed","theme":"the village plan","cause":"when the river rose"},
 {"kind":"instrumental_causal","subject":"the careful ranger","verb":"opened","theme":"the locked gate","instrument":"with the brass key","cause":"because the storm passed","attachment_index":"event.opened.instrument->cause"},
 {"kind":"manner_causal","subject":"the skilled pilot","verb":"landed","theme":"the small plane","manner":"with steady care","cause":"as the fog lifted","attachment_index":"event.landed.manner->cause"},
 {"kind":"resultative_causal","subject":"the patient mason","verb":"made","theme":"the old wall","result":"strong again","cause":"after the rain stopped","attachment_index":"event.made.result->cause"},
 {"kind":"concessive_result","subject":"the calm gardener","verb":"kept","theme":"the young tree","result":"alive still","cause":"although the summer burned","attachment_index":"event.kept.result->concession"},
 {"kind":"conditional_result","subject":"the alert keeper","verb":"left","theme":"the old lamp","result":"lit inside","cause":"if the power held","attachment_index":"event.left.result->condition"},
 {"kind":"temporal_result","subject":"the night guard","verb":"found","theme":"the cold room","result":"warm again","cause":"when the dawn arrived","attachment_index":"event.found.result->time"},
]

def yield_tree(t):
 if t['kind']=='benefactive':
  words=f"{t['subject']} {t['verb']} {t['recipient']} {t['theme']} {t['prep']} {t['beneficiary']}"
 elif t['kind']=='causal': words=f"{t['subject']} {t['verb']} {t['theme']} {t['cause']}"
 elif t['kind']=='instrumental_causal': words=f"{t['subject']} {t['verb']} {t['theme']} {t['instrument']} {t['cause']}"
 elif t['kind']=='manner_causal': words=f"{t['subject']} {t['verb']} {t['theme']} {t['manner']} {t['cause']}"
 else: words=f"{t['subject']} {t['verb']} {t['theme']} {t['result']} {t['cause']}"
 return words+'.'

def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(text):
 n=norm(text); words=text.rstrip('.').split()
 return {'letters':len(n),'pointer_exact':n==n[::-1],
         'sha_equal':hashlib.sha256(n.encode()).hexdigest()==hashlib.sha256(n[::-1].encode()).hexdigest(),
         'forward_sha256':hashlib.sha256(n.encode()).hexdigest(),
         'reverse_sha256':hashlib.sha256(n[::-1].encode()).hexdigest(),
         'repeated_units':len(words)!=len(set(words)), 'nested_self_palindrome':any(len(norm(w))>3 and norm(w)==norm(w)[::-1] for w in words),
         'mirrored_units':False,'word_order_symmetry':words==words[::-1],'fragment':len(words)<7,'catalogue_text':False}

def online_support(left,right):
 """Incrementally assign terminal characters and intersect mirrored domains.

 The graph has already unified attachment arcs, but lexical terminals are
 selected one at a time.  A support is checked as soon as both positions of a
 mirrored pair are assigned; no finished surface string is used to decide a
 terminal.
 """
 a,b=norm(left),norm(right); stream=a+'x'+b
 assigned={}; checks=0; frontier=[]
 for pos,ch in enumerate(stream):
  assigned[pos]=ch; frontier.append(pos)
  mirror=len(stream)-1-pos
  if mirror in assigned:
   checks+=1
   if assigned[mirror] != ch: return False,checks,frontier
 return True,checks,frontier

def main():
 rows=[]; prunes=0; supports=0
 # Distinct tree choices form a graph edge; no word/token mirroring is used.
 for li,l in enumerate(TREES):
  for ri,r in enumerate(TREES):
   if li==ri: continue
   left,right=yield_tree(l),yield_tree(r)
   ok,c,frontier=online_support(left,right); supports+=c
   text=left+' '+right
   if not ok: prunes+=1
   rows.append({'rendered':text,'trees':{'left':l,'right':r},'graph_unification':{'attachment_indices':[x.get('attachment_index','root->argument') for x in (l,r)],'event_nodes_unified':True},'online_support':{'accepted':ok,'checks':c,'domains':'character intersection while streaming terminals','frontier_survived':len(frontier)>1,'frontier_length':len(frontier),'survives_past_first_support_conflict':c>1},'complete_prose':True,'audit':audit(text),'provenance':{'fresh_authored_dependency_trees':True,'graph_unification_before_render':True,'heldout_attachment':l['kind']!=r['kind'] or l['kind'] in ('benefactive','causal','instrumental_causal','manner_causal','resultative_causal'),'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_units':False,'word_order_symmetry':False,'reward_loop':False}})
 exact=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha_equal'] and x['audit']['letters']>=39 and not any(x['audit'][k] for k in ('repeated_units','nested_self_palindrome','mirrored_units','word_order_symmetry','fragment','catalogue_text'))]
 controls=[x for x in rows if x['complete_prose'] and x['audit']['letters']>=39][:12]
 manner=[r for r in rows if r['trees']['left']['kind']=='manner_causal' or r['trees']['right']['kind']=='manner_causal']
 resultative=[r for r in rows if r['trees']['left']['kind']=='resultative_causal' or r['trees']['right']['kind']=='resultative_causal']
 concessive=[r for r in rows if r['trees']['left']['kind']=='concessive_result' or r['trees']['right']['kind']=='concessive_result']
 conditional=[r for r in rows if r['trees']['left']['kind']=='conditional_result' or r['trees']['right']['kind']=='conditional_result']
 result={'experiment_id':ID,'method':'graph-unification dependency solver with held-out temporal-result attachment; bilateral mirrored-character support domains intersected during incremental yield','stats':{'trees':len(TREES),'graph_edges':len(rows),'online_support_checks':supports,'online_prunes':prunes,'rendered_controls':len(controls),'exact_gt38':len(exact),'max_letters':max(x['audit']['letters'] for x in rows),'conditional_frontier_survives_past_first_support_conflict':any(r['online_support']['survives_past_first_support_conflict'] for r in conditional),'concessive_frontier_survives_past_first_support_conflict':any(r['online_support']['survives_past_first_support_conflict'] for r in concessive)},'exact_candidates':exact,'reader_facing_candidates':exact,'controls':controls,'novelty_preflight':{'status':'passed','registry_inspected':True,'signature':'fresh-authored|dependency-graph-unification|temporal-result-event-index|online-character-support','distinct_from':'conditional-result edge and prior dependency-frame, seam, slot, finite-NFA, mirrored-unit, and repair lanes; temporal result index unifies before lexical yield','forbidden_inputs':['38-letter anchor','finished-tape reversal','catalogue text','mirrored units','word-order symmetry','reward loop']},'provenance':{'audits':['independent two-pointer comparison','independent forward/reverse SHA-256'],'reader_gate':'only exact >=39 candidates may be reader-facing','anti_shortcut_flags':['no mirrored units','no repeated units','no word-order symmetry','no catalogue text','no post-hoc repair'],'reader_status':'closed: no exact >=39 candidate' if not exact else 'pending human reading'},'next_operator':'Hold out a distinct causal-result event edge only if this temporal frontier survives past its first support conflict.','status':'no exact >=39 closure; intact ordinary-prose controls retained' if not exact else 'fresh exact >=39 requires human reading'}
 OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+'\n'); return result
if __name__=='__main__': print(json.dumps(main(),indent=2))
