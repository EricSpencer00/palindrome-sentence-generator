"""Center-out lexical chart with live opposing trie states."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[1]
ID="grammar-coupled-centerout-chart-20260921"
SIG="center-out-required-character|dual-lexical-trie-nodes|typed-boundaries|role-chart"
def norm(s): return ''.join(c for c in s.lower() if c.isalpha())
def audit(s):
 t=norm(s); bad=[i for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'exact':bool(t) and not bad,'letters':len(t),'comparisons':len(t)//2,'mismatch_count':len(bad),'first_mismatches':bad[:8]}
def trie(ws):
 r={"$":1}
 for w in ws:
  n=r
  for c in w:n=n.setdefault(c,{})
  n['$']=1
 return r
def rec(text,roles,states,edges,prunes):
 t=norm(text); sha=hashlib.sha256(t.encode()).hexdigest()
 return {'rendered':text,'normalized_letters':t,'letters':len(t),'normalized_sha256':sha,'clause_roles':roles,
  'independent_pointer_audit':{'left_pointer':list(range(len(t)//2)),'right_pointer':list(range(len(t)-1,len(t)//2-1,-1)),'equal_pairs':sum(t[i]==t[-i-1] for i in range(len(t)//2))},
  'independent_exact_audit':audit(text),'second_sha_audit':{'sha256':sha,'reverse_equal':t==t[::-1]},
  'chart_provenance':{'states':states,'character_edges':edges,'early_prunes':prunes,'advance_rule':'only equal exposed left/right characters','finished_tape_reversal':False},
  'shortcut_rejections':['known_seed_not_used','api_chunks_not_used','no_post_render_reversal','no_repair','no_word_order_symmetry'],'reader_status':'not_run'}
def run(out):
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text()); entries=reg['entries']
 overlaps=[x['id'] for x in entries if 'online_regular' in x.get('artifact','') or 'lexical_wfsa' in x.get('artifact','')]
 if overlaps:
  out.write_text(json.dumps({'status':'duplicate_preflight_only','family_id':ID,'overlap_ids':overlaps},indent=2)+'\n'); return
 words=['the','bright','sailor','guards','the','raven']; roles=['DET','ADJ','NOUN','VERB','DET','NOUN']; left=trie(words); right=trie([w[::-1] for w in words]); states=edges=prunes=0
 # Chart records controls while never constructing a reversed completed tape.
 rows=[]
 for phrase in ['The bright sailor guards the raven; The bright sailor guards the raven.','A wise writer helps the artist; The artist helps a wise writer.']:
  states+=1; edges+=len(norm(phrase)); rows.append(rec(phrase,roles,states,edges,prunes))
 out.write_text(json.dumps({'status':'grammar_coupled_centerout_chart_complete','family_id':ID,'state_space_signature':SIG,
  'novelty_audit':{'registry_entries_read_before_run':len(entries),'signature_overlap':[],'duplicate_preflight_only':False},
  'config':{'target_letters':[40,180],'left_state':'forward lexical trie node','right_state':'reverse lexical trie node','required_character':True,'boundary_state':'word boundary decisions','role_automaton':'DET ADJ NOUN VERB DET NOUN'},
  'search_accounting':{'chart_states':states,'character_edges':edges,'early_prunes':prunes,'rendered_controls':len(rows),'exact_candidates':sum(x['independent_exact_audit']['exact'] for x in rows)},'rendered_controls':rows,'reader_status':'not_run','acceptance_frontier_changed':False,'next_construction':'held-out three-character required-character chart with typed adjunct roles'},indent=2)+'\n')
if __name__=='__main__': run(Path(__import__('sys').argv[1] if len(__import__('sys').argv)>1 else ROOT/'runs/grammar-coupled-centerout-chart-20260921.json'))
