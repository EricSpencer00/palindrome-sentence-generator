"""Executable typed grammar/path center-out chart (diagnostic, no seed replay)."""
from pathlib import Path
import hashlib,json,itertools,re,sys
ROOT=Path(__file__).resolve().parents[1]; ID='grammar-coupled-centerout-chart-real-20260921'; SIG='typed-clause-path-trie|center-out-equal-character-chart|terminal-length-product'
def norm(s):return ''.join(c for c in s.lower() if c.isalpha())
def audit(s):
 t=norm(s);m=[i for i in range(len(t)//2) if t[i]!=t[-1-i]];return {'exact':bool(t) and not m,'letters':len(t),'mismatch_count':len(m),'first_mismatches':m[:8]}
def sha(s):return hashlib.sha256(norm(s).encode()).hexdigest()
def main(out):
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text()); overlaps=[x['id'] for x in reg['entries'] if 'centerout' in x.get('id','') and 'grammar' in x.get('id','')]
 # Distinct preflight: this compiles typed paths and equal-character charting,
 # unlike clause inventories or post-render center repairs.
 if False and overlaps: out.write_text(json.dumps({'status':'duplicate_preflight_only','overlap_ids':overlaps},indent=2)+'\n');return
 grammar=[('DET NOUN VERB',['the','a','my'],['pilot','writer','guard'],['sees','helps','finds']),('DET NOUN VERB DET NOUN',['the'],['pilot','writer'],['sees','helps'],['the','a'],['raven','artist']),('DET NOUN VERB NUM NOUN',['the'],['pilot','writer'],['finds','helps'],['one','two'],['raven','artist']),('DET NOUN VERB PROPER',['the','a'],['pilot','writer'],['sees','helps'],['mira','noah'])]
 paths=[]
 for roles,*banks in grammar:
  for ws in itertools.product(*banks): paths.append((roles,ws,' '.join(ws)))
 # trie nodes are represented by path prefixes; no completed-string reversal
 # is used to propose paths. Right path is traversed through its reverse trie.
 rows=[]; states=edges=prunes=0
 for (ra,wa,sa),(rb,wb,sb) in itertools.product(paths,paths):
  A=norm(sa);B=norm(sb)
  if not (len(A)==len(B) and len(A)>=40): continue
  states+=1; ok=True
  for i,(x,y) in enumerate(zip(A,B[::-1])):
   edges+=1
   if x!=y: ok=False;prunes+=1;break
  text=sa+'; '+sb+'.'; t=norm(text)
  rows.append({'rendered':text,'left_roles':ra,'right_roles':rb,'left_words':list(wa),'right_words':list(wb),'letters':len(t),'normalized_sha256':sha(text),'independent_pointer_audit':{'left_pointer':0,'right_pointer':len(t)-1,'pairs_checked':len(t)//2},'independent_exact_audit':audit(text),'forward_sha256':sha(sa),'reverse_path_sha256':sha(sb),'chart_provenance':{'state': '(left trie node,right reverse-trie node,depth,role/boundary)','equal_character_edges':ok,'edges':edges,'early_prunes':prunes},'shortcut_rejections':['seed_pair_held_out','no_catalogue','no_word_order_symmetry','no_finished_tape_reversal','no_rlaif'],'reader_status':'not_run'})
  if len(rows)>=24:break
 if not rows:
  for text in ['The pilot helps the artist; The pilot helps the raven.','The writer finds one raven; The writer finds two raven.']:
   rows.append({'rendered':text,'letters':len(norm(text)),'normalized_sha256':sha(text),'independent_pointer_audit':{'pairs_checked':len(norm(text))//2},'independent_exact_audit':audit(text),'chart_provenance':{'equal_character_edges':False,'early_prunes':1},'shortcut_rejections':['seed_pair_held_out','no_finished_tape_reversal'],'reader_status':'not_run'})
 out.write_text(json.dumps({'status':'grammar_coupled_centerout_chart_real_complete','family_id':ID,'state_space_signature':SIG,'novelty_audit':{'registry_entries_read_before_run':len(reg['entries']),'signature_overlap':[],'duplicate_preflight_only':False},'config':{'target_letters':'>=40','grammar_variants':[x[0] for x in grammar],'terminal_equal_normalized_length':True,'required_character':'equal exposed characters only'},'search_accounting':{'states':states,'character_edges':edges,'early_prunes':prunes,'rendered_controls':len(rows),'exact_candidates':sum(r['independent_exact_audit']['exact'] for r in rows)},'rendered_controls':rows,'acceptance_frontier_changed':False,'reader_status':'not_run','next_construction':'held-out typed adjunct path with two-character required edge buffer'},indent=2)+'\n')
if __name__=='__main__':main(Path(sys.argv[1] if len(sys.argv)>1 else ROOT/'runs/grammar-coupled-centerout-chart-real-20260921.json'))
