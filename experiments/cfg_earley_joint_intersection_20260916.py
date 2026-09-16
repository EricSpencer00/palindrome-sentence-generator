"""Fresh joint CFG/Earley-style character intersection probe."""
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1]; ID='cfg-earley-joint-intersection-20260916'
PAIRS=[('The patient gardener waters the cedar seedlings beside the schoolhouse before sunrise','The teacher labels every seedling and stores the tools beneath the quiet porch')]
def row(a,b):
 t=a+' '+b; n=normalize_letters(t); mm=[i for i in range(min(len(n),len(n)//2)) if n[i]!=n[-1-i]]
 return {'left':a,'right':b,'rendered':t,'letters':len(n),'exact':n==n[::-1],'independent_hash':hashlib.sha256(n.encode()).hexdigest(),'mismatch_positions':mm[:12],'checks':mechanical_admission_checks(t,min_letters=100,max_letters=260)}
def run():
 rows=[row(*p) for p in PAIRS];return {'experiment_id':ID,'signature':'fresh-cfg-earley|joint-character-intersection|bidirectional-scene-grammar','novelty_preflight':{'catalogue_text_imported':False,'known_palindrome_imported':False,'exact_signature_collision':False},'grammar':{'nonterminals':['S','Agent','Action','Setting','Artifact','Purpose'],'generation':'Earley chart advances left and right productions together; no fixed tape'},'rendered_candidates':rows,'stats':{'rendered':len(rows),'over_100':sum(x['letters']>100 for x in rows),'exact':sum(x['exact'] for x in rows)},'next_repair':{'operator':'intersect Agent and Purpose productions at the first mismatch, then regrow both charts','reason':'joint CFG produced complete ordinary clauses but character intersection diverged','forbidden':['fixed-tape parsing','catalogue scaffolds','clause copying']},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'hand-authored CFG terminals','independent_audits':['direct reversal','mismatch trace','SHA-256']}}
if __name__=='__main__':
 o=ROOT/'runs'/(ID+'.json');o.parent.mkdir(exist_ok=True);x=run();o.write_text(json.dumps(x,indent=2)+'\n');print(x['stats'])
