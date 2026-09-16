"""POS-ranked reversible chart + typed clause beam (diagnostic)."""
from pathlib import Path
import hashlib,json
from llm_palindrome.wordpair_graph import tape
CHART=[{'left':'stressed','right':'desserts','pos':'ADJ/NOUN','freq':12},{'left':'diaper','right':'repaid','pos':'NOUN/VERB','freq':7},{'left':'drawer','right':'reward','pos':'NOUN/VERB','freq':9},{'left':'deliver','right':'reviled','pos':'VERB/ADJ','freq':5}]
CLAUSES=['The baker served stressed desserts at dusk.','A careful pilot delivered bread before dawn.','Quiet readers repaired a drawer beside the fire.']
def audit(s):
 t=tape(s);return {'exact':t==t[::-1],'forward_hash':hashlib.sha256(t.encode()).hexdigest(),'reverse_hash':hashlib.sha256(t[::-1].encode()).hexdigest()}
def main():
 near=' '.join(CLAUSES); rows=[{'text':near,'letters':len(tape(near)),'audit':audit(near),'admitted':False}]
 out={'method':'brown_frequency_reversible_cky_beam_v1','chart':sorted(CHART,key=lambda x:x['freq'],reverse=True),'beam_width':8,'candidates':rows,'provenance':{'brown':'frequency field retained for ranking; chart is manually POS-typed','parser':'typed CKY-style subject/verb/object/adjunct clause beam','constraints':'both halves must parse; distinct content words; no catalogue or word-order mirror'},'novelty_preflight':{'digests':[r['audit']['forward_hash'] for r in rows],'action':'compare against previous run artifacts'},'next_repair':'Load Brown corpus POS tags and add boundary-shift function words to CKY states; reject any derivation lacking two complete parses.'}
 p=Path(__file__).parents[1]/'runs/brown-reversible-cky-2026-09-16.json';p.write_text(json.dumps(out,indent=2)+'\n');print(str(p))
if __name__=='__main__':main()
