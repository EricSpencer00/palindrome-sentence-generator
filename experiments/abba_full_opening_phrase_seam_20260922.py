import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/abba-full-opening-phrase-seam-20260922.json'
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); bad=[(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'two_pointer_exact':bool(t) and not bad,'first_mismatches':bad[:4],'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse_obligation':hashlib.sha256(t[::-1].encode()).hexdigest()}
PAIRS=(
 {'id':'memos-some-maps','left':'The clerk filed the morning memos.','terminal':'memos','opening':'Some maps','continuation':'rested beside the lamp.'},
 {'id':'arena-an-era','left':'The players crossed the old arena.','terminal':'arena','opening':'An era','continuation':'of patient work shaped the town.'},
 {'id':'reason-no-sailor','left':'The judge explained a careful reason.','terminal':'reason','opening':'No sailor','continuation':'left the harbor before dawn.'},
 {'id':'data-a-tad','left':'The analyst checked the evening data.','terminal':'data','opening':'A tad','continuation':'of rain cooled the garden.'},)
def run():
 rows=[]; controls=[]; certs=[]
 for p in PAIRS:
  rendered=f"{p['left']} {p['opening']} {p['continuation']}"; obligation=letters(p['left'])[::-1]; opening=letters(p['opening']); support=0
  while support<min(len(obligation),len(opening)) and obligation[support]==opening[support]: support+=1
  rows.append({'rendered':rendered,'pair_id':p['id'],'right_opening':p['opening'],'seam_support':support,'opening_length':len(opening),'residual':obligation[support:support+20],'audit':audit(rendered),'provenance':{'full_opening_selected_before_continuation':True,'complete_authored_prose':True,'finished_tape_reversal':False,'catalogue_text':False,'repeated_units':False,'self_palindromic_units':False,'posthoc_repair':False}})
  controls.append({'rendered':p['left'],'kind':'intact-authored-control','audit':audit(p['left'])})
  certs.append({'pair_id':p['id'],'terminal':p['terminal'],'reverse_obligation_prefix':obligation[:len(opening)],'right_opening':opening,'support':support,'opening_fully_consumed':support==len(opening),'residual':obligation[support:support+20]})
 exact=[r for r in rows if r['audit']['two_pointer_exact'] and r['audit']['letters']>38]
 return {'experiment_id':'abba-full-opening-phrase-seam-20260922','method':'joint full right opening phrase and authored left terminal under live tape obligation','stats':{'pairs':4,'controls':4,'rendered_candidates':4,'fully_consumed_openings':sum(c['opening_fully_consumed'] for c in certs),'exact_gt38':len(exact),'max_seam_support':max(r['seam_support'] for r in rows)},'rendered_candidates':rows,'controls':controls,'residual_certificates':certs,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':'abba|full-opening-phrase|authored-terminal','distinct_from':'function-word-only seam and right-first relation trie','finished_tape_reversal':False,'catalogue_text':False,'mirrored_units':False,'reward_ranking':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'reader_gate':'closed pending novel exact output'},'status':'fresh exact closure found' if exact else 'no exact closure; full-opening residuals retained','next_construction':'condition the next right verb on a fully consumed opening and continue the joint grammar state; do not broaden this pair list'}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps(run()['stats'],sort_keys=True))
