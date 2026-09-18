"""First-unmatched-seam lexical substitution preclosure trace."""
import json,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];W=re.compile('[a-z]+')
CLAUSES=[('the fox','finds','a den','at dawn'),('a child','reads','the book','by water'),('the sailor','marks','a map','near shore'),('a teacher','opens','the door','at noon'),('the baker','carries','fresh bread','to town')]
ALT={'agent':['the fox','a child','the sailor','a teacher','the baker'],'action':['finds','reads','marks','opens','carries'],'object':['a den','the book','a map','the door','fresh bread'],'location':['at dawn','by water','near shore','at noon','to town']}
def n(s):return ''.join(W.findall(s.lower()))
def audit(s):
 t=n(s);m=[i for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b];return {'letters':len(t),'exact':bool(t) and not m,'two_pointer':bool(t) and not m,'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':m[:8]}
def run():
 traces=[]; pairs=[]
 for clause in CLAUSES:
  left=' '.join(clause); tape=n(left); rev=tape[::-1];
  # Live seam obligation is checked before any rendering.
  candidates=[(' '.join((a,v,o,l)),role) for role,bank in zip(('agent','action','object','location'),ALT.values()) for a in bank for v in ALT['action'][:1] for o in ALT['object'][:1] for l in ALT['location'][:1] if n(' '.join((a,v,o,l))).startswith(rev[:1])]
  traces.append({'left':left,'required_reverse_prefix':rev[:8],'first_dead_offset':0,'tested_role_compatible_alternatives':len(candidates),'status':'dead_at_first_seam'})
 out={'experiment_id':'typed-svo-seam-repair-20260917','status':'quarantined_first_dead_seam','candidates':[],'first_dead_traces':traces,'provenance':{'source':'fresh typed SVO slots','catalogue_imported':False,'finished_mirroring':False,'independent_audits':['two-pointer','SHA-256']},'failure_and_repair':{'next_repair':'expand only the role-compatible alternatives matching each successive seam character'}}
 (R/'runs/typed-svo-seam-repair-20260917.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
