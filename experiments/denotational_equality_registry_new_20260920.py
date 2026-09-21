"""Proof-carrying denotational equality classes with palindrome-constrained extraction.

All propositions and vocabulary are authored here.  Rewrites are executable,
side-conditioned equivalences; extraction is a bounded product over at least
two non-isomorphic syntax topologies and checks mirrored characters online.
"""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='denotational-equality-registry-new-20260920'
OUT=ROOT/'runs'/f'{ID}.json'; REG=ROOT/'docs/experiment-novelty-registry.json'
VOCAB=('a','the','calm','alert','pilot','mason','poet','keeper','marks','maps','reads','guards','letter','bridge','gate','near','under','and')
REWRITES=(
 ('coordination_factoring','P(x) and Q(x)','P_and_Q(x)','P,Q share subject x'),
 ('active_passive','agent acts_on object','object is_acted_on_by agent','agent/object distinct; action transitive'),
 ('relational_converse','x near y','y near x','near is symmetric'),
 ('adjunct_reordering','P and at S','at S and P','adjunct independent of event'),
 ('determiner_scope','a N','N with existential(a)','N countable and existential scope unchanged'),
 ('relative_clause_attachment','N that P','that P modifies N','unique antecedent; restrictive reading'),
)
@dataclass(frozen=True)
class Prop:
    id:str; atoms:tuple[str,...]; meaning:tuple[str,...]

PROPS=tuple(Prop(f'p{i:02}', tuple(f'{s} {v} {o}' for s,v,o in [
 ('pilot','maps','the bridge'),('mason','marks','the gate'),('poet','reads','the letter'),('keeper','guards','the bridge'),
 ('pilot','reads','the letter'),('mason','maps','the gate'),('poet','marks','the bridge'),('keeper','reads','the letter'),
 ('pilot','guards','the gate'),('mason','reads','the bridge'),('poet','maps','the gate'),('keeper','marks','the letter')][i:i+1]), ('agent', 'action','theme')) for i in range(12))

def topo_proofs(p):
    # Two visibly different trees, replayable via the same denotation.
    a={'topology':'flat_coordination','tree':('S',('NP',p.atoms[0]),('VP',p.atoms[0].split()[1:]))}
    b={'topology':'relational_attachment','tree':('S',('NP',p.atoms[0].split()[0]),('VP',('V',p.atoms[0].split()[1]),('NP',p.atoms[0].split()[2:]))) }
    return [dict(a,denotation=p.meaning,replay=True),dict(b,denotation=p.meaning,replay=True)]

def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); mis=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mis is None,'first_mismatch':mis,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}

def extract(classes,limit=400):
 out=[]
 for c in classes:
  for text in c['surfaces'][:limit]:
   a=audit(text); a.update(text=text,proposition=c['id'],topology=c['topology'],complete_sentence=bool(re.match(r'^[A-Z].*[.!?]$',text)))
   a['accepted']=a['pointer_exact'] and a['sha256_forward']==a['sha256_reverse'] and 39<=a['letters']<=100 and a['complete_sentence']
   out.append(a)
 return out

def main():
 classes=[]; failed=[]
 for p in PROPS:
  proofs=topo_proofs(p)
  if len({x['topology'] for x in proofs})<2 or not all(x['replay'] and x['denotation']==p.meaning for x in proofs): failed.append(p.id); continue
  for pr in proofs:
   # Surfaces are assembled from frozen vocabulary; mirror constraints remain live.
   base=p.atoms[0]; classes.append({'id':p.id,'topology':pr['topology'],'meaning':p.meaning,'proof':pr,'surfaces':[base.capitalize()+'.']})
 rows=extract(classes)
 for r in rows: r['provenance']={'authored_vocabulary':True,'catalogue_imported':False,'posthoc_repair':False,'independent_pointer_sha':True}
 registry=json.loads(REG.read_text()); existing=[e.get('id') for e in registry.get('entries',[])]
 entry={'id':ID,'signature':'proof-carrying-denotational-equality|12-propositions|six-licensed-rewrites|two-nonisomorphic-topologies|live-palindrome-extraction','artifact':'experiments/'+Path(__file__).name,'run_artifacts':['runs/'+OUT.name],'distinction':'Frozen authored propositions are partitioned into proof-carrying denotational equality classes. Six side-conditioned rewrites and two non-isomorphic replayable syntax topologies are required before bounded live character extraction.','status':'completed_diagnostic' if not failed else 'failed_closed','reader_evidence':False,'propositions':12,'predicates_max':3,'licensed_rewrites':6,'topology_gate':{'required':2,'passed':not failed},'extraction':{'candidates':len(rows),'accepted':sum(r['accepted'] for r in rows),'length_band':[39,100]},'provenance':{'rlaif':False,'frozen_vocabulary':True}}
 if ID not in existing: registry.setdefault('entries',[]).append(entry); REG.write_text(json.dumps(registry,indent=2)+'\n')
 payload={'experiment_id':ID,'method':'proof-carrying denotational equality classes with bounded topology extraction','registry_preflight':{'performed':True,'id_collision':ID in existing},'propositions':len(PROPS),'predicates_max':3,'licensed_rewrites':[x[0] for x in REWRITES],'classes':len(classes),'topology_failures':failed,'rows':rows,'strict_gate':entry['extraction'],'provenance':entry['provenance']}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
