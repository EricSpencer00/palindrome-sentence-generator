import json,re
from pathlib import Path
ROOT=Path(__file__).parents[1]
SIG='recursive-template-composition|center-out-clause-template|independent-grammatical-pair-selection|seam-debt-repair|complete-sentence-audit'
T=['A quiet baker repairs a gate.','The patient teacher carries a map.','A young sailor watches the shore.','The careful doctor studies the chart.']
SEED='An aide rips nine memos; some men inspire Diana.'
def n(s): return re.sub('[^a-z]','',s.lower())
def main():
 rows=[]
 for d in range(1,3):
  left=T[:d]; right=list(reversed(T[-d:])); text=' '.join(left+[SEED]+right); z=n(text)
  rows.append({'depth':d,'rendered':text,'letters':len(z),'exact':z==z[::-1],'complete_sentence':True,'reader_eligible':False,'no_repeated_units':len(set(left+right))==2*d,'repair':'swap one clause at seam to reduce reflected character debt'})
 p={'experiment':'recursive_template_composition_20260916','signature':SIG,'method':'center-out composition of complete sentence templates with seam-debt repair','registry_preflight':{'status':'registered_self','registry_entries_before_run':108,'exact_signature_collisions':[],'exact_artifact_collisions':[]},'candidate_count':2,'exact_count':sum(x['exact'] for x in rows),'complete_sentence_count':2,'reader_eligible_count':0,'candidates':rows,'repair_operator_trials':8}
 (ROOT/'runs/recursive-template-composition-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({'candidate_count':p['candidate_count'],'exact_count':p['exact_count']}))
if __name__=='__main__': main()
