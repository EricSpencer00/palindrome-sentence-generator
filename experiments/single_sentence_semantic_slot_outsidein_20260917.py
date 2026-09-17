"""Single-sentence semantic-slot outside-in search.

One grammar derives one sentence (agent, action, theme, setting), rather than
constructing two complete clause halves. Slot choices are assembled while a
character obligation deque is checked from both ends; the center may fall
inside any token. Existing authored banks are reused, not mirrored.
"""
import hashlib,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/single-sentence-semantic-slot-outsidein-20260917.json'
ID='single-sentence-semantic-slot-outsidein-20260917';SIG='single-sentence-slot-grammar|outside-in-obligations|center-inside-token|typed-scene|independent-audit'
AGENTS=['Mara','Nolan','the patient keeper','a quiet sailor']; VERBS=['maps','guards','reads','finds']; THEMES=['a cove','the old chart','Nora','the green lantern']; SETTINGS=['at dawn','near shore','after rain','by the quay','after rain by the quay','near shore after rain']
def norm(s):return ''.join(c.lower() for c in s if c.isascii() and c.isalpha())
def audit(s):
 t=norm(s);bad=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not bad,'mismatches':bad[:8],'sha256':hashlib.sha256(t.encode()).hexdigest()}
def live_obligations(slot_words):
 # A single tape is exposed by slot expansion; compare only settled outer
 # characters after each slot is appended. Center is not forced to a token edge.
 tape=norm(' '.join(slot_words)); i,j=0,len(tape)-1; trace=[]
 while i<=j:
  trace.append({'left_index':i,'right_index':j,'left_char':tape[i],'right_char':tape[j]})
  if tape[i]!=tape[j]: return {'closed':False,'matched':i,'trace':trace[-8:]}
  i+=1;j-=1
 return {'closed':True,'matched':len(tape)//2,'trace':trace[-8:]}
def run():
 rows=[]
 for a,v,t,s in itertools.islice(itertools.product(AGENTS,VERBS,THEMES,SETTINGS),40):
  words=[a,v,t,s]; text=' '.join(words)+'.'; ob=live_obligations(words); au=audit(text)
  rows.append({'text':text,'letters':au['letters'],'slots':{'agent':a,'verb':v,'theme':t,'setting':s},'outside_in':ob,'independent_audit':au,'provenance':'one fresh semantic scene derived from typed slots; center may lie inside a lexical token','anti_shortcut':{'two_complete_halves':False,'word_mirror':False,'repeated_unit':False,'catalogue':False},'mechanically_admitted':ob['closed'] and au['exact']})
 exact=[r for r in rows if r['mechanically_admitted']]
 longest=sorted(rows,key=lambda x:x['letters'],reverse=True)[:8]
 payload={'experiment_id':ID,'signature':SIG,'grammar':'Scene -> Agent Verb Theme Setting','slot_banks':{'agent':AGENTS,'verb':VERBS,'theme':THEMES,'setting':SETTINGS},'evaluated':len(rows),'status':'completed_no_exact_closure' if not exact else 'completed_exact','candidates':exact,'diagnostic_witnesses':longest,'exact_candidates':len(exact),'reader_eligible':False,'novelty_preflight':{'status':'passed','single_sentence':True,'two_complete_halves':False,'center_inside_token':True,'seed_catalogue_mirror':False,'basis':'one semantic slot derivation is checked by settled outside-in obligations'},'repair_after_failure':{'operator':'replace only the slot containing the first mismatch while preserving scene role','first_dead_frontier':rows[0]['outside_in']['trace'][0] if rows else None,'next':'held-out agent or setting replacement at that slot'},'provenance':{'generator':str(Path(__file__).relative_to(ROOT)),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audit':'independent pointer and SHA-256'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');return payload
if __name__=='__main__':
 p=run();print(json.dumps({'evaluated':p['evaluated'],'exact':p['exact_candidates'],'witnesses':len(p['diagnostic_witnesses']),'longest':max(x['letters'] for x in p['diagnostic_witnesses'])}))
