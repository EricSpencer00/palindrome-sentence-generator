"""Bounded genuine forward Brown phrase search, paper-aligned seam engine."""
import json,re,time,hashlib,gzip
from pathlib import Path
from collections import Counter
from dataclasses import asdict
from llm_palindrome.bilateral_seams import Chunk,CenterOutGrammar,BilateralGrammar,phrase_payoff_utility
from llm_palindrome.admission import normalize_letters as norm
OUT=Path('research/block-seams/multiword-seams-017')
def tag(t):
 t=t.upper().split('-')[0].split('+')[0]
 if t in ('AT','DT','DTS','DTI','DTP','AP','PP$'):return 'D'
 if t.startswith('JJ'):return 'J'
 if t.startswith('NN') or t.startswith('NP'):return 'N'
 if t.startswith('VB') or t in ('BER','BED','BEDZ','BEZ','BE','HVD','HV','HVZ','DO','DOD','DOZ'):return 'V'
 if t in ('IN','TO'):return 'P'
 if t.startswith('RB'):return 'A'
 return 'X'
def mine():
 counts=Counter();sources={};lines=0;tokens=0;root=Path('/Users/eric/nltk_data/corpora/brown');human={'man','woman','men','women','boy','girl','child','children','mother','father','friend','wife','husband','doctor','nurse','teacher','officer','student','aide','people','workers','brother','sister'}
 for file in sorted(root.iterdir()):
  if not file.is_file() or file.name in ('README','CONTENTS'):continue
  for li,line in enumerate(file.read_text(errors='ignore').splitlines()):
   lines+=1
   parsed=[]
   for item in line.split():
    if '/' not in item:continue
    w,t=item.rsplit('/',1);w=w.lower()
    if re.fullmatch('[a-z]+',w):parsed.append((w,tag(t)))
    else:parsed.append(('', 'X'))
   tokens+=len(parsed)
   for n in range(2,6):
    for i in range(len(parsed)-n+1):
     span=parsed[i:i+n];words=[x[0] for x in span];tags=''.join(x[1] for x in span)
     if not all(words):continue
     kind=None
     if re.fullmatch('D?J*N+',tags):kind='HUMAN_NP' if words[-1] in human else 'NP'
     elif re.fullmatch('VA*P',tags):kind='V_PREP'
     elif re.fullmatch('VD?J*N+',tags):kind='V_OBJECT'
     if kind:
      k=(kind,' '.join(words));counts[k]+=1;sources.setdefault(k,dict(file=file.name,line=li+1,tags=tags))
   if lines>=2000:break
  if lines>=2000:break
 kinds={}
 for kind,cap in [('HUMAN_NP',100),('NP',180),('V_PREP',80),('V_OBJECT',120)]:
  kinds[kind]=sorted((k for k in counts if k[0]==kind),key=lambda k:(-counts[k],k))[:cap]
 return kinds,counts,sources,dict(lines=lines,tokens=tokens,source_root=str(root))
def run():
 OUT.mkdir(parents=True,exist_ok=True);t=time.monotonic();kinds,counts,sources,scan=mine();chunks=[]
 def add(text,a,b,role,source):chunks.append(Chunk('p'+str(len(chunks)).zfill(3),text,a,b,role,source))
 for kind,ks in kinds.items():
  for k in ks:
   text=k[1]+' ';source=json.dumps(dict(kind='forward_Brown_attested_span',**sources[k],count=counts[k]))
   if kind=='HUMAN_NP':add(text,'START','SUBJ','human subject NP',source)
   if kind=='NP':add(text,'OBJECT','END','object NP',source)
   if kind=='V_PREP':add(text,'SUBJ','OBJECT','verb+preposition taking object NP',source)
   if kind=='V_OBJECT':add(text,'SUBJ','END','verb with nominal object',source)
 # Productive postnominal location grammar; multiword prepositional units.
 # Separate location state allows descriptive extension without sentence loops.
 for text in ['next to ','close to ','out of ']:add(text,'END','OBJECT','postnominal location relation','authored ordinary two-word prepositional unit')
 initial_inventory=[asdict(c) for c in chunks];assert len(chunks)<=512
 results=[];deadline=time.monotonic()+6
 for midpoint in ['SUBJ','OBJECT']:
  for direction in ['centerout','outsidein']:
   seconds=max(.001,min(2,deadline-time.monotonic()))
   if time.monotonic()>=deadline:break
   g=CenterOutGrammar(chunks,[],center_state=midpoint) if direction=='centerout' else BilateralGrammar(chunks)
   r=g.search(max_steps=10,max_states=3000,max_outputs=20,seconds=seconds);results.append(dict(midpoint=midpoint,direction=direction,**r))
 # Genuine grammatical multiword seam controls from existing native/paper
 # sources; isolated from mined-run success counts, no arbitrary word slices.
 controls=[]
 for cs,mid in [([Chunk('l','No lemon','START','M','negative food NP','native original bridge004'),Chunk('r',', no melon.','M','END','negative food NP','native original bridge004')],'M'),([Chunk('l','A Santa lived ','START','M','subject with manner-taking predicate','native original scene007'),Chunk('c','as a ','M','N','manner preposition/determiner','native original scene007'),Chunk('r','devil at NASA.','N','END','predicate noun and location','native original scene007')],None)]:
  g=CenterOutGrammar(cs,[],center_state=mid) if mid else CenterOutGrammar(cs,['c']);r=g.search(max_steps=8,max_states=100,max_outputs=5,seconds=1);controls.append(r)
 def save(n,o):(OUT/n).write_text(json.dumps(o,indent=2)+'\n')
 save('plan.json',dict(scan_max_lines=2000,phrase_words=[2,5],inventory_caps={k:len(v) for k,v in kinds.items()},total_mined_run_seconds_cap=6,max_steps=10,max_states_per_arm=3000,max_outputs_per_arm=20,midpoints=['SUBJ','OBJECT'],directions=['centerout','outsidein'],controls_separate=True,no_models=True,no_paidcalls=True,stop='4bounded minedphrasearms,2isolatedrealphrasecontrols;noadaptivefollowup'))
 save('inventory.json',initial_inventory);save('scan.json',scan);save('mined-results.json',results);save('real-phrase-controls.json',controls)
 raw=[x for r in results for x in r['outputs']];unique={r['tape']:r for r in raw};save('mined-unique-outputs.json',list(unique.values()))
 heuristic=dict(old_raw_ratios={'one_letter_paid':1,'eight_letter_phrase_paid':1},new_utilities={'one_letter_paid':phrase_payoff_utility(dict(paid=1,added=1),'r'),'eight_letter_phrase_paid':phrase_payoff_utility(dict(paid=8,added=8),'some men')},rule='paid/(added+4)*min(words,3)/3; graded chunk-size/phrase-coverage reward; exact compatibility unchanged; no microscopic chunk presentation in real phrase run')
 save('payoff-bias.json',heuristic)
 summary=dict(scan=scan,phrase_inventory=len(chunks),all_chunks_two_or_more_words=all(len(re.findall('[a-z]+',c.text.lower()))>=2 for c in chunks),arms=[r['receipt'] for r in results],mined_raw_outputs=len(raw),mined_unique_outputs=len(unique),real_controls=[x['text'] for r in controls for x in r['outputs']],controls_not_counted_as_progress=True,semantic_limit='Brown span+POS seam licensing does not prove agreement,valency or coherent progression; actor slot human NP constrained, object role broad',elapsed_seconds=time.monotonic()-t,quality_ratings=0)
 save('summary.json',summary);save('checkpoint.json',dict(status='bounded_phrase_run_complete',summary=summary,next='Read actual mined candidates if any; do not count source controls as progress or substitute POS for semantic review.'));print(json.dumps(summary,indent=2));print('\n'.join(r['text'] for r in unique.values()))
if __name__=='__main__':run()
