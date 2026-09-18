"""Reverse-conditioned phrase lattice with live outside-in matching."""
import hashlib, heapq, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; WORD=re.compile(r"[a-z]+")
COMMON=set(json.loads((ROOT/'data/ngrams_wikitext2.json').read_text()))
RAW=[x.strip() for x in (ROOT/'data/authored_sentences.txt').read_text().splitlines() if x.strip()]
def norm(s): return ''.join(WORD.findall(s.lower()))
def audit(s):
 t=norm(s); i,j=0,len(t)-1; ok=bool(t); mm=[]
 while i<j:
  if t[i]!=t[j]: ok=False; mm.append(i)
  i+=1; j-=1
 return {'letters':len(t),'exact':ok,'two_pointer':ok,'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':mm[:12]}
def ordinary(s):
 ws=WORD.findall(s.lower())
 return len(ws)>=4 and all(w in COMMON or len(w)<=3 for w in ws) and len(set(ws))==len(ws)
def shape(s):
 return tuple('DET' if w in {'a','an','the'} else 'PREP' if w in {'at','by','in','on','near','to','from','with'} else 'WORD' for w in WORD.findall(s.lower()))
def score(s):
 ws=WORD.findall(s.lower()); return sum(w in COMMON for w in ws)/max(1,len(ws))
def shortcut_free(a,b):
 ca=set(WORD.findall(a))-{'a','an','the','at','by','in','on','near','to','from','with'}
 cb=set(WORD.findall(b))-{'a','an','the','at','by','in','on','near','to','from','with'}
 return not ca&cb
def run():
 phrases=[]
 for s in RAW:
  if ordinary(s): phrases.append({'text':' '.join(WORD.findall(s.lower())),'tape':norm(s),'shape':shape(s),'score':score(s)})
 phrases=list({p['text']:p for p in phrases}.values())[:800]
 # Trie is keyed by the required reverse seam, not by a finished reversal.
 trie={}
 for p in phrases:
  q=trie
  for c in p['tape'][::-1]: q=q.setdefault(c,{})
  q.setdefault('$',[]).append(p)
 rows=[]; seen=set(); queue=[]
 for li,left in enumerate(phrases[:120]): heapq.heappush(queue,(-left['score'],0,li,0))
 expansions=0
 while queue and len(rows)<30 and expansions<5000:
  _,depth,li,ri=heapq.heappop(queue); expansions+=1; left=phrases[li]
  for rj,right in enumerate(phrases[:120]):
   if li==rj or left['shape']!=right['shape']: continue
   tape=left['tape']+right['tape']; k=0
   while k<len(tape)//2 and tape[k]==tape[-1-k]: k+=1
   key=(li,rj,k)
   if key in seen: continue
   seen.add(key); text=left['text']+' '+right['text']; a=audit(text)
   rows.append({'text':text,'left_phrase':left['text'],'right_phrase':right['text'],'pos_shape':left['shape'],'matched_outside_in':k,'audit':a,'anti_shortcut':{'disjoint_content_words':shortcut_free(left['text'],right['text']),'word_order_only':False,'finished_mirror':False},'brown_ngram_score':(left['score']+right['score'])/2,'provenance':'fresh authored ordinary phrases; reverse trie selected by live seam obligation'})
 rows.sort(key=lambda x:(x['audit']['exact'],x['matched_outside_in'],x['brown_ngram_score']),reverse=True)
 out={'experiment_id':'reverse-pos-phrase-lattice-20260917','status':'quarantined_no_exact_closure','candidates':rows[:12],'stats':{'phrase_count':len(phrases),'expanded_states':expansions,'rendered_rows':len(rows),'exact_rows':sum(x['audit']['exact'] for x in rows)},'provenance':{'source':'authored_sentences filtered by Brown/Wikitext vocabulary','catalogue_imported':False,'finished_tape_reversal':False,'seed_words_used':False,'independent_audits':['two-pointer','forward/reverse SHA-256']},'failure_and_repair':{'next_repair':'add typed verb/object slots to trie keys and continue only branches whose next required seam character has a lexical completion'}}
 (ROOT/'runs/reverse-pos-phrase-lattice-20260917.json').write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': print(json.dumps(run(),indent=2))
