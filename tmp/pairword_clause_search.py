import itertools, re

def tape(s): return ''.join(c for c in s if c.isalpha())

WORDS = {
 'DET':['a','an','the','no','one','some'],
 'SUBJ':['man','men','woman','sailor','baker','teacher','writer','artist','farmer','poet','pilot','nurse','aide','captain','gardener','dog','cat','rat','star','smart','regal','denim','gateman','drawer','stressed','deliver','diaper','parts','straw','flow','live','time','saw','step','stop','spot','ten','stun','snap','spit','rail','loop','room','raw','war','rat','tar','tab','bat','ton','won','now'],
 'VERB':['was','saw','is','are','will','can','did','let','stop','live','draw','deliver','repaid','emit','flow','parts','spit','snap','turns','reads','writes','helps','guides','keeps','marks','rips','inspires','sees','makes','takes','gives','finds','names'],
 'ADJ':['smart','stressed','evil','regal','raw','live','old','new','red','calm','kind','brave','quiet','wise'],
 'NOUN':['drawer','desserts','reward','reviled','diaper','repaid','gateman','nametag','parts','strap','straw','warts','regal','lager','denim','mined','flow','wolf','live','evil','time','emit','saw','was','step','pets','stop','pots','spot','tops','ten','net','star','rats','stun','nuts','snap','pans','spit','tips','rail','liar','loop','pool','room','moor','raw','war','rat','tar','tab','bat','ton','not','won','now','man','men','aide','memos','diana','plan','canal','map','road','boat','book','letter','name','title','answer','garden','stone','song','meal'],
 'ADP':['in','on','at','by','near','over','under','after','before','with'],
}
for k in WORDS: WORDS[k]=list(dict.fromkeys(WORDS[k]))
LT=[['DET','SUBJ','VERB','DET','ADJ','NOUN'],['DET','ADJ','SUBJ','VERB','DET','NOUN'],['SUBJ','VERB','DET','NOUN'],['DET','SUBJ','VERB','DET','NOUN','ADP','DET','NOUN']]
RT=[['DET','SUBJ','VERB','DET','ADJ','NOUN'],['DET','SUBJ','VERB','DET','NOUN'],['SUBJ','VERB','DET','NOUN'],['DET','NOUN','VERB','ADP','DET','NOUN']]

def parse(target,T):
 out=[]
 idx=INDEX
 def go(i,pos,words):
  if i==len(T):
   if pos==len(target):out.append(words)
   return
  for w,tw in idx[T[i]].get(target[pos:pos+1],[]):
   if target.startswith(tw,pos):
    if T[i] in {'SUBJ','VERB','ADJ','NOUN'} and w in words:continue
    go(i+1,pos+len(tw),words+[w])
 go(0,0,[]);return out

INDEX={k:{} for k in WORDS}
for k,vals in WORDS.items():
 for w in vals: INDEX[k].setdefault(tape(w)[:1],[]).append((w,tape(w)))

if __name__=='__main__':
 # Start with the lexical edges that have an attested reverse reading; this
 # avoids spending the run on a Cartesian product of unrelated words.
 WORDS['SUBJ'] = [w for w in WORDS['SUBJ'] if tape(w)[::-1] in set(WORDS['NOUN'] + WORDS['VERB']) or w in {'man','men','aide','baker','sailor','teacher'}]
 WORDS['VERB'] = [w for w in WORDS['VERB'] if tape(w)[::-1] in set(WORDS['NOUN'] + WORDS['VERB']) or w in {'was','saw','is','are','live','draw','let','rips','inspires'}]
 WORDS['NOUN'] = [w for w in WORDS['NOUN'] if len(w) <= 7]
 LT = [['SUBJ','VERB','DET','NOUN'], ['DET','SUBJ','VERB','DET','NOUN']]
 INDEX = {k:{} for k in WORDS}
 for k,vals in WORDS.items():
  for w in vals: INDEX[k].setdefault(tape(w)[:1],[]).append((w,tape(w)))
 hits=[]
 for lt in LT:
  for vals in itertools.product(*(WORDS[k] for k in lt)):
   if any(lt[i] in {'SUBJ','VERB','ADJ','NOUN'} and vals[i] in vals[:i] for i in range(len(vals))):continue
   left=' '.join(vals); target=tape(left)[::-1]
   for rt in RT:
    for right in parse(target,rt):
     text=left+'; '+' '.join(right)+'.'; hits.append((len(tape(text)),text,lt,rt))
 print('hits',len(hits))
 for row in sorted(hits,reverse=True)[:100]:print(row)
