"""Exhaustive POS-lattice resegmentation of the 44-letter diagnostic tape.
This is a bounded, authored POS lexicon over every dictionary segmentation of the
fixed tape; it is an audit, never a readability claim.
"""
from pathlib import Path
import hashlib,json,re
TAPE='wasnoelaneraagasanitemmetinasagaarenaleonsaw'
LEX=set(x.strip().lower() for x in Path('data/lexicon.txt').read_text().splitlines() if x.strip())
# POS inventory intentionally small and transparent: only words occurring in tape.
TAGS={
 'a':{'DET'},'an':{'DET'},'was':{'V','AUX'},'is':{'V','AUX'},'met':{'V','N'},
 'in':{'P'},'at':{'P'},'on':{'P'},'no':{'DET','ADV'},'era':{'N'},'gas':{'N','V'},
 'saga':{'N'},'item':{'N'},'arena':{'N'},'leon':{'N','PROPN'},'saw':{'V'},
 'one':{'DET','NUM'},'as':{'P','ADV'},'we':{'PRO'},'he':{'PRO'},'she':{'PRO'},
 'name':{'N','V'},'man':{'N','V'},'men':{'N'},'mean':{'V','A'},'near':{'P','A'},
 'sea':{'N'},'sean':{'PROPN'},'noel':{'PROPN'},'alan':{'PROPN'},'anna':{'PROPN'},
}
# all contiguous dictionary words, including one-letter entries
edges={i:[] for i in range(len(TAPE))}
for i in range(len(TAPE)):
 for j in range(i+1,len(TAPE)+1):
  w=TAPE[i:j]
  if w in TAGS and w in LEX: edges[i].append((j,w,sorted(TAGS[w])))
# enumerate complete lattice paths, with a hard diagnostic cap only as protection
paths=[]
def walk(i,ws):
 if len(paths)>=200000:return
 if i==len(TAPE): paths.append(ws); return
 for j,w,t in edges.get(i,[]): walk(j,ws+[(w,t)])
walk(0,[])
# permissive clause grammar patterns (word classes): determiner NP, pronoun NP, proper NP;
# finite verb; optional PP/NP complement. This intentionally reports candidates for audit.
def coherent(ws):
 words=[x[0] for x in ws]; tags=[set(x[1]) for x in ws]
 # Accept one or two simple clauses separated only by punctuation in rendering (none here).
 for k in range(2,len(ws)-1):
  if (('DET' in tags[0] or 'PRO' in tags[0] or 'PROPN' in tags[0]) and ('V' in tags[k] or 'AUX' in tags[k])) or ('AUX' in tags[0] and ('PROPN' in tags[1] or 'PRO' in tags[1])):
   if any('N' in t or 'PROPN' in t for t in tags[1:k]): return True
 # compact subject-verb-object
 return len(ws)>=3 and ('PROPN' in tags[0] or 'PRO' in tags[0] or 'DET' in tags[0]) and ('V' in tags[1] or 'AUX' in tags[1]) and any('N' in t for t in tags[2:])
def hidden_spans(words):
 s=''.join(words); out=[]
 # proper contiguous word spans that are palindromes and >= 3 letters
 for i in range(len(s)):
  for j in range(i+3,len(s)+1):
   if j-i < len(s) and s[i:j]==s[i:j][::-1]: out.append(s[i:j])
 return sorted(set(out),key=lambda x:(-len(x),x))
def audit(ws):
 words=[x[0] for x in ws]; s=''.join(words)
 return {'rendered':' '.join(words),'words':words,'letters':len(s),'exact':s==TAPE,'reverse_tokens':words==words[::-1], 'hidden_palindromic_spans':hidden_spans(words), 'coherent_pos':coherent(ws)}
rows=[audit(p) for p in paths]
coh=[r for r in rows if r['coherent_pos']]
result={'experiment':'exhaustive-pos-lattice-44-tape-20260918','tape':TAPE,'letters':len(TAPE),'method':{'lexicon':'data/lexicon.txt intersected with tape; authored POS tags in source','enumeration':'all complete dictionary segmentations (no LM beam)','grammar':'permissive finite clause POS recognizer; candidate flag is not human readability'},'stats':{'dictionary_edges':sum(map(len,edges.values())),'complete_segmentations':len(rows),'pos_coherent_candidates':len(coh),'strict_admissions':0},'actual_segmentations':rows,'pos_coherent_candidates':coh,'strict_gate':{'status':'closed','rule':'exact tape + complete coherent English + no proper contiguous palindromic subspan + no reverse-token shortcut + provenance','admitted':0,'reason':'all POS-lattice candidates either fragmentary/overpermissive or contain hidden palindromic spans; no human reader test run'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexicon_sha256':hashlib.sha256(Path('data/lexicon.txt').read_bytes()).hexdigest(),'material':'fixed diagnostic tape only; no catalogue import; no finished-tape reversal'},'next_constructive_repair':'Author a new complete finite clause around a changed outer seam; do not continue resegmenting this immutable tape.'}
Path('runs/exhaustive-pos-lattice-44-tape-20260918.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result['stats'][k] for k in result['stats']}))
for r in coh[:30]: print(r['rendered'],'| hidden=',r['hidden_palindromic_spans'])
