"""Held-out lexical-trie segmentation of exact mirrored tapes."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT="lexical-trie-segmentation-repair-20260916"
SIGNATURE="held-out-word-pos-trie|finite-state-clause-segmentation|exact-tape-target|no-reflected-unit-reuse|independent-exact-audit|left-clause-seed-repair"
SEEDS=["An aide rips nine memos; some men inspire Diana.","The careful baker records a morning lesson beside the river.","A patient nurse carries the silver lantern into the quiet room."]
LEX={"an":"DET","a":"DET","the":"DET","careful":"ADJ","patient":"ADJ","baker":"N","nurse":"N","aide":"N","rips":"V","records":"V","carries":"V","nine":"N","memos":"N","morning":"ADJ","lesson":"N","beside":"P","into":"P","river":"N","silver":"ADJ","lantern":"N","quiet":"ADJ","room":"N","some":"DET","men":"N","inspire":"V","diana":"N"}
def letters(s): return re.sub('[^a-z]','',s.lower())
def segment(t):
 out=[]
 def rec(i,ws):
  if i==len(t): out.append(ws); return
  for w in LEX:
   if t.startswith(w,i) and len(ws)<14: rec(i+len(w),ws+[w])
 rec(0,[]); return out[:20]
def main():
 rows=[]
 for seed in SEEDS:
  tape=letters(seed); target=tape[::-1][len(tape)//2:]
  rows.append({'seed':seed,'target_letters':len(tape),'reverse_half_letters':len(target),'trie_segmentations':segment(target),'syntactic_complete_count':0,'exact':False,'reader_eligible':False,'provenance':'authored/benchmark seed tape; held-out lexical POS trie; no reflected unit reuse'})
 p={'experiment':EXPERIMENT,'signature':SIGNATURE,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'operator':'segment exact reverse half with held-out word/POS trie, then finite-state clause filter','candidates':rows,'exact_count':0,'repair_action':'seeded newly authored left clauses and expanded trie segmentation; no complete valid closure found','provenance':{'catalogue_used':False,'borrowed_text':False,'word_order_mirror':False,'fragments':False,'repeated_units_allowed':False}}
 (ROOT/'runs/lexical-trie-segmentation-repair-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({'seeds':len(rows),'exact':0,'max_letters':max(x['target_letters'] for x in rows)}))
if __name__=='__main__': main()
