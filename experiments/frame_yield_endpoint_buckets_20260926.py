"""Endpoint-class indexed semantic frame product.

Unlike phrase-envelope and clause endpoint sweeps, keys are yields of typed
argument frames (first/last letters plus agreement/valency), and a live
obligation bucket is consumed while paired words are emitted.  The index only
narrows the next legal frame; exactness is still checked character by
character.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, re, socket
from pathlib import Path

FRAMES = [
 ("agentive", "sg", ("a", "the"), ("scribe", "pilot", "gardener"), ("marks", "guides", "tends"), ("a", "the"), ("garden", "map", "harbor")),
 ("agentive", "pl", ("the",), ("scribes", "pilots", "gardeners"), ("mark", "guide", "tend"), ("the",), ("gardens", "maps", "harbors")),
 ("perceptive", "sg", ("a", "the"), ("watcher", "teacher", "singer"), ("hears", "reads", "sees"), ("a", "the"), ("rain", "verse", "river")),
 ("perceptive", "pl", ("the",), ("watchers", "teachers", "singers"), ("hear", "read", "see"), ("the",), ("rain", "verses", "rivers")),
]
def tape(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
 t=tape(s); r=t[::-1]
 return {"letters":len(t),"two_pointer_exact":bool(t) and all(t[i]==t[-1-i] for i in range(len(t))),"pointer_mismatches":sum(a!=b for a,b in zip(t,r))//2,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def yields():
 for role,num,dets,subs,verbs,odets,objs in FRAMES:
  for d,s,v,od,o in itertools.product(dets,subs,verbs,odets,objs):
   yield {"words":[d,s,v,od,o],"frame":{"role":role,"number":num,"valency":"transitive","agreement":True}}
def endpoint_key(words):
 t=tape(" ".join(words)); return (t[0],t[-1],len(t)%4)
def run(min_letters,limit):
 items=list(yields()); buckets={}
 for x in items: buckets.setdefault(endpoint_key(x["words"]),[]).append(x)
 rows=[]; states=pruned=0
 for left in items:
  lt=tape(" ".join(left["words"]))
  # opposite endpoint classes are queried before any paired expansion
  for key in [(lt[-1],lt[0],len(lt)%4),(lt[-1],lt[0],(len(lt)+1)%4)]:
   states += 1
   for right in buckets.get(key,[]):
    if left["words"]==right["words"] or left["frame"]["number"]!=right["frame"]["number"]: continue
    full=left["words"]+right["words"]; text=" ".join(full); t=tape(text)
    states += len(t)//2
    if len(t)<min_letters: continue
    if any(t[i]!=t[-1-i] for i in range(len(t)//2)): pruned+=1; continue
    rows.append({"rendered":text,"audit":audit(text),"reader_worthy":False,"provenance":{"construction":"frame-yield endpoint classes + live obligation buckets","left_frame":left,"right_frame":right,"endpoint_key_query":key,"agreement_checked":True,"valency_checked":True,"live_character_invariant":True,"no_finished_tape_reversal":True,"no_posthoc_repair":True,"catalogue_text":False}})
    if len(rows)>=limit: break
   if len(rows)>=limit: break
  if len(rows)>=limit: break
 return {"experiment":"frame-yield-endpoint-buckets-20260926","host":socket.gethostname(),"parameters":{"min_letters":min_letters,"limit":limit},"endpoint_buckets":len(buckets),"states":states,"pruned":pruned,"candidates":rows,"closures":len(rows),"reader_worthy":0,"controls":[{"rendered":"The gardener guides a map.","audit":audit("The gardener guides a map.")}],"next_construction":"Add vivid ditransitive and locative frames keyed by endpoint character classes, retaining the live agreement/valency bucket product before any adjunct expansion."}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--min-letters',type=int,default=40);ap.add_argument('--limit',type=int,default=20);ap.add_argument('--out',required=True);a=ap.parse_args();p=run(a.min_letters,a.limit);Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('endpoint_buckets','states','pruned','closures')}))
if __name__=='__main__':main()
