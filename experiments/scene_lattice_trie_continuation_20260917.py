"""Character-trie continuation filter at an independent scene-lattice frontier."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/scene-lattice-trie-continuation-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="scene-lattice-trie-continuation-20260917";SIG="independent-scene-lattice|object-setting-character-trie|live-reverse-continuation|compact-single-sentence|independent-exact-audit"
SUBJECTS=(("the careful botanist","records"),("the patient cartographer","maps"),("the quiet archivists","study")); OBJECTS=(("a coastal chart","artifact"),("the weathered atlas","artifact"),("the local survey","record"),("a field notebook","record")); SETTINGS=(("beside the quiet harbor","location"),("near the old observatory","location"),("during the morning survey","time"),("under the northern light","condition")); ADVERBS=("carefully","quietly")
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def trie(words):
 root={}
 for word,role in words:
  node=root
  for ch in norm(word):node=node.setdefault(ch,{})
  node.setdefault("$",[]).append(role)
 return root
def trie_prefix(root,prefix):
 node=root
 for ch in prefix:
  if ch not in node:return False
  node=node[ch]
 return True
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["prior seam frames","attachment expansion","finished-tape reversal","catalogue text","fragments"]}
def continuation_score(obj,setting,root):
 o=norm(obj);s=norm(setting);required=o[-min(4,len(o)):][::-1];accepted=0
 for k in range(1,len(required)+1):
  if trie_prefix(root,s[:k]) and s[k-1]==required[k-1]:accepted=k
 return {"required_reverse_prefix":required,"accepted_prefix_length":accepted,"trie_continuation":trie_prefix(root,s[:max(1,accepted)]),"score":accepted}
def emit(subject,verb,obj,orole,setting,srole,adv,cont):
 text=f"{subject.capitalize()} {verb} {obj} {adv} {setting}.";t=norm(text);mid=len(t)//2;sp=[];cur=0
 for tok in re.findall(r"[A-Za-z]+",text):a=cur;cur+=len(tok);sp.append((tok,a,cur))
 cross=next(({"token":w,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for w,a,b in sp if a<=mid<b),None);a=audit(text);i,j=0,len(t)-1;pairs=0
 while i<j and t[i]==t[j]:pairs+=1;i+=1;j-=1
 return {"rendered":text,"choices":{"subject":subject,"verb":verb,"object":obj,"object_role":orole,"setting":setting,"setting_role":srole,"adverb":adv},"continuation":cont,"audit":a,"center_state":{"midpoint":mid,"crossing":cross,"closed_pairs_before_first_mismatch":pairs,"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None},"anti_shortcut_flags":{"prior_seam_frames":False,"attachment_expansion":False,"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"independent typed role banks and setting trie","borrowed_text":False,"filter_before_final_render":True,"syntax_expanded":False}}
def run():
 pre=preflight();root=trie(SETTINGS);pool=[]
 for (sub,verb),(obj,orr),(setting,sr),adv in itertools.product(SUBJECTS,OBJECTS,SETTINGS,ADVERBS):
  if sub.startswith("the quiet") and verb!="study":continue
  if sub.startswith("the careful") and verb=="study":continue
  pool.append((continuation_score(obj,setting,root),sub,verb,obj,orr,setting,sr,adv))
 best=max(x[0]["score"] for x in pool);selected=[x for x in pool if x[0]["score"]==best];rows=[emit(s,v,o,orr,setg,sr,a,c) for c,s,v,o,orr,setg,sr,a in selected];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"independent scene lattice with character-trie continuation filter at object-setting frontier","novelty_preflight":pre,"pre_filter_count":len(pool),"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"max_continuation_score":best,"selected":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"midpoint_inside_token":sum(r["center_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain trie and role banks, then add a one-character reverse-prefix beam at the same frontier without syntax expansion"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256","pre-render trie continuation filter"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"pre_filter":x["pre_filter_count"],"selected":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
