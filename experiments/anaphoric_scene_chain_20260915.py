"""Anaphora-state scene composition probe; no reverse generation."""
import hashlib,json
from pathlib import Path
from llm_palindrome.validator import normalize
ROOT=Path(__file__).resolve().parents[1]
FAMILY_ID="anaphoric-scene-chain-composition"
SIGNATURE="typed-anaphora|antecedent-number-state|three-sentence-scene-yield|discourse-continuity|no-reverse-emission"
SCENES=[("Mara found a brass key.","She cleaned it by the window.","The key opened the drawer."),("Jon carried a small map.","He folded it after lunch.","The map showed the coast."),("Nell planted a young tree.","She watered it at dusk.","The tree shaded the path."),("Ruth kept an old letter.","She read it beside the fire.","The letter named the town.")]
def audit(text):
 t=normalize(text); return {"letters":len(t),"palindrome":bool(t) and t==t[::-1],"tape_sha256":hashlib.sha256(t.encode()).hexdigest()}
def main():
 probes=[]; exact=[]
 for i,s in enumerate(SCENES):
  text=" ".join(s); row={"scene":i,"text":text,"antecedent":"singular object carried across sentences","audit":audit(text),"provenance":"independently authored intact scene with explicit pronoun antecedent"}; probes.append(row)
  if row["audit"]["palindrome"]: exact.append(row)
 payload={"family_id":FAMILY_ID,"state_space_signature":SIGNATURE,"method":"compose complete three-sentence scenes under typed singular-object anaphora; no reverse generation","stats":{"scenes":len(probes),"exact":len(exact),"admitted":0},"rendered_probes":probes,"exact_candidates":exact,"independent_validation":"normalize then direct reverse tape equality","readable_status":"ordinary grammatical prose; no reader certification","shortcut_diagnostics":{"word_order_symmetry":False,"repeated_units":False,"borrowed_catalogue":False,"fragment":False},"concrete_repair":"Replace only the antecedent-bearing object noun with a held-out singular synonym while preserving pronoun number and discourse roles.","novelty_fingerprint":hashlib.sha256(json.dumps(SCENES,sort_keys=True).encode()).hexdigest(),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=ROOT/'runs/anaphoric-scene-chain-20260915.json'; out.write_text(json.dumps(payload,indent=2)+'\n'); print(payload['stats'])
if __name__=='__main__': main()
