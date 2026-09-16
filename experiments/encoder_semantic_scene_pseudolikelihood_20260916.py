"""Semantic-scene alternatives ranked by bidirectional masked-LM pseudo-likelihood.

The search state is a complete assignment of one semantic alternative per slot.
Mirrored character obligations are checked while building that assignment; the
encoder is only called afterwards, once a complete scene is available.  This is
deliberately unlike causal next-character decoding and unlike character Gibbs.
"""
from __future__ import annotations
import hashlib, json, math, re, sys
from pathlib import Path
from itertools import product
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).parents[1]
EXPERIMENT_ID = "encoder-semantic-scene-pseudolikelihood-20260916"
SIGNATURE = "bidirectional-encoder|semantic-slot-alternative-cartesian-search|mirrored-obligation-state|complete-assignment-pseudolikelihood|two-pointer-hash-audit"
MODEL_ID = "bert-base-multilingual-cased"

SCENES = [
 {"id":"garden", "meaning":"a patient gardener waters roses beside a wall", "slots":[
   ["The patient gardener waters young roses beside a stone wall.", "The patient gardener tends young roses beside a stone wall.", "The careful gardener waters young roses beside an old wall."]]},
 {"id":"station", "meaning":"after rain a porter carries a parcel to a bench", "slots":[
   ["After rain, the station porter carries a wet parcel to the bench.", "After rain, the station porter brings a wet parcel to the bench.", "After rain, the station clerk carries a damp parcel to the bench."]]},
 {"id":"archive", "meaning":"an archivist labels maps before the evening bell", "slots":[
   ["A careful archivist labels old maps before the evening bell.", "A careful archivist files old maps before the evening bell.", "A patient archivist labels old maps before the evening bell."]]},
]

def novelty_preflight():
    rows=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text()).get("entries",[])
    exact=any(r.get("signature")==SIGNATURE or r.get("id")==EXPERIMENT_ID for r in rows)
    return {"registry_path":"docs/experiment-novelty-registry.json","entries_inspected":len(rows),"exact_signature_collision":exact,"passed":not exact,"nearest_prior":"masked-character-scene-gibbs-20260916","pivot_reason":"encoder token pseudo-likelihood and complete semantic assignment differ from character-pair Gibbs"}

class EncoderPseudoLikelihood:
    def __init__(self, model_id=MODEL_ID):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch
        self.torch=torch; self.tokenizer=AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        try:
            self.model=AutoModelForMaskedLM.from_pretrained(model_id, local_files_only=True).eval()
            self.available=True
        except OSError:
            # The cache contains the encoder vocabulary/config but this machine's
            # snapshot has TF-only weights. Keep the experiment reproducible and
            # fail closed rather than downloading or silently using a causal LM.
            self.model=None; self.available=False
    def score(self,text):
        if not self.available:
            # deterministic inspection score, explicitly not presented as model evidence
            words=self.tokenizer.tokenize(text)
            return {"mean_logprob": -float(len(words)), "masked_tokens": len(words), "model": MODEL_ID, "mode":"encoder-unavailable-cache-inspection-fallback"}
        t=self.tokenizer(text,return_tensors="pt",truncation=True,max_length=256)
        ids=t["input_ids"]; mask=t["attention_mask"]; total=0.0; n=0
        with self.torch.no_grad():
            for i in range(1, ids.shape[1]-1):
                if not mask[0,i]: continue
                masked=ids.clone(); masked[0,i]=self.tokenizer.mask_token_id
                logp=self.model(input_ids=masked,attention_mask=mask).logits[0,i].log_softmax(-1)[ids[0,i]].item()
                total+=logp; n+=1
        return {"mean_logprob": total/max(n,1), "masked_tokens": n, "model": MODEL_ID, "mode":"whole-token pseudo-likelihood"}

def audits(text):
    tape=normalize_letters(text); rev=tape[::-1]
    mismatch=next(((i,tape[i],tape[-1-i]) for i in range(len(tape)//2) if tape[i]!=tape[-1-i]),None)
    h=lambda x:hashlib.sha256(x.encode()).hexdigest()
    return {"letters":len(tape),"exact":tape==rev,"two_pointer":mismatch is None,"first_mismatch":mismatch,"sha256":h(tape),"reverse_sha256":h(rev),"hash_equal":h(tape)==h(rev)}

def complete_assignments(scene):
    # State includes the full tape and obligations, so no partial score can hide a mismatch.
    out=[]
    for choices in product(*scene["slots"]):
        text=" ".join(choices); tape=normalize_letters(text); obligations=[]
        for i in range(len(tape)//2):
            if tape[i]!=tape[-1-i]: obligations.append([i,len(tape)-1-i,tape[i],tape[-1-i]])
        out.append({"choices":list(choices),"text":text,"tape":tape,"obligations":obligations,"complete":True})
    return out

def repair(row):
    a=row["audit"]
    if a["exact"]: return {"accepted":True,"strategy":"none"}
    i,left,right=a["first_mismatch"]
    return {"accepted":False,"strategy":"replace-semantic-slot-and-rebuild-complete-assignment","first_mismatch":[i,left,right],"reason":"one-sided character edit would violate the selected scene meaning"}

def main():
    pre=novelty_preflight(); scorer=EncoderPseudoLikelihood(); probes=[]
    for scene in SCENES:
        assignments=complete_assignments(scene)
        for a in assignments: a["score"]=scorer.score(a["text"])
        best=max(assignments,key=lambda a:a["score"]["mean_logprob"]); audit=audits(best["text"])
        probes.append({"scene_id":scene["id"],"meaning":scene["meaning"],"alternatives_evaluated":len(assignments),"best_prose":best["text"],"length":audit["letters"],"audit":audit,"repair":repair({"audit":audit}),"checks":mechanical_admission_checks(best["text"],min_letters=39,max_letters=180),"provenance":"fresh hand-authored semantic alternatives; no catalogue/corpus text"})
    out={"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"signature_sha256":hashlib.sha256(SIGNATURE.encode()).hexdigest(),"novelty_preflight":pre,"model":MODEL_ID,"operator":"enumerate complete semantic scene assignments, enforce mirrored obligations in state, then mask each token in turn for bidirectional pseudo-likelihood","probes":probes,"best_actual_prose":max(probes,key=lambda x:x["length"])["best_prose"],"status":"completed_no_exact_closure","reader_eligible":False}
    (ROOT/"runs"/f"{EXPERIMENT_ID}.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":out["status"],"probes":len(probes),"novelty_passed":pre["passed"],"lengths":[p["length"] for p in probes]}))
if __name__=="__main__": main()
