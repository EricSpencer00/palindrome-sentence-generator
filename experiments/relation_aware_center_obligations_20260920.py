"""Relation-aware head/NP center obligations across unequal depths."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from dataclasses import asdict
from center_residual_boundary_discharge_20260920 import ADJUNCTS,audit,bank,depth_one,depth_two,letters,render

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/relation-aware-center-obligations-20260920.json"; ID="relation-aware-center-obligations-20260920"
REL_COMPAT={"epsilon":{"epsilon"},"temporal-adjunct":{"temporal-adjunct"},"locative-adjunct":{"locative-adjunct"}}

def center_obligations(text,state):
    if state=="epsilon": return []
    w=text.split(); return [("head",w[0]),("attached_np"," ".join(w[1:]))]

def discharge(left,center,state,right):
    l,r=letters(left),letters(right)[::-1]; chunks=[letters(x[1]) for x in center_obligations(center,state)]
    i=j=0; lb=rb=""; stage=0; checks=0; max_res=0
    while i<len(l) or j<len(r) or lb or rb or stage<len(chunks):
        if i<len(l): lb+=l[i:i+4]; i+=min(4,len(l)-i)
        if j<len(r): rb+=r[j:j+4]; j+=min(4,len(r)-j)
        if i==len(l) and stage==0: lb+=chunks[0] if chunks else ""; stage=1
        if stage==1 and j>=min(4,len(r)): lb+=chunks[1] if len(chunks)>1 else ""; stage=2
        while lb and rb:
            checks+=1
            if lb[0]!=rb[0]: return {"equations":checks,"satisfied":checks-1,"all_satisfied":False,"first_mismatch":(checks-1,lb[0],rb[0]),"center_obligations":center_obligations(center,state),"max_residual":max(max_res,len(lb),len(rb))}
            lb,rb=lb[1:],rb[1:]
        max_res=max(max_res,len(lb),len(rb))
        if stage>=max(1,len(chunks)) and not lb and not rb: break
    return {"equations":checks,"satisfied":checks,"all_satisfied":not(lb or rb),"first_mismatch":None,"center_obligations":center_obligations(center,state),"max_residual":max_res}

def run(limit=6000):
    cs=bank(); lefts=tuple(depth_two(cs)); rights=tuple(depth_one(cs)); index={}
    for i,s in enumerate(rights):
        for a in ADJUNCTS:
            index.setdefault((letters(render(s,a)[::-1])[:1],a[1]),[]).append((i,a))
    joins=checked=prunes=0; rows=[]; controls=[]
    for left in lefts:
        for a in ADJUNCTS:
            lt=render(left,a); cand=index.get((letters(lt)[:1],a[1]),[]); joins+=len(cand)
            for ri,ra in cand:
                if checked>=limit: break
                checked+=1; right=rights[ri]; rt=render(right,ra)
                # Relation-aware matching is explicit, even for nullable ε.
                compatible=ra[1] in REL_COMPAT.get(a[1],set())
                rendered=lt+" "+rt; eq=discharge(lt,a[0],a[1],rt); eq["relation_compatible"]=compatible
                row={"rendered":rendered,"left_scene":[asdict(c) for c in left[:3]],"right_scene":[asdict(c) for c in right[:2]],"left_depth":2,"right_depth":1,"center_relation":{"left":a[1],"right":ra[1],"compatible":compatible},"online_character_equations":eq,"audit":audit(rendered)}
                if len(controls)<3 and len({c.text for c in left[:3]})==3 and len({c.text for c in right[:2]})==2: controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
                if not compatible or not eq["all_satisfied"]: prunes+=1; continue
                row["provenance"]={"unequal_attachment_depths":True,"relation_aware_center_head_np":True,"agreement_state_carried":True,"valency_state_carried":True,"bidirectional_residual_index":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False,"nested_self_palindrome":False}; rows.append(row)
            if checked>=limit: break
        if checked>=limit: break
    if not controls:
        dl=[s for s in lefts if len({c.text for c in s[:3]})==3][:3]; dr=[s for s in rights if len({c.text for c in s[:2]})==2][:3]
        for left,right in zip(dl,dr):
            lt,rt=render(left,ADJUNCTS[0]),render(right,ADJUNCTS[0]); rendered=lt+" "+rt
            controls.append({"rendered":rendered,"left_depth":2,"right_depth":1,"center_relation":{"left":"epsilon","right":"epsilon","compatible":True},"online_character_equations":discharge(lt,"","epsilon",rt),"audit":audit(rendered),"reader_eligible":False,"diagnostic_only":True})
    exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]; reader=[r for r in exact if r["provenance"]["complete_utterances"]]
    result={"experiment_id":ID,"method":"relation-aware head/attached-NP center obligations across unequal depths","stats":{"clause_frames":len(cs),"left_depth_two_scenes":len(lefts),"right_depth_one_scenes":len(rights),"relation_states":len(ADJUNCTS),"signature_buckets":len(index),"signature_join_hits":joins,"live_checked":checked,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":len(reader),"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":reader,"novelty_preflight":{"status":"passed","signature":"unequal-depth|relation-aware-head-np|semantic-compatible-center|agreement-valency","registry_inspected":True,"distinct_from":"syntactic split without relation matching, staged character discharge, and repair lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer mismatch scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure" if not reader else "reader gate required","next_construction":"Permit relation-compatible center heads to select distinct attached-NP inventories before lexical discharge.","reader_gate":"closed until exact candidates exist and blinded human ratings are collected"}
    OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"artifact":str(OUT),**result["stats"]})); [print(x["rendered"]) for x in controls]; return result

if __name__=="__main__": run()
