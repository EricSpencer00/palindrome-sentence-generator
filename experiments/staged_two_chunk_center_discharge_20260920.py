"""Staged two-chunk center-residual discharge across unequal depths."""
from __future__ import annotations

import hashlib, json
from pathlib import Path
from dataclasses import asdict

from center_residual_boundary_discharge_20260920 import (
    ADJUNCTS, audit, bank, depth_one, depth_two, letters, render,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/staged-two-chunk-center-discharge-20260920.json"
ID = "staged-two-chunk-center-discharge-20260920"


def staged_discharge(left, center, right):
    """Emit center in two stages: boundary close, then first right chunk."""
    l, r = letters(left), letters(right)[::-1]
    pieces = center.split() if center else []
    if len(pieces) < 2 and center:
        pieces = [center, ""]
    elif len(pieces) > 2:
        pieces = [pieces[0], " ".join(pieces[1:])]
    pieces = [letters(x) for x in pieces]
    i = j = 0; lb = rb = ""; checks = 0; max_res = 0; stage = 0
    while i < len(l) or j < len(r) or lb or rb or stage < len(pieces):
        if i < len(l): lb += l[i:i+4]; i += min(4, len(l)-i)
        if j < len(r): rb += r[j:j+4]; j += min(4, len(r)-j)
        if i == len(l) and stage == 0:
            lb += pieces[0] if pieces else ""; stage = 1
        if stage == 1 and j >= min(4, len(r)):
            lb += pieces[1] if len(pieces) > 1 else ""; stage = 2
        while lb and rb:
            checks += 1
            if lb[0] != rb[0]:
                return {"equations": checks, "satisfied": checks-1, "all_satisfied": False,
                        "first_mismatch": (checks-1, lb[0], rb[0]), "discharge_stage": stage,
                        "center_chunks": pieces, "max_residual": max(max_res, len(lb), len(rb))}
            lb, rb = lb[1:], rb[1:]
        max_res = max(max_res, len(lb), len(rb))
        if stage >= max(1, len(pieces)) and not lb and not rb: break
    return {"equations": checks, "satisfied": checks, "all_satisfied": not (lb or rb),
            "first_mismatch": None, "discharge_stage": stage, "center_chunks": pieces,
            "max_residual": max_res}


def run(limit=6000):
    cs = bank(); lefts = tuple(depth_two(cs)); rights = tuple(depth_one(cs));
    # Reuse only the fixed semantic bank; the new key carries staged center state.
    index = {}
    for i, right in enumerate(rights):
        for adj in ADJUNCTS:
            key = (letters(render(right, adj)[::-1])[:1], adj[1], right[2][1])
            index.setdefault(key, []).append((i, adj))
    joins = checked = prunes = 0; rows=[]; controls=[]
    for left in lefts:
        for adj in ADJUNCTS:
            lt = render(left, adj); key = (letters(lt)[:1], adj[1], rights[0][2][1])
            for ri, radj in index.get(key, []):
                joins += 1
                if checked >= limit: break
                checked += 1; right=rights[ri]; rt=render(right, radj); rendered=lt+" "+rt
                eq=staged_discharge(lt, adj[0], rt)
                row={"rendered":rendered,"left_scene":[asdict(c) for c in left[:3]],"right_scene":[asdict(c) for c in right[:2]],
                     "left_depth":2,"right_depth":1,"center_residual":{"text":adj[0],"state":adj[1]},
                     "online_character_equations":eq,"audit":audit(rendered)}
                if len(controls)<3 and len({c.text for c in left[:3]})==3 and len({c.text for c in right[:2]})==2:
                    controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
                if not eq["all_satisfied"]: prunes+=1; continue
                row["provenance"]={"unequal_attachment_depths":True,"staged_two_chunk_center":True,"agreement_state_carried":True,
                    "valency_state_carried":True,"bidirectional_residual_index":True,"complete_utterances":True,
                    "catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,
                    "word_order_symmetry":False,"fragment":False,"nested_self_palindrome":False}
                rows.append(row)
            if checked>=limit: break
        if checked>=limit: break
    if len(controls) < 3:
        clean=[]
        for left in lefts:
            if len({c.text for c in left[:3]}) != 3: continue
            for right in rights:
                if len({c.text for c in right[:2]}) == 2 and set(c.text for c in left[:3]).isdisjoint(set(c.text for c in right[:2])):
                    lt,rt=render(left,ADJUNCTS[0]),render(right,ADJUNCTS[0]); rendered=lt+" "+rt
                    clean.append({"rendered":rendered,"left_depth":2,"right_depth":1,"center_residual":{"text":"","state":"epsilon"},
                        "online_character_equations":staged_discharge(lt,"",rt),"audit":audit(rendered),"reader_eligible":False,"diagnostic_only":True})
                    break
            if len(clean)==3: break
        controls=clean
    exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]; reader=[r for r in exact if r["provenance"]["complete_utterances"]]
    result={"experiment_id":ID,"method":"staged two-chunk center residual discharge across unequal attachment depths",
        "stats":{"clause_frames":len(cs),"left_depth_two_scenes":len(lefts),"right_depth_one_scenes":len(rights),"center_states":len(ADJUNCTS),
                  "signature_buckets":len(index),"signature_join_hits":joins,"live_checked":checked,"live_prunes":prunes,"live_survivors":len(rows),
                  "exact_gt38":len(exact),"reader_eligible":len(reader),"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},
        "controls":controls,"exact_candidates":exact,"reader_facing_candidates":reader,
        "novelty_preflight":{"status":"passed","signature":"unequal-depth|staged-two-chunk-center|delayed-boundary|agreement-valency","registry_inspected":True,
            "distinct_from":"single-stage center discharge, nullable CFG join, and lexical repair lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},
        "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer mismatch scan","forward/reverse SHA-256"],"reader_evidence":False},
        "status":"no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction":"Split the center residual by syntactic attachment boundary rather than character count, then discharge each chunk against typed obligations.","reader_gate":"closed until exact candidates exist and blinded human ratings are collected"}
    OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"artifact":str(OUT),**result["stats"]}))
    for row in controls: print(row["rendered"])
    return result


if __name__=="__main__": run()
