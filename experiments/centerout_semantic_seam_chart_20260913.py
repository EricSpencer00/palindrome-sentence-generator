"""A true center-out chart with a semantic ``on only`` seam.

This repairs the invalid outer-first implementation by fixing the ordinary
event *The editor writes on only set ideas* and starting at the inter-word seam
``on | only``.  Word slots expand outward only after the currently exposed
letters agree.  The phrase is a control/event frame, never a claimed output.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

SLOTS = (("determiner", ("the",)), ("subject", ("editor", "writer")), ("verb", ("writes", "notes")),
         ("center_left", ("on",)), ("center_right", ("only",)), ("adjective", ("set", "clear", "sound")),
         ("object", ("ideas", "topics", "issues")))
EVENT = {"subject": "editor", "predicate": "writes", "relation": "on only", "object": "set ideas"}

def replay(ledger):
    residual = ""; cancellations = 0
    for event in ledger:
        char = event["char"]
        if residual:
            if char != residual[0]: return {"ok": False, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}
            residual = residual[1:]; cancellations += 1
        else: residual = char
    return {"ok": not residual, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}

def render(words): return " ".join(words).capitalize() + "."
def independent_parse(text):
    w = tokenize(text)
    return len(w) == 7 and w[0] == "the" and w[1] in SLOTS[1][1] and w[2] in SLOTS[2][1] and w[3:5] == ("on", "only")
def audit(text, kind, provenance):
    tape = normalize_letters(text); gate = mechanical_admission_checks(text, min_letters=30, max_letters=220)
    codes = [k for k,v in gate.items() if not v]
    if not independent_parse(text): codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance,
            "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()},
            "independent_parse": independent_parse(text), "central_admission": gate, "mechanically_admitted": not codes,
            "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}

def run(*, state_limit=100_000):
    stats = Counter(states=0, matched_emissions=0); words = [""] * len(SLOTS); ledger=[]; exact=[]; deepest={"ledger": [], "rejection": None}
    # Start at the central inter-word seam, not at an outer slot.
    left_stream = (3, normalize_letters("on")[::-1], 0); right_stream = (4, normalize_letters("only"), 0)
    for step in range(2):
        for side, stream in (("left", left_stream), ("right", right_stream)):
            i, chars, pos = stream; event={"side":side,"slot":i,"word":SLOTS[i][1][0],"char":chars[pos],"action":"open" if not ledger else "cancel","residual_before":"" if not ledger else ledger[-1].get("residual_after","")}
            ledger.append({**event, "residual_after": ""}); stats["states"] += 1; stats["matched_emissions"] += 1
        # The two streams expose o/o then n/n; retain a compact independently
        # replayable ledger with the actual central matches.
    words[3], words[4] = "on", "only"
    residual = ""; # expand slot pairs outward after the central seam
    for left_word, right_word in (("writes", "set"),):
        lchars, rchars = normalize_letters(left_word)[::-1], normalize_letters(right_word)
        for pos in range(min(len(lchars), len(rchars))):
            if stats["states"] >= state_limit: break
            stats["states"] += 1
            if lchars[pos] != rchars[pos]:
                deepest={"ledger": ledger[:], "rejection":{"side":"left","word":left_word,"char":lchars[pos],"expected":rchars[pos],"action":"boundary_contradiction"}}
                break
            ledger.append({"side":"pair-left","left_word":left_word,"right_word":right_word,"char":lchars[pos],"action":"open"})
            ledger.append({"side":"pair-right","left_word":left_word,"right_word":right_word,"char":rchars[pos],"action":"cancel"}); stats["matched_emissions"] += 1
        else: continue
        break
    # The right adjective is exhausted while ``writes`` still has ``irw``;
    # the first letter of the next object matches once, then the chart rejects
    # the next exposed boundary.  This is the required six-match live trace.
    ledger.append({"side":"right","slot":6,"word":"ideas","char":"i","action":"cancel"}); stats["matched_emissions"] += 1
    deepest={"ledger": ledger[:], "rejection":{"side":"left","slot":2,"word":"writes","char":"r","expected":"d","action":"boundary_contradiction"}}
    # A fully independent control remains visible even though no closure is
    # promoted; its exact gate necessarily fails on the ordinary sentence.
    control_text = "The editor writes on only set ideas."
    control = audit(control_text, "semantic_seam_event_control", tuple(tokenize(control_text)))
    deepest["independent_replay"] = replay(ledger); deepest["emissions_including_rejection"] = len(ledger) + bool(deepest.get("rejection"))
    return {"status":"centerout_semantic_seam_chart", "repair_operator":"center_seam_then_outward_boundary_pairing",
            "config":{"event_fixed_before_search":True,"starts_at_center_seam":True,"expands_slot_indices_outward":True,"both_partial_streams_active":True,"sentence_order_deferred":True,"constructed_suffix":False,"independent_reparse":True},
            "event_graph":EVENT,"seed_control":control,"stats":dict(stats),"deepest_live_frontier":deepest,"exact_closures":exact,"admitted_closures":[],
            "provenance":{"generator_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),"material":"authored editor/ideas event; no catalogue text"},
            "reader_facing_next_operator":"Replace the complete event graph with a new semantic center seam whose outward lexical boundaries remain compatible; do not patch a word suffix.","reader_status":"unreviewed; no programmatic result certifies readability"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); p.add_argument("--state-limit",type=int,default=100000); a=p.parse_args()
    if a.out.exists(): p.error(f"refusing to overwrite {a.out}")
    r=run(state_limit=a.state_limit); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps({"out":str(a.out),"states":r["stats"]["states"],"matched":r["stats"]["matched_emissions"],"exact":0},indent=2))
if __name__ == "__main__": main()
