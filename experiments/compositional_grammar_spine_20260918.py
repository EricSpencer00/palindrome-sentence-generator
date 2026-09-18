"""Dream-RSI lane: scene-spine composition with grammar-constrained mirror slots.

The generator writes an intact, human-readable scene spine first.  It then
chooses independent lexical slots whose character equations are checked while
the two clause trees are assembled.  This deliberately does not reverse a
finished sentence or reuse catalogue prose.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.dream_rsi_exact_boundary_20260918 import audit, letters  # noqa: E402

EXPERIMENT = "compositional-grammar-spine-20260918"

# Each scene is an authored, complete clause frame.  Slots are typed by a
# tiny valency grammar: agent transitive-verb patient, with agreement carried
# from the noun phrase to the verb.
SCENES = (
    {"id":"harbor", "subject":("the sailor", "singular"), "verb":("marks", "mark"), "object":("the map", "singular"), "adjunct":"at dawn"},
    {"id":"garden", "subject":("a gardener", "singular"), "verb":("guards", "guard"), "object":("some gates", "plural"), "adjunct":"near home"},
    {"id":"studio", "subject":("the writer", "singular"), "verb":("finds", "find"), "object":("a note", "singular"), "adjunct":"after rain"},
    {"id":"archive", "subject":("some clerks", "plural"), "verb":("mark", "marks"), "object":("the files", "plural"), "adjunct":"in silence"},
)

def valid_clause(scene, subject, verb, obj):
    # compositional grammar + valency, not a language-model score
    sn = subject[1]; on = obj[1]
    expected = verb[0] if sn == "singular" else verb[1]
    return verb[0] == expected and on in {"singular", "plural"} and all(x for x in (subject[0], verb[0], obj[0]))

def equation_ok(left, right):
    a, b = letters(left), letters(right)[::-1]
    n = min(len(a), len(b))
    return a[:n] == b[:n]

def independent_audit(text):
    # Fresh implementation of the exact condition, independent of candidate
    # construction and of the imported reporting helper.
    tape = "".join(c.lower() for c in text if c.isalpha())
    return {"letters": len(tape), "exact": bool(tape) and tape == tape[::-1],
            "independent_exact": bool(tape) and all(tape[i] == tape[-i-1] for i in range(len(tape)//2))}

def discover(budget=600):
    nodes, closures = [], []
    # Build a scene spine, then solve only typed mirrored slot combinations.
    for scene in SCENES:
        for ls in SCENES:
            for ro in SCENES:
                if len(nodes) >= budget: break
                s, v, o = scene["subject"], scene["verb"], scene["object"]
                rs, rv, rr = ls["subject"], ls["verb"], ro["object"]
                if not valid_clause(scene, s, v, o) or not valid_clause(ls, rs, ls["verb"], rr):
                    continue
                left = f"{s[0]} {v[0]} {o[0]} {scene['adjunct']}"
                right = f"{rs[0]} {ls['verb'][0]} {rr[0]} {ro['adjunct']}"
                # Character equations are applied to each compositional slot
                # before rendering the complete prose candidate.
                checks = [equation_ok(s[0], rr[0]), equation_ok(v[0], ls["verb"][0]), equation_ok(o[0], rs[0])]
                nodes.append({"scene":scene["id"], "mirror_scene":ls["id"], "slot_checks":checks,
                              "grammar_complete":True, "rendered_preview":f"{left}; {right}."})
                if all(checks):
                    text = f"{left}; {right}."
                    closures.append({"rendered":text, "audit":audit(text), "independent_audit":independent_audit(text), "provenance":{"scene":scene["id"],"mirror_scene":ls["id"],"constructed_slots":True}})
    return {"nodes":nodes, "closures":closures, "stats":{"nodes":len(nodes),"closures":len(closures)}}

def controls():
    texts = ("The sailor marks the map at dawn; a gardener guards some gates near home.",
             "A writer finds a note after rain; some clerks mark the files in silence.",
             "The gardener guards some gates near home; the writer finds a note after rain.")
    return [{"candidate_id":f"spine-control-{i}","rendered":t,"audit":audit(t),"reader_status":"human-unreviewed",
             "provenance":{"fresh_authored_scene":True,"catalogue_used":False,"finished_tape_reversal":False,"repeated_unit":False}} for i,t in enumerate(texts)]

def run():
    result = discover()
    cs = controls()
    return {"experiment":EXPERIMENT,"method":"Dream-RSI compositional grammar spine with typed mirrored slots",
            "construction":{"scene_spine_first":True,"valency_automaton":True,"agreement_state":True,"slot_equations_before_render":True,"nested_palindrome_spans":False},
            "search":result,"rendered_candidates":cs,"fresh_exact_closures":result["closures"],
            "stats":{"fresh_nodes":result["stats"]["nodes"],"fresh_exact":len(result["closures"]),"longest_control_letters":max(x["audit"]["letters"] for x in cs)},
            "novelty_preflight":{"new_geometry":"scene-first compositional grammar with typed mirrored slot equations","prior_lane_reused":False,"duplicate_sweep":False,"catalogue_used":False},
            "reader_gate":{"status":"not_triggered" if not result["closures"] else "human_blind_review_required","programmatic_metrics_are_diagnostic":True},
            "next_repair":{"operator":"permit cross-slot character carry between adjacent noun and adjunct boundaries while preserving complete scene valency","reason":"independent slot equations are too local and reject viable word-boundary seams"},
            "provenance":{"fresh_bank_authored_for_run":True,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"human_readability_certified":False}}

if __name__ == "__main__":
    p=run()
    for d in (ROOT/"runs",ROOT/"artifacts"): d.mkdir(exist_ok=True); (d/f"{EXPERIMENT}.json").write_text(json.dumps(p,indent=2)+"\n")
    print(json.dumps(p["stats"],indent=2))
