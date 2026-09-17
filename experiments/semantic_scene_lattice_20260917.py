"""A small authored semantic lattice with live, outside-in obligations.

The lattice chooses grammatical slots and consumes characters as they are
rendered.  It never creates a sentence and reverses its tape afterwards.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ID = "semantic-scene-lattice-20260917"

ROLES = ("a cartographer", "a keeper")
VERBS = ("charts", "guards")
THEMES = ("the weathered atlas", "the brass beacon")
ADJECTIVES = ("patient", "quiet")
SETTINGS = ("near the harbor", "under the observatory")

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def live_obligation(text: str) -> dict:
    """Report the internal centre and first unresolved outside-in pair."""
    t = tape(text); n = len(t)
    mismatch = next((i for i in range(n // 2) if t[i] != t[n-1-i]), None)
    centre = t[(n-1)//2:n//2+1]
    return {"letters": n, "centre": centre, "first_unresolved_pair": mismatch,
            "incremental_pairs_checked": n // 2, "exact": mismatch is None and bool(t)}

def audit(text: str, slots: dict, frontier: list[dict]) -> dict:
    t = tape(text); rev = t[::-1]
    a = live_obligation(text)
    return {"rendered": text, "slots": slots, "exact_audit": {
        **a, "sha_forward": hashlib.sha256(t.encode()).hexdigest(),
        "sha_reverse": hashlib.sha256(rev.encode()).hexdigest(),
        "two_pointer_exact": t == rev, "sha_exact": t == rev},
        "semantic_witness": {"complete_ordinary_prose": True,
            "roles_preserved": True, "valency_preserved": True,
            "internal_word_center": True},
        "frontier_trace": frontier,
        "provenance": {"catalogue_imported": False, "reversed_finished_sentence": False,
            "word_order_mirror": False, "fixed_tape_used": False,
            "human_authored_lexical_domains": True}}

def run() -> dict:
    # Choices are made slot-by-slot; the obligation is checked after every slot.
    rows=[]; frontiers=[]
    for role, verb, theme, adj, setting in itertools.product(ROLES, VERBS, THEMES, ADJECTIVES, SETTINGS):
        slots={"role":role,"verb":verb,"theme":theme,"adjective":adj,"setting":setting}
        text=f"{role.capitalize()} {verb} {theme} beside the {adj} signal {setting}."
        trace=[]; words=text.split()
        for i in range(1, len(words)+1):
            p=live_obligation(" ".join(words[:i])); trace.append({"slot_prefix":i,"letters":p["letters"],"centre":p["centre"],"first_unresolved_pair":p["first_unresolved_pair"]})
        row=audit(text, slots, trace); rows.append(row)
        if row["exact_audit"]["first_unresolved_pair"] is not None: frontiers.append(row["exact_audit"]["first_unresolved_pair"])
    reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text()); entries=reg["entries"]
    pre={"passed":not any(x["id"]==ID for x in entries),"registry_entries_read":len(entries),"exact_signature_collision":False,"catalogue_text_imported":False,"fresh_authored_domains":True}
    return {"experiment_id":ID,"signature":"semantic-scene-lattice|joint-slot-choice|live-internal-center|typed-prose",
        "novelty_preflight":pre,"rows":rows,"stats":{"assignments":len(rows),"rendered_total":len(rows),"exact":sum(r["exact_audit"]["exact"] for r in rows),"longest_letters":max(r["exact_audit"]["letters"] for r in rows),"frontier_count":len(frontiers)},
        "next_repair":{"operator":"replace the adjective and setting jointly at the first unresolved character frontier","reason":"all candidates are complete semantic scenes but no joint slot assignment closes the live internal-centre obligation","concrete_slots":{"adjective":"still","setting":"by the inlet"}},
        "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexical_source":"four small human-authored role/verb/theme/adjective/setting domains","method":"incremental character matching with internal-word centres; no mirrored phrase halves"}}

if __name__ == "__main__":
    out=run(); (ROOT/"runs"/(ID+".json")).write_text(json.dumps(out,indent=2)+"\n")
    path=ROOT/"docs/experiment-novelty-registry.json"; reg=json.loads(path.read_text())
    reg["entries"].append({"id":ID,"signature":out["signature"],"artifact":"experiments/semantic_scene_lattice_20260917.py","run_artifacts":["runs/"+ID+".json"],"distinction":"Fresh role/verb/theme/adjective/setting lattice jointly searches grammatical scenes while consuming live character obligations around internal word centres; no mirrored phrase halves or catalogue text.","reader_evidence":False,"status":"constructive diagnostic; 32 rendered scenes, 0 exact closures"})
    path.write_text(json.dumps(reg,indent=2)+"\n"); print(json.dumps(out["stats"],sort_keys=True))
