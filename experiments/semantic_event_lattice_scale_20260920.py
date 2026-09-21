"""Scaled semantic event lattice with live character obligations.

Unlike a product of finished sentences, this lane first joins typed event
roles and world transitions, then consumes opposing character prefixes while
choosing lexical realizations.  The lattice is frozen and authored; it is not
the API catalogue and does not score completed strings with a reward model.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/semantic-event-lattice-scale-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "semantic-event-lattice-scale-20260920"
SIGNATURE = "frozen-authored-event-lattice|typed-role-transition|live-character-obligation|lexical-realization"

# Authored event schemas: lexical alternatives are selected only after the
# role/state edge is compatible.  The alternatives deliberately include
# ordinary, complete English verbs and nouns rather than catalogue phrases.
EVENTS = (
 ("guard_open", "guard", "open", "gate", "closed_gate", "open_gate", ("the guard opened the gate", "a guard opened a gate")),
 ("keeper_lift", "keeper", "lift", "flag", "down_flag", "up_flag", ("the keeper lifted the flag", "a keeper raised a flag")),
 ("pilot_light", "pilot", "light", "beacon", "dark_beacon", "lit_beacon", ("the pilot lit the beacon", "a pilot lit a beacon")),
 ("nurse_find", "nurse", "find", "key", "at_table", "has_key", ("the nurse found the key", "a nurse found a key")),
 ("scribe_mark", "scribe", "mark", "map", "blank_map", "marked_map", ("the scribe marked the map", "a scribe marked a map")),
 ("cook_fill", "cook", "fill", "jar", "empty_jar", "full_jar", ("the cook filled the jar", "a cook filled a jar")),
 ("farmer_harvest", "farmer", "harvest", "grain", "ripe_grain", "stored_grain", ("the farmer harvested grain", "a farmer gathered grain")),
 ("smith_shape", "smith", "shape", "iron", "raw_iron", "formed_iron", ("the smith shaped iron", "a smith shaped iron")),
 ("porter_carry", "porter", "carry", "crate", "sealed_crate", "moved_crate", ("the porter carried the crate", "a porter moved a crate")),
 ("teacher_read", "teacher", "read", "letter", "sealed_letter", "read_letter", ("the teacher read the letter", "a teacher read a letter")),
 ("judge_seal", "judge", "seal", "record", "open_record", "sealed_record", ("the judge sealed the record", "a judge sealed a record")),
 ("sailor_moor", "sailor", "moor", "boat", "loose_boat", "moored_boat", ("the sailor moored the boat", "a sailor moored a boat")),
 ("artist_hang", "artist", "hang", "portrait", "bare_wall", "hung_portrait", ("the artist hung the portrait", "an artist hung a portrait")),
 ("monk_ring", "monk", "ring", "bell", "silent_bell", "ringing_bell", ("the monk rang the bell", "a monk rang a bell")),
 ("warden_unlock", "warden", "unlock", "door", "locked_door", "open_door", ("the warden unlocked the door", "a warden unlocked a door")),
 ("singer_begin", "singer", "begin", "song", "unsung_song", "sung_song", ("the singer began the song", "a singer began a song")),
 ("gardener_water", "gardener", "water", "rose", "dry_rose", "wet_rose", ("the gardener watered the rose", "a gardener watered a rose")),
 ("healer_bind", "healer", "bind", "wound", "open_wound", "bound_wound", ("the healer bound the wound", "a healer bound a wound")),
 ("watcher_spot", "watcher", "spot", "ship", "hidden_ship", "seen_ship", ("the watcher spotted the ship", "a watcher saw a ship")),
 ("host_welcome", "host", "welcome", "guest", "expected_guest", "welcomed_guest", ("the host welcomed the guest", "a host welcomed a guest")),
 ("clerk_file", "clerk", "file", "claim", "new_claim", "filed_claim", ("the clerk filed the claim", "a clerk filed a claim")),
 ("poet_write", "poet", "write", "verse", "blank_verse", "written_verse", ("the poet wrote the verse", "a poet wrote a verse")),
 ("carver_cut", "carver", "cut", "wood", "raw_wood", "cut_wood", ("the carver cut the wood", "a carver cut wood")),
 ("rider_mount", "rider", "mount", "horse", "ready_horse", "mounted_horse", ("the rider mounted the horse", "a rider mounted a horse")),
 ("reader_open", "reader", "open", "book", "closed_book", "open_book", ("the reader opened the book", "a reader opened a book")),
 ("visitor_enter", "visitor", "enter", "yard", "open_gate", "inside_yard", ("the visitor entered the yard", "a visitor entered a yard")),
 ("signal_answer", "signalman", "answer", "call", "up_flag", "answered_call", ("the signalman answered the call", "a signalman answered a call")),
)

def letters(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t=letters(s); i,j=0,len(t)-1
    while i<j and t[i]==t[j]: i,j=i+1,j-1
    return {"letters":len(t),"pointer_exact":bool(t) and i>=j,
            "first_mismatch":None if i>=j else [i,j,t[i],t[j]],
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

def preflight():
    d=json.loads(REGISTRY.read_text()); rows=d.get("entries",[])+d.get("excluded",[])
    rows = [x for x in rows if x.get("id") != EXPERIMENT_ID]
    return {"status":"passed" if not any(x.get("signature")==SIGNATURE for x in rows) else "blocked",
            "signature_overlaps":[x["id"] for x in rows if x.get("signature")==SIGNATURE],
            "excluded_routes":["finished-tape reversal","post-hoc repair","catalogue/API text","repeated self-palindromic units"]}

def run():
    p=preflight()
    if p["status"]!="passed": raise RuntimeError(p)
    # A shared state is the only bridge: one event's effect must satisfy the
    # next event's precondition.  This prunes before any prose is complete.
    edges=[]; semantic_prunes=0; expanded=0
    for a in EVENTS:
      for b in EVENTS:
        expanded+=1
        compatible = a[5] == b[4]
        if compatible: edges.append((a,b))
        else: semantic_prunes+=1
    character_prunes=0; rows=[]; states=0
    for a,b in edges:
      for la in a[6]:
       for rb in b[6]:
        states+=1; left=letters(la); right=letters(rb)[::-1]
        n=min(len(left),len(right))
        if left[:n] != right[:n]: character_prunes+=1; continue
        text=la+", then "+rb+"."
        rows.append({"rendered":text,"audit":audit(text),"event_edge":[a[0],b[0]],
          "provenance":{"semantic_edge_before_render":True,"lexical_alternatives_authored":True,
          "live_character_obligation":True,"catalogue_text":False,"finished_tape_reversal":False,
          "post_hoc_repair":False,"repeated_units":False}})
    controls=[]
    for a,b in edges[:8]:
      text=a[6][0]+", then "+b[6][0]+"."
      controls.append({"rendered":text,"audit":audit(text),"reader_eligible":False,
       "provenance":{"intact_authored_prose":True,"semantic_edge":a[5]==b[4],"shuffled_control":False}})
    exact=[x for x in rows if x["audit"]["pointer_exact"] and x["audit"]["sha256_forward"]==x["audit"]["sha256_reverse"] and x["audit"]["letters"]>38]
    out={"experiment_id":EXPERIMENT_ID,"method":"scaled frozen semantic event lattice with prelexical typed-state edges and live character obligations",
      "config":{"events":len(EVENTS),"lexical_realizations":sum(len(x[6]) for x in EVENTS),"semantic_before_surface":True,"post_search_scoring":False},
      "stats":{"event_pairs_expanded":expanded,"semantic_valid_edges":len(edges),"semantic_prunes":semantic_prunes,"obligation_states":states,"character_prunes":character_prunes,"rendered_candidates":len(rows),"prose_controls":len(controls),"exact_gt38":len(exact),"max_control_letters":max(x["audit"]["letters"] for x in controls)},
      "rendered_candidates":rows[:20],"prose_controls":controls,"exact_candidates":exact,"novelty_preflight":p,
      "provenance":{"independent_audit":["two-pointer","forward/reverse SHA-256"],"frozen_authored_lattice":True,"catalogue_text":False,"reader_evidence":False},
      "next_operator":"carry typed recipient/theme and discourse focus through multi-edge event paths before lexical realization","status":"fresh exact >38 requires blinded human reading" if exact else "no exact >38 closure; intact event-lattice controls retained"}
    RUN.write_text(json.dumps(out,indent=2)+"\n"); return out
if __name__=="__main__": print(json.dumps(run(),indent=2))
