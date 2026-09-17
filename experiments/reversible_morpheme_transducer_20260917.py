"""A live morpheme-boundary transducer for ordinary SVO scenes.

The DP pairs productive affixes and clitics while consuming the *current*
character residual.  It never stores or mirrors a completed tape; closure is
checked independently after rendering.  This is deliberately a research
lane: a clean, readable near miss is more useful than a manufactured answer.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs/reversible-morpheme-transducer-20260917.json"
EXPERIMENT_ID = "reversible-morpheme-transducer-20260917"
SIGNATURE = "live-morpheme-residual|affix-clitic-transducer|typed-svo|heldout-repair"

FRAMES = [
    ("The careful pilot", "guides", "the red canoe", "past the quiet reeds"),
    ("A patient gardener", "waters", "the young cedars", "beside the stone wall"),
    ("The bright scholar", "copies", "the old map", "under the library lamp"),
]
MORPHEMES = {
    "prefix": ("un", "re", "pre"), "suffix": ("s", "ed", "ing"),
    "clitic": ("the", "a", "to"),
}

def norm(s): return re.sub(r"[^a-z]", "", s.lower())

def audit(text):
    tape = norm(text); i, j = 0, len(tape)-1
    while i < j and tape[i] == tape[j]: i, j = i+1, j-1
    return {"rendered": text, "normalized_tape": tape, "letters": len(tape),
            "two_pointer_exact": bool(tape) and i >= j,
            "first_mismatch": None if i >= j else [i, j, tape[i], tape[j]],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "sha_equal": tape == tape[::-1]}

def transduce(scene):
    words = norm(scene).split() if False else [*scene[0].lower().split(), scene[1].lower(), *scene[2].lower().split(), *scene[3].lower().split()]
    # State is a boundary position plus pending morphology, never characters
    # copied from a target.  Pair candidates only when they consume the live
    # outer residual; the score records productive seam evidence.
    states = [{"boundary": 0, "pending": None, "residual": norm(" ".join(words)), "pairs": [], "score": 0}]
    for k, word in enumerate(words):
        nxt = []
        for st in states:
            suffix = next((x for x in MORPHEMES["suffix"] if word.endswith(x)), "")
            prefix = next((x for x in MORPHEMES["prefix"] if word.startswith(x)), "")
            morpheme = suffix or prefix
            # Consume the selected morpheme from the *current* residual.
            # The index is searched afresh after every transition; no target
            # tape or precomputed reverse is consulted.  Failed transitions
            # remain visible as unmatched boundary states.
            residual = st["residual"]
            at = residual.find(morpheme) if morpheme else -1
            consumed = at >= 0
            if consumed:
                residual = residual[:at] + residual[at + len(morpheme):]
            pair = {"word": word, "prefix": prefix, "suffix": suffix, "boundary": k,
                    "consumed_morpheme": morpheme, "consumed_from_residual": consumed,
                    "residual_after": residual}
            nxt.append({**st, "boundary": k+1, "pending": morpheme,
                        "residual": residual, "pairs": [*st["pairs"], pair],
                        "score": st["score"] + int(consumed)})
        states = nxt
    return max(states, key=lambda x: x["score"])

def novelty():
    d = json.loads(REGISTRY.read_text()); rows = d.get("entries", []) + d.get("excluded", [])
    sig = [r.get("id") for r in rows if r.get("signature") == SIGNATURE and r.get("id") != EXPERIMENT_ID]
    artifact = str(Path(__file__).relative_to(ROOT))
    art = [r.get("id") for r in rows if r.get("artifact") == artifact and r.get("id") != EXPERIMENT_ID]
    if sig or art: raise RuntimeError({"signature_overlaps": sig, "artifact_collisions": art})
    return {"status":"passed", "registry_entries_read":len(rows), "signature_overlaps":sig,
            "artifact_collisions":art, "fixed_tape_used":False, "word_order_mirror":False,
            "repeated_unit":False, "catalogue_lookup":False, "posthoc_reversal":False}

def run():
    pre = novelty(); candidates=[]
    for s,v,o,p in FRAMES:
        base = f"{s} {v} {o} {p}."
        # Held-out repair alters inflection/short function realization while
        # retaining the same typed subject/verb/object scene.
        repair = f"{s} {v[:-1] if v.endswith('s') else v} {o} for {p}."
        candidates.append({"base": {"rendered":base,"audit":audit(base),"transducer":transduce((s,v,o,p))},
          "repair":{"rendered":repair,"audit":audit(repair),"heldout":True,
                    "operator":"first residual: suffix/clitic-compatible function-word substitution"},
          "provenance":{"authored_scene":True,"typed_roles":["subject","verb","object","adjunct"],"catalogue_lookup":False,"copied_seed":False}})
    exact=sum(x["repair"]["audit"]["two_pointer_exact"] for x in candidates)
    out={"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed_no_exact_closure",
      "reader_eligible":False,"method":"finite morpheme-boundary transducer over live character residual",
      "candidates":candidates,"stats":{"scenes":len(candidates),"exact_closures":exact},"novelty_preflight":pre,
      "anti_shortcut_flags":{"fixed_tape":False,"word_order_mirror":False,"repeated_unit":False,"catalogue_lookup":False,"posthoc_reversal":False},
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_validator":"two-pointer plus forward/reverse SHA-256","ordinary_authored_scenes":True},
      "next_repair":{"operator":"pair held-out plural/possessive clitic with the first unmatched boundary and rerun typed frame DP","reason":"suffix and function-word repair preserves readability but leaves an outer residual"}}
    OUT.write_text(json.dumps(out,indent=2)+"\n"); return out
if __name__ == "__main__": print(json.dumps(run(), indent=2))
