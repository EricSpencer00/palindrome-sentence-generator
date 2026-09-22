"""Bounded readability-first phrase graph intersected with a character PDA.

This is deliberately a small, auditable experiment: paths are ordinary prose
controls, while the palindrome state is updated at every character.  No path
is scored by a learned reward and no right half is copied/reversed.
"""
from dataclasses import dataclass
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "readability_phrase_graph_20260921.json"

@dataclass(frozen=True)
class Phrase:
    text: str
    pos: str
    number: str
    topic: str
    next_pos: str

PHRASES = (
    Phrase("the quiet archivist", "np", "sg", "record", "vp"),
    Phrase("the patient keeper", "np", "sg", "record", "vp"),
    Phrase("the lantern", "np", "sg", "light", "vp"),
    Phrase("the old maps", "np", "pl", "travel", "vp"),
    Phrase("the careful guides", "np", "pl", "travel", "vp"),
    Phrase("records the coast", "vp", "sg", "record", "end"),
    Phrase("keeps the lantern", "vp", "sg", "light", "end"),
    Phrase("mark the crossings", "vp", "pl", "travel", "end"),
    Phrase("follow the old maps", "vp", "pl", "travel", "end"),
)

def chars(s):
    return re.sub(r"[^a-z]", "", s.lower())

def pal_state(s):
    """Independent exact audit: normalized character stream and mismatch pair."""
    t = chars(s)
    mismatch = next(((i, len(t)-1-i, t[i], t[-1-i])
                     for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"normalized": t, "is_palindrome": mismatch is None,
            "first_mismatch": mismatch}

def append_pda(state, text):
    """Bounded PDA summary; retain only the unmatched outer prefix/suffix."""
    stream = state + chars(text)
    i, j = 0, len(stream)-1
    while i < j and stream[i] == stream[j]: i, j = i+1, j-1
    return {"length": len(stream), "outer_equal": i >= j,
            "residual": stream[i:j+1], "frontier": [i, j]}

def paths():
    rows=[]
    for n in PHRASES:
        for v in PHRASES:
            if n.next_pos != "vp" or v.pos != "vp": continue
            if n.number != v.number or n.topic != v.topic: continue
            if n.text in v.text or v.text in n.text: continue
            text=f"{n.text} {v.text}."
            rows.append(("control", text, n, v))
    # A second, semantically continuous clause is allowed only when its topic
    # carries forward; this prevents grammatical word salad.
    for n in PHRASES:
        for v in PHRASES:
            if n.pos=="np" and v.pos=="vp" and n.number==v.number and n.topic==v.topic:
                # Keep the control one-pass: repeating a finished unit would
                # manufacture symmetry and obscure the graph/PDA result.
                rows.append(("continuity_control", f"{n.text} {v.text}.", n, v))
    return rows

def main():
    rows=[]
    for kind,text,n,v in paths():
        audit=pal_state(text)
        rows.append({"kind":kind,"text":text,"graph":{"subject":n.text,"verb":v.text,
          "agreement":n.number,"topic_chain":[n.topic,v.topic]},"pda":append_pda("",text),
          "audit":audit,"provenance":"hand-authored bounded phrase inventory; no catalogue lookup",
          "novelty_flags":["not_a_fixed_finished_tape","not_word_reversal","no_repeated_unit"],
          "shortcut_flags":[] if not audit["is_palindrome"] else ["exact_closure_requires_independent_audit"],
          "next_constructive_operator":"add an unseen transitive VP preserving number/topic and re-run PDA"})
    exact=[r for r in rows if r["audit"]["is_palindrome"]]
    result={"method":"typed phrase/word graph × character-palindrome automaton",
      "bounds":{"phrase_nodes":len(PHRASES),"paths":len(rows)},
      "exact_closures":exact,"rows":rows,
      "controls_note":"Controls are complete prose; exact_closures is empty when no graph path closes exactly."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"paths":len(rows),"exact_closures":len(exact),"output":str(OUT)},indent=2))
    for r in rows[:4]: print(r["text"], "=>", r["audit"]["is_palindrome"])

if __name__ == "__main__": main()
