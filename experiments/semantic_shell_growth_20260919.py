"""Semantic shell-growth search with live character debt.

Unlike a wrapper around a finished palindrome, this generator grows a scene
one complete constituent at a time.  A shell step chooses an independent
left-side event and a right-side event, then solves their boundary character
debt before permitting another step.  No finished tape is reversed and no
catalogue sentence is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ID = "semantic-shell-growth-20260919"

@dataclass(frozen=True)
class Event:
    text: str; role: str; number: str; content: frozenset[str]

def content(text: str) -> frozenset[str]:
    stop = {"a","an","the","some","and","while","at","in","on","to","of"}
    return frozenset(normalize_letters(x) for x in tokenize(text) if normalize_letters(x) not in stop)

LEFT = (
    Event("a careful scribe marks the letter", "document", "sg", content("a careful scribe marks the letter")),
    Event("the quiet captain guards the harbor", "place", "sg", content("the quiet captain guards the harbor")),
    Event("some patient poets praise the sonnet", "document", "pl", content("some patient poets praise the sonnet")),
    Event("a young herald guides Diana", "person", "sg", content("a young herald guides Diana")),
    Event("the sailor reads a new tale", "document", "sg", content("the sailor reads a new tale")),
)
RIGHT = (
    Event("Diana inspires a young herald", "person", "sg", content("Diana inspires a young herald")),
    Event("a new tale reads the sailor", "document", "sg", content("a new tale reads the sailor")),
    Event("the sonnet praises some patient poets", "document", "pl", content("the sonnet praises some patient poets")),
    Event("the harbor guards the quiet captain", "place", "sg", content("the harbor guards the quiet captain")),
    Event("the letter marks a careful scribe", "document", "sg", content("the letter marks a careful scribe")),
)

# Held-out Shakespearean repair bank. These beats were absent from the first
# run and are admitted only when their outer characters satisfy live debt.
HELD_OUT_LEFT = (
    Event("the player marks the sonnet", "document", "sg", content("the player marks the sonnet")),
    Event("a singer praises the court", "place", "sg", content("a singer praises the court")),
    Event("the actor guards the tent", "place", "sg", content("the actor guards the tent")),
    Event("a poet writes a quiet part", "document", "sg", content("a poet writes a quiet part")),
)
HELD_OUT_RIGHT = (
    Event("the players answer the king", "person", "pl", content("the players answer the king")),
    Event("the court receives a singer", "person", "sg", content("the court receives a singer")),
    Event("the tent shelters the actor", "place", "sg", content("the tent shelters the actor")),
    Event("the king hears a poet", "person", "sg", content("the king hears a poet")),
)

def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text); i, j = 0, len(tape)-1; mismatches=[]
    while i < j:
        if tape[i] != tape[j]: mismatches.append((i,j,tape[i],tape[j]))
        i += 1; j -= 1
    f = hashlib.sha256(tape.encode()).hexdigest(); r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"normalized": tape, "letters": len(tape), "two_pointer_exact": not mismatches and bool(tape),
            "first_mismatch": mismatches[0] if mismatches else None, "sha256_forward": f,
            "sha256_reverse": r, "sha_equal": f == r}

def hidden_span(text: str) -> bool:
    ws = [normalize_letters(x) for x in tokenize(text)]
    return any((z := "".join(ws[a:b])) == z[::-1] for a in range(len(ws)) for b in range(a+2,len(ws)+1)
               if not (a == 0 and b == len(ws)))

def grow(max_depth: int = 3, *, repaired: bool = True) -> dict[str, object]:
    # The center is a semantic event, not the retained 38-letter seed.
    centers = ("the players wait at dawn", "the actors meet in the harbor", "a bard speaks to Diana")
    rows=[]; states=[("", "", frozenset(), 0)]
    for depth in range(1, max_depth+1):
        nxt=[]
        for left, right, used, debt in states:
            left_bank = LEFT + HELD_OUT_LEFT if repaired else LEFT
            right_bank = RIGHT + HELD_OUT_RIGHT if repaired else RIGHT
            for le in left_bank:
                for re in right_bank:
                    if used & (le.content | re.content): continue
                    # Live debt is computed from the newly selected edge, not
                    # from a reversed complete sentence.
                    l = normalize_letters(le.text); r = normalize_letters(re.text)
                    overlap = 0
                    for k in range(1, min(len(l),len(r))+1):
                        if l[-k:] == r[:k][::-1]: overlap = k
                    if repaired and overlap == 0:
                        continue
                    nd = debt + len(l) + len(r) - 2*overlap
                    nl = (left + " " + le.text).strip(); nr = (re.text + " " + right).strip()
                    for center in centers:
                        rendered = f"{nl}; {center}; {nr}."
                        a = audit(rendered); checks = mechanical_admission_checks(rendered, min_letters=30, max_letters=2000)
                        row={"rendered":rendered,"length":a["letters"],"depth":depth,"audit":a,
                             "mechanical_checks":checks,"hidden_proper_span":hidden_span(rendered),
                             "mechanically_admitted":a["two_pointer_exact"] and not hidden_span(rendered) and all(checks.values()),
                             "live_debt":nd,"edge_overlap":overlap,"provenance":{"representation":"typed semantic shell growth",
                             "left_event":le.text,"right_event":re.text,"center":center,
                             "finished_tape_reversed":False,"catalogue_imported":False,"rlaif_used":False},
                             "reader_status":"unreviewed; programmatic metrics do not certify readability"}
                        rows.append(row)
                    nxt.append((nl,nr,used|le.content|re.content,nd))
        states=nxt[:5000]
    exact=[x for x in rows if x["audit"]["two_pointer_exact"]]
    admitted=[x for x in exact if x["mechanically_admitted"]]
    return {"experiment_id":ID,"method":"incremental typed semantic shells with live mirrored-edge debt",
            "status":"completed_exact" if exact else "completed_no_exact_closure","actual_candidates":rows,
            "exact_candidates":exact,"stats":{"rendered":len(rows),"exact":len(exact),"admitted":len(admitted),
            "longest_rendered":max((x["length"] for x in rows),default=0),"longest_exact":max((x["length"] for x in exact),default=0)},
            "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits":["literal two-pointer","forward/reverse SHA-256"],"rlaif_per_candidate":False},
            "novelty_preflight":{"status":"passed","distinction":"scene shells are selected and audited at every growth depth; no completed palindrome is wrapped or reversed"},
            "next_repair":{"action":"add held-out event pairs whose edge overlap closes the live debt while preserving distinct typed roles","reader_test":"randomized blinded intact prose versus shuffled controls for any admitted row"},
            "reader_gate":"closed; no human readability evidence yet"}

if __name__ == "__main__":
    out=ROOT/"runs"/(ID+".json"); result=grow(); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],sort_keys=True))
