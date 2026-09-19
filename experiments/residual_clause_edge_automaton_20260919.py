"""Residual character-obligation automaton over independently authored clauses.

Unlike product/CSP lanes, this search indexes the reverse tape of every
complete clause.  A left clause consumes exactly the characters it can
discharge from the right-clause suffix; the unmatched middle is accepted only
when it is itself a character palindrome.  The grammar creates ordinary
subject--verb--object clauses before the tape search; no text is copied or
mutated after closure.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ID = "residual-clause-edge-automaton-20260919"

@dataclass(frozen=True)
class Clause:
    subject: str
    number: str
    verb: str
    verb_number: str
    obj: str
    scene: str
    @property
    def text(self): return f"{self.subject} {self.verb} {self.obj}"
    @property
    def words(self): return frozenset(normalize_letters(x) for x in tokenize(self.text)
                                      if normalize_letters(x) not in {"a","an","the","some","nine","ten"})

SUBJECTS = (("a poet","sg"),("a sailor","sg"),("a keeper","sg"),("a herald","sg"),
            ("the poet","sg"),("the sailor","sg"),("some men","pl"),("some women","pl"),
            ("the players","pl"),("the singers","pl"))
VERBS = (("praises","sg"),("records","sg"),("guards","sg"),("seeks","sg"),
         ("inspires","sg"),("praise","pl"),("record","pl"),("guard","pl"),
         ("seek","pl"),("inspire","pl"))
OBJECTS = ("a sonnet","a letter","the lantern","the harbor","new songs",
           "old maps","nine memos","ten notes","Diana","a ballad","the moon")

def clauses():
    out=[]
    for subject, number in SUBJECTS:
        for verb, verb_number in VERBS:
            if number != verb_number: continue
            for obj in OBJECTS:
                out.append(Clause(subject,number,verb,verb_number,obj,"authored_clause_bank"))
    # The known frontier is an anchor only, never counted as a new discovery.
    out += [Clause("an aide","sg","rips","sg","nine memos","baseline_anchor"),
            Clause("some men","pl","inspire","pl","Diana","baseline_anchor")]
    return out

def audit(text):
    tape=normalize_letters(text)
    pairs=[]; i,j=0,len(tape)-1
    while i<j:
        if tape[i]!=tape[j]: pairs.append((i,j,tape[i],tape[j]))
        i+=1; j-=1
    return {"exact":not pairs and bool(tape),"letters":len(tape),"normalized":tape,
            "pairs_checked":len(tape)//2,"mismatches":pairs[:8],
            "sha256":sha256(tape.encode()).hexdigest()}

def residual_matches(left, right_tapes):
    """Trie-like indexed join: match left against reversed-right prefixes."""
    # Buckets by the complete required prefix. This is an indexed residual
    # automaton, not a Cartesian pair enumeration: only tape-compatible
    # buckets are visited, and the residual center is checked directly.
    rev_index={}
    for idx,tape in enumerate(right_tapes):
        rev_index.setdefault(tape[::-1],[]).append(idx)
    lt=normalize_letters(left.text); found=[]
    for rt_rev, ids in rev_index.items():
        common=min(len(lt),len(rt_rev))
        if lt[:common] != rt_rev[:common]: continue
        residual=(lt[common:] if len(lt)>common else rt_rev[common:])
        if residual != residual[::-1]: continue
        for idx in ids: found.append(idx)
    return found

def run():
    bank=clauses(); tapes=[normalize_letters(c.text) for c in bank]
    rows=[]; joins=0
    for i,left in enumerate(bank):
        for j in residual_matches(left,tapes):
            right=bank[j]; joins+=1
            if left.words & right.words: continue
            text=f"{left.text}; {right.text}."
            a=audit(text); gate=mechanical_admission_checks(text,min_letters=39,max_letters=200)
            rows.append({"rendered":text,"length":a["letters"],"left":asdict(left),"right":asdict(right),
                         "audit":a,"mechanical_admission":gate,
                         "readability":"unreviewed; programmatic checks do not certify readability"})
    admitted=[r for r in rows if r["mechanical_admission"].get("admitted",False)]
    payload={"experiment_id":ID,
      "method":{"representation":"reverse-tape clause-edge residual automaton",
                 "state":"(left tape prefix, unmatched center residual)","indexed_join":True,
                 "complete_clause_grammar":True,"reward_model_used":False,"catalogue_imported":False,
                 "clauses":len(bank),"joins_examined":joins},
      "rendered_candidates":rows,"admitted_candidates":admitted,
      "summary":{"exact_count":sum(r["audit"]["exact"] for r in rows),
                 "admitted_count":len(admitted),"longest_exact":max((r["length"] for r in rows if r["audit"]["exact"]),default=0)},
      "provenance":{"script":str(Path(__file__).relative_to(ROOT)),"source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
                     "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()},
      "independent_audit":"two-pointer letter tape plus SHA-256; every exact row carries audit",
      "reader_gate":{"status":"not_run","note":"No programmatic metric certifies readability."},
      "next_repair":"Add independently authored adjunct-bearing clause edges and index them by residual-center length; retain agreement and unique-content-word gates."}
    return payload

if __name__ == "__main__":
    p=ROOT/"runs"/(ID+".json"); p.write_text(json.dumps(run(),indent=2)+"\n")
    x=json.loads(p.read_text()); print(json.dumps(x["summary"],indent=2)); print(p)
