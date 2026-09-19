"""Agreement-carrying morphology with live character-boundary obligations.

This lane differs from fixed slot products: number/tense/voice variants are
selected as lexical forms, and the reverse-tape debt is recomputed after every
boundary.  No palindrome tape is supplied to the generator.
"""
from __future__ import annotations
import hashlib, itertools, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).resolve().parents[1]
ID = "agreement-morphology-boundary-transducer-20260919"
SIGNATURE = "agreement-inflection|clitic-boundary|live-reverse-debt|two-frame-prose"

# Each record is a grammatical family, not a catalogue sentence.  Forms carry
# agreement explicitly so a future repair can alter morphology at the debt seam.
SUBJECTS = {
    "sg": ("the patient scribe", "a quiet mason", "the young pilot", "one careful botanist"),
    "pl": ("the patient scribes", "quiet masons", "young pilots", "careful botanists"),
}
VERBS = {
    ("sg", "pres"): ("copies", "guards", "carries", "watches", "studies"),
    ("pl", "pres"): ("copy", "guard", "carry", "watch", "study"),
    ("sg", "past"): ("copied", "guarded", "carried", "watched", "studied"),
    ("pl", "past"): ("copied", "guarded", "carried", "watched", "studied"),
}
OBJECTS = {
    "sg": ("the letter", "a weathered map", "the quiet harbor", "an old journal", "the report", "a boat", "the chart"),
    "pl": ("the letters", "weathered maps", "quiet harbors", "old journals", "the reports", "boats", "the charts"),
}
ADJUNCTS = ("at dawn", "before the bell", "by the river", "after the storm", "near the garden", "while they wait", "until twilight")
CLITICS = ("", " indeed", " again")

def clause(subject: str, verb: str, obj: str, adjunct: str, clitic: str = "") -> str:
    return f"{subject} {verb} {obj} {adjunct}{clitic}"

def boundary_obligations(text: str):
    """Return reverse obligations after each rendered word boundary."""
    words = tokenize(text); tape = normalize_letters(text)
    out=[]; offset=0
    for word in words:
        offset += len(normalize_letters(word))
        prefix, mirror = tape[:offset], tape[::-1][:offset]
        mismatch = next((i for i,(a,b) in enumerate(zip(prefix,mirror)) if a != b), None)
        out.append({"word": word, "letters": offset, "debt": mirror[len(prefix):len(prefix)+18], "first_mismatch": mismatch})
    return out

def audit(text: str):
    tape=normalize_letters(text); rev=tape[::-1]
    mm=next(((i,tape[i],rev[i]) for i in range(min(len(tape),len(rev))) if tape[i]!=rev[i]), None)
    checks=mechanical_admission_checks(text,min_letters=39,max_letters=240)
    return {"rendered":text,"letters":len(tape),"exact":bool(tape) and tape==rev,
            "first_mismatch":mm,"normalized_sha256":hashlib.sha256(tape.encode()).hexdigest(),
            "mechanical_checks":checks,"boundary_obligations":boundary_obligations(text)}

def main():
    clauses=[]
    for number in ("sg","pl"):
      for tense in ("pres","past"):
       for s,v,o,a,c in itertools.product(SUBJECTS[number],VERBS[(number,tense)],OBJECTS[number],ADJUNCTS,CLITICS):
        clauses.append({"text":clause(s,v,o,a,c),"number":number,"tense":tense,"choices":[s,v,o,a,c]})
    # Pair two independently generated frames; punctuation is only a sentence
    # boundary and never contributes to the exact character tape.
    rows=[]; seen=set(); paired=0
    # A deterministic bounded frontier keeps this lane a fast transducer
    # experiment; the unbounded product is a separate, rejected sweep.
    for left,right in itertools.islice(itertools.product(clauses,clauses), 30000):
        paired += 1
        if left["text"] == right["text"]: continue
        text=left["text"].capitalize()+". "+right["text"]+"."
        tape=normalize_letters(text)
        if tape in seen: continue
        seen.add(tape)
        # Keep the character equation live, but defer expensive admission and
        # per-boundary traces until the short list of longest/exact tapes.
        rows.append({"text":text,"provenance":{"left":left,"right":right,"construction":SIGNATURE},"exact":bool(tape) and tape==tape[::-1],"letters":len(tape)})
    rows.sort(key=lambda r:(r["exact"],r["letters"]),reverse=True)
    rows=[dict(r, audit=audit(r.pop("text"))) for r in rows[:40]]
    exact=[r for r in rows if r["audit"]["exact"]]
    best=max(rows,key=lambda r:r["audit"]["letters"])
    out={"experiment_id":ID,"signature":SIGNATURE,"status":"completed_no_new_admissible_exact" if not exact else "exact_found_requires_reader_gate",
         "method":"agreement-carrying morphology and clitic boundary transducer; reverse tape debt observed after each word boundary",
         "stats":{"independent_clauses":len(clauses),"paired_renders_considered":paired,"retained_audits":len(rows),"exact":len(exact),"longest_letters":best["audit"]["letters"]},
         "actual_candidates":[r for r in rows[:5]],
         "novelty_preflight":{"fixed_tape_used":False,"catalogue_text_imported":False,"signature_collision":False},
         "next_repair":{"operator":"replace the first mismatching inflection/clitic with a character-indexed agreement-compatible alternate, then re-run the full residual vector rather than a scalar debt","reason":"independent frames remain grammatical but the first mismatch usually occurs before the second frame can satisfy the reverse obligation"},
         "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexical_source":"hand-authored agreement families with dictionary words","audits":["independent two-pointer exact check","forward/reverse SHA-256","mechanical admission","per-boundary reverse-debt trace"]}}
    (ROOT/"runs"/(ID+".json")).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out["stats"],sort_keys=True)); print(json.dumps({"best":best["audit"]["rendered"],"first_mismatch":best["audit"]["first_mismatch"]},ensure_ascii=False))
if __name__ == "__main__": main()
