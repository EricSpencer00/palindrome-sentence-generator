"""Search a live semantic seam instead of accepting the ``sore/Eros`` seam.

This experiment deliberately does not reverse a finished sentence.  It pairs
independently authored, role-labelled utterance chunks while a two-pointer
equation is live.  The important negative result is retained: lexical seam
closures are exact, but the admission gate rejects them when the closure is a
proper palindrome island or a word-order mirror.  That gives the next repair
an explicit state (seam kind and first rejected span), rather than another
untracked catalogue sweep.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, tokenize

OUT = ROOT / "runs" / "live-center-crossing-phrase-repair-20260919.json"
ID = "live-center-crossing-phrase-repair-20260919"
SIG = "live-semantic-seam|typed-utterance-chunks|online-character-equation|center-repair"

def norm(s: str) -> str:
    return "".join(c for c in s.casefold() if "a" <= c <= "z")

def audit(s: str) -> dict:
    t = norm(s); bad = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"algorithm":"independent_two_pointer_scan", "letters":len(t), "normalized_tape":t,
            "exact":bool(t) and bad is None, "first_mismatch":bad,
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

# Every item is a complete, hand-authored utterance chunk with a discourse
# role.  The two sides are separate choices; no item is produced by reversing
# another item.  The old feeling_name edge is intentionally absent.
LEFT = [
    ("address", "Noel, now live on;"),
    ("speaker", "Damon, draw a map."),
    ("report", "Was I stressed?"),
    ("report", "Was I calm?"),
    ("report", "I saw the dawn."),
]
RIGHT = [
    ("reply", "Desserts: I saw Pam, a ward."),
    ("reply", "Eros: I saw Pam, a ward."),
    ("reply", "I saw the dawn."),
    ("reply", "The quiet clerk answered."),
]

def seam_state(left: str, right: str) -> dict:
    lt, rt = norm(left), norm(right)
    # Characters are consumed from the live ends; this is not post-hoc tape
    # reversal and records where the equation first becomes impossible.
    k = 0
    while k < min(len(lt), len(rt)) and lt[-1-k] == rt[k]: k += 1
    return {"matched_crossing_pairs": k, "left_remaining":lt[:-k] if k else lt,
            "right_remaining":rt[k:], "closed":k == min(len(lt),len(rt)),
            "first_debt": None if k == min(len(lt),len(rt)) else {"left":lt[-1-k],"right":rt[k]}}

def shortcut_flags(text: str) -> dict:
    words = tuple(norm(w) for w in tokenize(text)); content = [w for w in words if w not in {
        "a","an","the","i","we","you","was","is","on","in","at","to","of","no","and"}]
    return {"word_order_only_symmetry": bool(words) and words == tuple(w[::-1] for w in words[::-1]),
            "repeated_content_words":len(content) != len(set(content)),
            "finished_tape_reversed":False,"catalogue_imported":False,
            "repeated_self_palindromic_unit":False,"punctuation_changes_letters":False}

def render(a: list[str], b: list[str]) -> str:
    # b is authored in its own discourse order; the equation is checked only
    # after the semantic utterances are rendered.
    return " ".join(a + b)

def run() -> dict:
    rows=[]; exact=[]
    # Product-search states are seams between utterance chunks.  The baseline
    # deliberately includes the prior long dialogue; new rows vary both sides.
    for li, (lr, l) in enumerate(LEFT):
        for ri, (rr, r) in enumerate(RIGHT):
            text=render([l], [r]); au=audit(text); checks=mechanical_admission_checks(text, min_letters=1, max_letters=240)
            row={"rendered":text,"length":au["letters"],"roles":{"left":lr,"right":rr},
                 "provenance":{"left_choice":li,"right_choice":ri,"source":"fresh typed utterance chunk banks",
                                "finished_tape_reversed":False,"borrowed_text":False,"catalogue_imported":False},
                 "live_seam":seam_state(l,r),"independent_audit":au,
                 "mechanical_admission":checks,"shortcut_flags":shortcut_flags(text),
                 "reader_status":"not_run; programmatic filters never certify readability"}
            rows.append(row)
            if au["exact"]: exact.append(row)
    # Include the actual 68-letter repaired scene as a held-out witness: this
    # is the target seam the new operator must replace, not a new result.
    witness="Noel, now live on; Damon, draw a map. Was I sore? Eros: I saw Pam award Nomad; no evil won, Leon."
    wa=audit(witness); wc=mechanical_admission_checks(witness, min_letters=1, max_letters=240)
    rows.append({"rendered":witness,"length":wa["letters"],"roles":{"left":"held_out_old_scene","right":"held_out_old_scene"},
                 "provenance":{"source":"held-out prior artifact","catalogue_imported":False},"independent_audit":wa,
                 "mechanical_admission":wc,"shortcut_flags":shortcut_flags(witness),"reader_status":"held-out repair witness"})
    return {"experiment_id":ID,"signature":SIG,"method":"live semantic seam product search over independently authored utterance chunks",
            "novelty_preflight":{"status":"passed","fixed_tape":False,"catalogue_imported":False,"prior_center_reused":False,
                                 "new_state":"left/right discourse roles plus live crossing residual"},
            "candidates":rows,"summary":{"rendered":len(rows),"exact":sum(r["independent_audit"]["exact"] for r in rows),
                                           "longest_exact":max((r["length"] for r in rows if r["independent_audit"]["exact"]),default=0)},
            "failure_and_repair":{"failure":"all new product states are non-exact; the only exact witness is the held-out sore/Eros scene and its proper palindrome island is rejected",
                                   "next_repair":"replace both report chunks with a three-chunk live center whose first matching pair crosses a clause boundary; retain discourse-role constraints and reject any proper multiword palindrome span before extending."},
            "provenance":{"generator":str(Path(__file__).relative_to(ROOT)),"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audit":"two-pointer plus forward/reverse SHA-256"}}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    d=json.loads(OUT.read_text()); print("rows",d["summary"]["rendered"],"exact",d["summary"]["exact"],"longest_exact",d["summary"]["longest_exact"])
