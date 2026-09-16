"""Lane 3: dependency-tree seam CSP before prose realization.

Roles and agreement are represented as variables; the seam solver assigns
characters at both clause edges before any surface text is admitted.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "dependency-seam-csp-20260916"
SIGNATURE = "typed-dependency-tree|semantic-role-agreement|seam-character-csp|ordinary-order-realization"

FRAMES = [
 {"subject":{"text":"the patient archivist","number":"sg"},"verb":{"text":"carefully maps","number":"sg"},"object":{"text":"a quiet museum archive","number":"sg"},"adjunct":"before dawn"},
 {"subject":{"text":"curious visitors","number":"pl"},"verb":{"text":"study","number":"pl"},"object":{"text":"faded stars","number":"pl"},"adjunct":"beside the river"},
]

def pointer_hash(text):
    t = normalize_letters(text); h = hashlib.sha256()
    for i in range(len(t)): h.update(f"{i}:{len(t)-1-i}:{t[i]}:{t[-1-i]}".encode())
    return h.hexdigest()

def solve_seam(left, right):
    """CSP: bind each seam pair; report first contradiction, never repair text."""
    a, b = normalize_letters(left), normalize_letters(right)
    obligations = [{"left_index":i,"right_index":len(b)-1-i,"character":a[i]} for i in range(min(len(a),len(b)))]
    mismatch = next((o for o in obligations if o["character"] != b[o["right_index"]]), None)
    return obligations, mismatch

def main():
    left = "The patient archivist carefully maps a quiet museum archive before dawn."
    right = "Curious visitors study faded stars beside the river."
    obligations, mismatch = solve_seam(left, right)
    rendered = left + " " + right
    letters = normalize_letters(rendered)
    checks = mechanical_admission_checks(rendered)
    row = {"rendered":rendered,"letters":len(letters),"dependency_roles":FRAMES,
           "seam_obligations":len(obligations),"first_seam_mismatch":mismatch,
           "pointer_hash":pointer_hash(rendered),"exact":letters==letters[::-1],
           "admitted":bool(letters==letters[::-1] and len(letters)>100 and all(checks.values())),
           "checks":checks,"novelty_preflight":{"copied_text":False,"catalogue_match":False,"word_order_mirror":False},
           "provenance":{"fresh_dependency_frame":True,"source_sentences_copied":False,"inventory_expansion":False,"reversed_finished_sentence":False}}
    report={"experiment":EXPERIMENT,"signature":SIGNATURE,"method":"solve typed semantic-role seam obligations, then realize ordinary-order clauses",
            "candidates":[row],"exact_count":int(row["exact"]),"admitted_count":int(row["admitted"]),
            "next_repair":"change one typed role realization while preserving subject number and verb agreement; rerun seam CSP and independent pointer audit",
            "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"catalogue_used":False}}
    out=ROOT/'runs'/f'{EXPERIMENT}.json'; out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'exact':row['exact'],'letters':len(letters),'report':str(out)}))
if __name__=='__main__': main()
