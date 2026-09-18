"""Bounded direct Ollama authoring experiment with independent audit evidence.

The repair is diagnostic only: it mirrors mismatched letters mechanically and is
never presented as authored prose or counted as a reader candidate.
"""
import hashlib, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/direct-joint-authoring-20260917.json"
MODEL = "gpt-oss:20b"
PROMPT = ("Author ONE original, grammatical English sentence, 100 or more English letters. "
          "Build the sentence jointly from the center outward: every letter pair must be "
          "chosen together so that, ignoring case, spaces and punctuation, the complete "
          "sentence is an exact palindrome. Use one concrete scene, varied distinct words, "
          "and natural syntax. No half-then-reverse trick, repeated clauses/words, quotation, "
          "catalogue example, list, or fragment. Output only the sentence.")

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def two_pointer(t):
    mismatches=[]; i,j=0,len(t)-1
    while i<j:
        if t[i] != t[j]: mismatches.append({"left":i,"right":j,"a":t[i],"b":t[j]})
        i += 1; j -= 1
    return mismatches
def repair_operator(s):
    """Mirror left chars onto right chars; output remains quarantined."""
    chars=list(norm(s)); i,j=0,len(chars)-1
    while i<j: chars[j]=chars[i]; i+=1; j-=1
    return "".join(chars)
def audit(s, catalogue):
    t=norm(s); words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",s)
    lower=[w.lower().replace("'","") for w in words]
    mism=two_pointer(t)
    cat= t in catalogue
    return {"text":s,"normalized_sha256":hashlib.sha256(t.encode()).hexdigest(),
            "letters":len(t),"exact":bool(t) and not mism,"mismatch_count":len(mism),
            "first_mismatches":mism[:5],"distinct_words":len(set(lower)),
            "repeated_word":len(lower)!=len(set(lower)),"catalogue_match":cat,
            "sentence_shape":bool(re.search(r"[.!?]$",s.strip())) and len(words)>=8,
            "reader_eligible":bool(t) and not mism and len(t)>=100 and not cat and
                              len(lower)==len(set(lower)) and len(words)>=8}
def ask(p):
    r=subprocess.run(["ollama","run",MODEL,p],capture_output=True,text=True,timeout=35)
    return r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""
def main():
    known=json.loads((ROOT/"data/known_palindromes.json").read_text())
    catalogue=set(known) if isinstance(known,list) else set(known.keys())
    model_hash=hashlib.sha256(subprocess.check_output(["ollama","show",MODEL,"--modelfile"])).hexdigest()
    rows=[]
    for i in range(4):
        try: text=ask(PROMPT+f"\nAttempt {i+1}: make this distinct from all prior attempts."); err=None
        except Exception as e: text=""; err=f"{type(e).__name__}: {e}"
        a=audit(text,catalogue) if text else None
        rows.append({"id":f"attempt-{i+1}","prompt":PROMPT,"model":MODEL,"model_hash":model_hash,
                     "provenance":"local-ollama-direct-joint-authoring","attempt":a,"error":err})
    seed=next((r["attempt"]["text"] for r in rows if r["attempt"]),"")
    repaired=repair_operator(seed) if seed else ""
    repair_a=audit(repaired,catalogue) if repaired else None
    out={"experiment":"direct-joint-authoring-20260917","attempt_budget":4,
         "signature":"joint-center-out-prompt|no-half-reverse|independent-two-pointer|sha256|catalogue-and-word-repetition-rejection",
         "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"candidates":rows,
         "repair":{"operator":"mirror-left-onto-right-normalized-tape","source":"first-nonempty",
                   "reader_eligible":False,"audit":repair_a},
         "exact_count":sum(bool(r["attempt"] and r["attempt"]["reader_eligible"]) for r in rows)}
    OUT.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"attempts":4,"reader_eligible":out["exact_count"],"output":str(OUT)}))
if __name__ == "__main__": main()
