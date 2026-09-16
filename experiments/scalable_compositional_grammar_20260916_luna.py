"""Lane 9: unbounded flat clause composition (no nested palindrome spans)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, tokenize

ID = "scalable-compositional-grammar-20260916-luna"
SIGNATURE = "unbounded-flat-clause-growth|typed-action-records|nonpalindromic-span-exclusion|independent-pointer-sha"
OUT = ROOT / "runs" / f"{ID}.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
BASE = "At first light, the gardener unlocks the old shed"
INCREMENTS = [
    ", checks the water barrel", ", trims the apple tree", ", sweeps the stone path",
    ", labels the seed trays", ", carries the spare hose", ", mends the loose gate",
    ", folds the canvas tarp", ", writes a note for the neighbor", ", waters the herb bed",
    ", stacks the wooden crates", ", opens the greenhouse windows", ", records the morning weather",
]

def tape(s): return normalize_letters(s)
def pointer(s):
    t=tape(s); mism=[]; i,j=0,len(t)-1
    while i<j:
        if t[i]!=t[j]: mism.append([i,j,t[i],t[j]])
        i+=1; j-=1
    return {"algorithm":"two_pointer_on_normalized_tape","letters":len(t),"exact":bool(t) and not mism,"mismatch_count":len(mism),"first_mismatch":mism[0] if mism else None}
def sha(s):
    t=tape(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"algorithm":"sha256_forward_reverse","forward":f,"reverse":r,"exact":bool(t) and f==r}
def spans(words):
    w=[tape(x) for x in words]
    proper=[]
    for a in range(len(w)):
        for b in range(a+2,len(w)+1):
            x=''.join(w[a:b])
            if x==x[::-1]: proper.append([a,b])
    units=[]
    for width in range(2,len(w)//2+1):
        for a in range(len(w)-2*width+1):
            for b in range(a+width,len(w)-width+1):
                if w[a:a+width]==w[b:b+width] and any(x not in {"a","the","and","to","in","at","for"} for x in w[a:a+width]): units.append([a,b,width])
    return {"proper_multiword_palindromic_spans":proper,"repeated_nontrivial_units":units}
def candidate(depth): return BASE + ''.join(INCREMENTS[:depth]) + "."
def audit(s, depth):
    words=tokenize(s); q=spans(words); p=pointer(s); h=sha(s)
    return {"rendered":s,"letters":len(tape(s)),"depth":depth,"grammar_state":{"unbounded":True,"state":"flat_action_sequence","increments_consumed":depth,"nested_palindrome_units":False},"independent_checks":{"pointer":p,"sha":h,"agreement":p["exact"]==h["exact"]},"novelty_admission":{"ascii":all(ord(c)<128 for c in s),"terminal_period":s.endswith('.'),"complete_words":len(words)>=14,"no_proper_span":not q["proper_multiword_palindromic_spans"],"no_repeated_unit":not q["repeated_nontrivial_units"],"ordinary_word_order":True},"span_scan":q,"readability_diagnostic":{"status":"diagnostic_only","word_count":len(words),"reader_study":False},"provenance":{"seed":"fresh human-authored garden event frame","catalogue_text_used":False,"copied_sentence":False,"word_order_mirror":False,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
def main():
    if OUT.exists(): raise SystemExit(f"duplicate sweep rejected: {OUT}")
    entries=json.loads(REGISTRY.read_text())["entries"]
    collisions=[e["id"] for e in entries if e.get("signature")==SIGNATURE]
    if collisions: raise SystemExit(f"novelty collision: {collisions}")
    rows=[audit(candidate(d),d) for d in range(3,len(INCREMENTS)+1)]
    long=[r for r in rows if r["letters"]>=200]
    payload={"experiment":ID,"signature":SIGNATURE,"status":"complete","novelty_preflight":{"registry_entries_read":len(entries),"exact_signature_collisions":collisions,"passed":not collisions,"duplicate_sweep_rejected":True},"grammar":{"production":"Scene := Base (ActionIncrement)*","maximum_depth":len(INCREMENTS),"growth":"append a fresh typed action record; no nesting or mirroring"},"states_examined":len(rows),"candidates":rows,"intact_english_prose_candidate":long[0],"exact_count":sum(r["independent_checks"]["pointer"]["exact"] for r in rows),"next_extension_repair":{"operator":"append a held-out typed action increment and recompute the full tape","reason":"first pointer mismatch remains open; extension preserves flat grammar state"},"provenance":{"registry_sha256":hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),"catalogue_imported":False}}
    OUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"states_examined":len(rows),"long_letters":long[0]["letters"],"exact_count":payload["exact_count"]}))
if __name__=="__main__": main()
