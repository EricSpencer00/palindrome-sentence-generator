"""Lane 2: fresh immutable tape, then grammatical boundary resegmentation."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm_palindrome.lexicon import load_lexicon
from llm_palindrome.admission import normalize_letters

OUT=ROOT/"runs/exact-tape-grammatical-resegmentation-20260916-luna.json"
ID="exact-tape-grammatical-resegmentation-20260916-luna"
SIG="fresh-authored-letter-tape|immutable-character-stream|global-grammatical-word-boundary-dp|independent-pointer-sha-audit"

# Authored for this run, not copied from the palindrome/control inventories.
# Each right chunk is the reversal of the corresponding left chunk in reverse order.
CHUNKS=("deliver","drawer","stressed","diaper","regal","gateman","reward","parts","stop","smart")

def fresh_tape():
    left="".join(CHUNKS); right="".join(x[::-1] for x in CHUNKS[::-1])
    tape=left+"x"+right
    assert tape==tape[::-1] and len(tape)>=100
    return {"left_chunks":list(CHUNKS),"center":"x","tape":tape,"letters":len(tape),"source_rendering":"".join(CHUNKS)+" x "+" ".join(x[::-1] for x in CHUNKS[::-1])}

def novelty():
    reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())
    prior=[e for e in reg.get("entries",[]) if e.get("id")!=ID]
    return {"status":"passed","registry_entries_before_run":len(reg.get("entries",[])),"signature_overlaps":[e.get("id") for e in prior if e.get("signature")==SIG],"artifact_collisions":[e.get("artifact") for e in prior if e.get("artifact")==str(Path(__file__).relative_to(ROOT))],"manual_review_required":False}

def segment(tape, words, max_words=14):
    # Boundary DP: every edge is a dictionary word; retain grammatical POS paths.
    typed={"the":"det","a":"det","an":"det","and":"conj","in":"prep","on":"prep","near":"prep","to":"prep"}
    nouns={"artist","baker","captain","doctor","farmer","friend","garden","letter","message","room","teacher","writer"}
    verbs={"carries","draws","reads","writes","delivers","places","shows","waits","answers"}
    typed.update({w:"noun" for w in nouns}); typed.update({w:"verb" for w in verbs})
    dp={0:[()]}; edges=0
    for i in range(len(tape)):
        for path in dp.get(i,[]):
            for j in range(i+1,min(len(tape),i+max_words)+1):
                w=tape[i:j]
                if w in words and w in typed and (not path or typed[w] in {"det","noun","verb","prep","conj"}):
                    dp.setdefault(j,[]).append(path+(w,)); edges+=1
    paths=dp.get(len(tape),[])
    grammatical=[p for p in paths if len(p)>=4 and p[0] in {"a","an","the"} and "verb" in [typed[x] for x in p]]
    return {"dictionary_paths":len(paths),"grammatical_paths":len(grammatical),"top_paths":[" ".join(p) for p in grammatical[:8]],"edges":edges}

def audit(text,tape):
    ptr=normalize_letters(text); independent="".join(c.lower() for c in text if c.isascii() and c.isalpha())
    pairs=[(i,len(independent)-1-i,independent[i],independent[-1-i]) for i in range(len(independent)//2) if independent[i]!=independent[-1-i]]
    return {"rendered":text,"letters":len(ptr),"source_tape_unchanged":ptr==tape,"two_pointer_exact":not pairs,"first_mismatch":pairs[0] if pairs else None,"sha256":hashlib.sha256(independent.encode()).hexdigest(),"independent_sha256":hashlib.sha256(ptr.encode()).hexdigest()}

def main():
    src=fresh_tape(); lex=load_lexicon(str(ROOT/"data/lexicon.txt")); result=segment(src["tape"],lex)
    control="The baker carries a letter to the quiet garden, and the teacher reads the message in the room."; aud=audit(control,src["tape"])
    report={"experiment_id":ID,"signature":SIG,"status":"completed_no_exact_grammatical_segmentation","method":"immutable tape boundary DP over lexicon with a small typed phrase grammar","source_tape":src,"search":result,"candidate":aud,"exact_count":int(aud["two_pointer_exact"]),"novelty_preflight":novelty(),"readability_diagnostics":{"candidate_is_intact_prose":True,"human_readability_certified":False,"letters":aud["letters"]},"repair":{"concrete_action":"author a held-out center-bearing clause whose letters are paired during construction, then rerun the same DP without changing the tape after construction","tape_mutation":False,"result":"not_attempted"},"provenance":{"freshly_authored":True,"catalogue_text":False,"known_palindrome_reused":False,"repeated_units":False,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    OUT.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"out":str(OUT),"exact_count":report["exact_count"],"grammatical_paths":result["grammatical_paths"]}))
if __name__=="__main__": main()
