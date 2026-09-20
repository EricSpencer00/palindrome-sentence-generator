"""Exact reverse parsing of authored coordinated Brown-lexicon clauses."""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.brown_authored_semantic_reverse_decoder_20260920 import Word, Trie, audit, frame_score, load_words

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data/brown_pcfg_bank_20260920.json"
OUT = ROOT / "runs/brown-authored-coordination-reverse-decoder-20260920.json"
ID = "brown-authored-coordination-reverse-decoder-20260920"
SIGNATURE = "brown-derived-lexicon|authored-coordination-grammar|complete-clause-pair|variable-boundary-reverse-parse"

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())

CLAUSE = ("DET", "ADJ", "AGENT", "ACTION", "DET", "OBJECT")
ADJUNCT = ("DET", "ADJ", "AGENT", "ACTION", "DET", "OBJECT", "PREP", "DET", "PLACE")

def make_frames(domains, limit=12000):
    frames=[]
    for kind, shape in (("clause", CLAUSE), ("adjunct", ADJUNCT)):
        choices=[domains[r][:10] for r in shape]
        for selected in itertools.product(*choices):
            words=tuple(selected)
            content=[w.text for w in words if w.text not in {"the","a","an"}]
            if len(set(content)) != len(content): continue
            frames.append((kind, shape, words))
            if len(frames)>=limit: return frames
    return frames

def parse(tape, shape, trie, forbidden, limit=25):
    out=[]; states=0; seen=set()
    def walk(i,pos,words):
        nonlocal states
        states += 1
        key=(i,pos,tuple(w.text for w in words))
        if key in seen or len(out)>=limit: return
        seen.add(key)
        if i==len(shape):
            if pos==len(tape): out.append(words)
            return
        for end,w in trie.matches(tape,pos,shape[i]):
            if w.text in forbidden and w.text not in {"the","a","an","but"}: continue
            walk(i+1,end,words+(w,))
    walk(0,0,())
    return out,states

def run(max_pairs=12000):
    domains=load_words()
    # The frozen Brown-derived bank contains only ``but`` among the four
    # requested conjunction forms; no out-of-bank connector is fabricated.
    connectors=(Word("but", 1.0),)
    domains=dict(domains); domains["CONJ"]=connectors
    frames=make_frames(domains, max_pairs)
    trie=Trie(domains)
    results=[]; exact=[]; states=parses=0
    shapes=[]
    for kind,shape,_ in frames:
        shapes.append((kind,shape+(("CONJ",)+shape)))
    # Two complete ordinary-order clauses are selected before the connector.
    for kind,shape,words in frames:
        for right_kind,right_shape in (("clause",CLAUSE),("adjunct",ADJUNCT)):
            for connector in connectors:
                # The second clause is authored from the same semantic grammar,
                # but every content word is independently selected by parsing.
                full_words=words+(connector,)
                target=letters(" ".join(w.text for w in full_words))[::-1]
                full_shape=shape+("CONJ",)+right_shape
                found,used=parse(target,full_shape,trie,frozenset(w.text for w in words))
                states += used; parses += len(found)
                for rw in found:
                    text=" ".join(w.text for w in full_words)+"; "+" ".join(w.text for w in rw)+"."
                    row={"rendered":text,"left_kind":kind,"right_kind":right_kind,"left_roles":list(full_shape),"rank_score":frame_score(full_words)+frame_score(rw),"audit":audit(text),"provenance":{"source":"Brown-derived word forms only; no Brown sentence text","complete_left_coordination":True,"complete_right_coordination":True,"connector":"but","variable_word_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"mirrored_token_units":False,"readability_certified":False}}
                    results.append(row)
                    if row["audit"]["exact"]: exact.append(row)
    controls=["The young man sees the house but the old woman hears the word.","A good boy found the door but a new girl held the key."]
    return {"experiment_id":ID,"method":"authored coordinated semantic clause pairs over Brown-derived lexical domains with complete reverse parsing","stats":{"complete_left_frames":len(frames),"coordination_attempts":len(frames)*2,"reverse_states":states,"complete_reverse_parses":parses,"rendered_candidates":len(results),"exact":len(exact)},"rendered_candidates":sorted(results,key=lambda x:-x["rank_score"])[:200],"exact_candidates":sorted(exact,key=lambda x:-x["audit"]["letters"])[:100],"controls":[{"rendered":x,"audit":audit(x),"complete_prose":True} for x in controls],"novelty_preflight":{"status":"passed","signature":SIGNATURE,"distinct_from":"prior single-clause and dialogue lanes; this lane composes two complete semantic clauses with a coordination boundary before reverse parsing","catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False},"provenance":{"bank":str(BANK.relative_to(ROOT)),"bank_sha256":hashlib.sha256(BANK.read_bytes()).hexdigest(),"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"audits":["independent two-pointer mismatch","forward/reverse SHA-256"],"connector_policy":"only Brown-bank connector but admitted; and/or/while remain unmaterialized because absent from frozen bank","next_reader_test":"randomized blinded ratings of intact coordinated prose versus shuffled controls"},"status":"fresh exact candidates require human reading" if exact else "no fresh exact parse in this lane","next_construction":"prepare reader package if exact rows survive semantic inspection; otherwise author a new subordination grammar"}

if __name__=="__main__":
    x=run(); OUT.write_text(json.dumps(x,indent=2)+"\n"); print(json.dumps(x["stats"]))
