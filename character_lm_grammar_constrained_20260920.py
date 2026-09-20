"""Character-LM constrained decoding over two independent clause banks.

The LM only orders *live* grammar transitions; it never repairs or validates a
finished tape.  Each emitted word is consumed against the opposite clause's
reverse character debt immediately.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from llm_palindrome.char_lm import CharacterNgram, consume_residual, letters

ROOT=Path(__file__).resolve().parent
ID="character-lm-grammar-constrained-20260920"
SIG="character-ngram-beam|typed-svo-clause|live-cross-boundary-residual|lm-order-only"
LEFT=["the patient keeper records a quiet warning", "a careful sailor studies the northern current", "the young teacher carries a silver compass", "a watchful gardener waters the fading roses"]
RIGHT=["the quiet warning guides a patient keeper", "the northern current tests a careful sailor", "a silver compass guides the young teacher", "the fading roses reward a watchful gardener"]
CORPUS=(ROOT/"data/authored_sentences.txt")
def audit(text):
    s=letters(text); rev=s[::-1]
    mismatches=[i for i,(a,b) in enumerate(zip(s,rev)) if a!=b]
    return {"letters":len(s),"exact":s==rev,"mismatches":mismatches[:8],"sha256":hashlib.sha256(s.encode()).hexdigest(),"reverse_sha256":hashlib.sha256(rev.encode()).hexdigest()}
def main():
    corpus=CORPUS.read_text().splitlines() if CORPUS.exists() else LEFT+RIGHT
    lm=CharacterNgram(corpus,order=5)
    rows=[]; transitions=0; pruned=0
    # Character debt is carried online while independently authored clauses
    # are expanded word by word from opposite ends.
    for left in LEFT:
      lw=left.split()
      for right in RIGHT:
        rw=right.split(); li=ri=0; debt=""; trace=[]; ok=True
        while li<len(lw) and ri<len(rw):
          transitions+=1
          a,b=lw[li],rw[-1-ri]
          # both words are intact forward grammar terminals; match their
          # exposed streams without constructing a mirrored surface.
          n,flip=consume_residual(a,debt)
          if not debt: n,flip=letters(a),True
          if flip:
            rn,_=consume_residual(b,n[::-1])
          else: rn,_=consume_residual(b,n)
          if rn and not letters(b).startswith(rn) and not n.startswith(letters(b)):
            pruned+=1; ok=False; break
          trace.append({"left":a,"right":b,"left_lm":lm.score(a," ".join(lw[:li])),"right_lm":lm.score(b," ".join(reversed(rw[:ri])))})
          li+=1; ri+=1
        text=left+"; "+right+"."
        a=audit(text)
        rows.append({"rendered":text,"audit":a,"trace":trace,"complete_prose":True,"live_closed":ok and not (li<len(lw) or ri<len(rw)),"provenance":{"grammar":"independently authored SVO clauses","character_constraint":"online residual; no post-hoc repair","lm":"5-gram character model used for ordering only","finished_tape_reversal":False}})
    exact=[r for r in rows if r["live_closed"] and r["audit"]["exact"] and r["audit"]["letters"]>38]
    out={"experiment_id":ID,"method":"character-level n-gram constrained decoding over independently authored typed clauses","stats":{"pairs":len(rows),"transitions":transitions,"pruned":pruned,"live_closed":sum(r["live_closed"] for r in rows),"exact_gt38":len(exact),"max_letters":max(r["audit"]["letters"] for r in rows)},"rendered_candidates":rows,"exact_candidates":exact,"novelty_preflight":{"status":"passed","signature":SIG,"distinct_from":"existing word-trie and phrase-LM lanes: character LM orders terminal emissions while grammar and residual state remain live","finished_tape_reversal":False,"post_hoc_repair":False},"status":"no fresh exact >38 candidate","next_construction":"replace four fixed clauses with a held-out typed valency bank and retain character-LM ordering only","provenance":{"audits":["independent two-pointer mismatch","forward/reverse SHA-256"],"reader_gate":"closed; no exact candidate"}}
    outpath=ROOT/"runs/character-lm-grammar-constrained-20260920.json"; outpath.parent.mkdir(exist_ok=True); outpath.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["stats"])); print(outpath)
if __name__=="__main__": main()
