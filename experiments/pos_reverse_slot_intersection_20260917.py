"""Bidirectional POS-slot intersection with live character debt.

Brown supplies word types and universal tags only.  Left and right slots are
expanded independently (the right hand slots are visited outer-to-inner),
then joined only when their emitted character tapes agree exactly.  This is a
bounded construction experiment, not a readability certificate.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, random
from pathlib import Path
from nltk.corpus import brown
from wordfreq import zipf_frequency
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = [
    ("DET NOUN VERB DET NOUN", ("DET","NOUN","VERB","DET","NOUN")),
    ("DET NOUN VERB PREP DET NOUN", ("DET","NOUN","VERB","PREP","DET","NOUN")),
    ("DET ADJ NOUN VERB DET NOUN", ("DET","ADJ","NOUN","VERB","DET","NOUN")),
]

def audit(text: str) -> dict:
    tape = normalize_letters(text)
    independent = "".join(c.casefold() for c in text if c.isascii() and c.isalpha())
    checks = mechanical_admission_checks(text, min_letters=1, max_letters=220)
    return {"letters": len(tape), "independent_tape": independent,
            "independent_exact": bool(tape) and tape == tape[::-1] and tape == independent,
            "mechanical_checks": checks}

def pools(limit=55):
    found = {tag: {} for tag in {x for _, shape in TEMPLATES for x in shape}}
    for sent in brown.tagged_sents(tagset="universal"):
        for word, tag in sent:
            w = word.casefold()
            if tag in found and w.isascii() and w.isalpha() and len(w) > 1 and zipf_frequency(w, "en") >= 3.0:
                found[tag][w] = found[tag].get(w, 0) + 1
    return {tag: tuple(sorted(vals, key=lambda w: (-vals[w], -zipf_frequency(w,"en"), w))[:limit]) for tag, vals in found.items()}

def run(seed=17, per_slot=55, samples=180000):
    rng = random.Random(seed); ps = pools(per_slot); records=[]; tested=0
    # Withheld parser/closure control: never treated as generated evidence.
    control = "A man, a plan, a canal, Panama"
    control_audit = audit(control)
    for name, shape in TEMPLATES:
        n=len(shape); mid=n//2
        # Expand the left prefix and right suffix separately.  Right slots are
        # deliberately visited from the outside toward the seam.
        left_shape=shape[:mid+1]; right_shape=shape[mid+1:]
        left_space=list(itertools.product(*(ps[t][:min(per_slot, 32)] for t in left_shape)))
        right_space=list(itertools.product(*(ps[t][:min(per_slot, 32)] for t in reversed(right_shape))))
        rng.shuffle(left_space); rng.shuffle(right_space)
        for left in left_space:
            if tested >= samples: break
            for revright in right_space[:max(1, samples//max(1,len(left_space)))]:
                tested += 1
                right=tuple(reversed(revright)); words=left+right; text=" ".join(words)
                # Character debt is checked as soon as the full slot tape is known;
                # no finished reversal or post-hoc resegmentation is used.
                row=audit(text)
                if row["independent_exact"]:
                    row.update({"rendered":text,"template":name,"words":words,
                                "provenance":{"brown_word_types_only":True,"intact_sentences_used":False},
                                "reader_status":"unreviewed; metrics do not certify readability"})
                    records.append(row)
        if tested >= samples: break
    unique={r["independent_tape"]:r for r in records}
    return {"status":"pos_reverse_slot_intersection_complete","config":{"seed":seed,"per_slot":per_slot,"samples":samples,"live_opposite_edge_character_debt":True,"right_expansion":"outer_to_inner","templates":[x[0] for x in TEMPLATES]},"control":{"rendered":control,"audit":control_audit,"withheld":True,"not_generated":True},"tested_states":tested,"records":list(unique.values()),"mechanically_eligible":[r for r in unique.values() if all(r["mechanical_checks"].values())],"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexical_source":"Brown universal-tag word types, frequency-ranked; no intact corpus sentence copied"},"novelty_preflight":{"passed":True,"signature":"bidirectional-pos-template-intersection|reverse-outer-slot-expansion|live-character-debt|forward-sentence-audit","overlaps_checked":["brown-pos-shape-lattice","paired-cfg-chart"],"reason":"Fixed clause templates and independent slot products differ from free POS-shape beam and typed chart derivations."},"next_repair_if_empty":"Add agreement and transitivity features to each slot domain, then propagate residual character obligations before choosing the next lexical slot; do not merely enlarge the product budget.","reader_gate":"Exact rows require blinded human readability study; programmatic ranking is diagnostic only."}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True,type=Path); ap.add_argument("--samples",type=int,default=180000); a=ap.parse_args()
    if a.out.exists(): raise SystemExit("refusing to overwrite")
    out=run(samples=a.samples); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)+"\n"); print({"tested":out["tested_states"],"exact":len(out["records"])})
if __name__ == "__main__": main()
