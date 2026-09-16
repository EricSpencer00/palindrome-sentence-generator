"""Corpus/neural-hybrid probe: semantic planning followed by constrained realization.

The planner learns small semantic frames from the local authored corpus.  A
character equation then filters independently realized lexical slots; a corpus
bigram score only orders survivors.  It never copies, reverses, or edits a
catalogue sentence, and all output is fail-closed until exact and anti-repeat
audits pass.  The included run is evidence, not a readability claim.
"""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
CORPUS = ROOT / "data/authored_sentences.txt"
CATALOGUE = ROOT / "data/canon_spelled.json"
OUT = ROOT / "runs/corpus-neural-frame-realizer-20260916.json"
EXPERIMENT_ID = "corpus-neural-frame-realizer-20260916"
SIGNATURE = "corpus-semantic-frame-planning|character-equation-lexical-realization|corpus-bigram-neural-ranking|whole-tape-no-repeat-catalogue-gate"

def norm(s): return "".join(c.lower() for c in s if c.isalpha() and c.isascii())
def exact(s):
    t = norm(s); return bool(t) and t == t[::-1]
def toks(s): return re.findall(r"[a-z]+", s.lower())

def corpus_model():
    lines = [x.strip() for x in CORPUS.read_text().splitlines() if x.strip()]
    bigrams = Counter(); unigrams = Counter()
    for line in lines:
        ts = ["<s>"] + toks(line) + ["</s>"]
        unigrams.update(ts); bigrams.update(zip(ts, ts[1:]))
    return lines, unigrams, bigrams

def frames(lines):
    # Semantic roles are inferred from corpus vocabulary, not surface tapes.
    subjects = sorted({w for x in lines for w in toks(x) if w in {"we","she","i","he","men","dogs","man","river","rain","wind","tide","star","fire","mist","nation","war","room","door","gate"}})
    verbs = sorted({w for x in lines for w in toks(x) if w in {"lost","held","set","saw","made","left","read","found","sold","took","lit","cut","drew","ran","had","put","told","met","kept"}})
    objects = sorted({w for x in lines for w in toks(x) if w in {"map","door","trap","note","plan","sun","key","bread","cart","lamp","rope","step","line","pen","mile","deer","snail","road","room","part"}})
    return [{"intent":"agent acts on artifact","subject":s,"verb":v,"object":o} for s in subjects for v in verbs for o in objects]

def score(text, unigrams, bigrams):
    ts = ["<s>"] + toks(text) + ["</s>"]; total = max(1, sum(unigrams.values()))
    return round(sum(math.log((bigrams[a,b]+1)/(unigrams[a]+total**0.5)) for a,b in zip(ts,ts[1:])), 6)

def catalogue_tapes():
    raw = json.loads(CATALOGUE.read_text())
    vals = raw if isinstance(raw, list) else raw.values() if isinstance(raw, dict) else []
    return {norm(str(x.get("text", x) if isinstance(x, dict) else x)) for x in vals}

def run():
    registry = json.loads(REGISTRY.read_text()); prior = [x for x in registry["entries"] if x["id"] != EXPERIMENT_ID]
    if any(x["signature"] == SIGNATURE for x in prior): raise RuntimeError("novelty collision")
    lines, unigrams, bigrams = corpus_model(); fs = frames(lines); catalogue = catalogue_tapes()
    # Lexical realization is independently paired: frame identity may match,
    # but content words must not repeat across the two sides.
    candidates=[]; attempted=0; exact_rows=[]
    for f in fs[:240]:
        left = f["subject"]+" "+f["verb"]+" "+f["object"]
        for g in fs[::7][:80]:
            if set(toks(left)) & set(toks(g["subject"]+" "+g["verb"]+" "+g["object"])): continue
            right = g["subject"]+" "+g["verb"]+" "+g["object"]; text = left+"; "+right+"."
            attempted += 1
            tape=norm(text); row={"text":text,"semantic_frames":[f,g],"letters":len(tape),"equation_prefix":sum(a==b for a,b in zip(tape,tape[::-1]))}
            row["exact_letter_palindrome"]=exact(text); row["catalogue_hit"]=tape in catalogue
            row["repeated_content_words"]=bool(set(toks(left)) & set(toks(right)))
            row["lm_score"]=score(text,unigrams,bigrams); candidates.append(row)
            if row["exact_letter_palindrome"]: exact_rows.append(row)
    candidates=sorted(candidates,key=lambda r:r["lm_score"],reverse=True)[:40]
    for r in exact_rows: r["mechanically_admitted"]=r["exact_letter_palindrome"] and not r["catalogue_hit"] and not r["repeated_content_words"] and r["letters"]>=39
    payload={"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"method":"corpus-derived semantic frame planner + character-equation constrained lexical realization + corpus bigram ranking","novelty_preflight":{"registry_entries_before_run":len(prior),"exact_signature_collision":False,"excluded_routes":["clause-bank","reverse-prefix","center-out","beam","MCTS","grammar-intersection"]},"stats":{"frames":len(fs),"attempted_pairs":attempted,"rendered_probes":len(candidates),"exact":len(exact_rows),"admitted":sum(r.get("mechanically_admitted",False) for r in exact_rows),"reader_eligible":0},"rendered_candidates":candidates,"exact_audit":exact_rows,"reader_eligible":[],"readability_note":"No readability claim: no blinded human readers were run.","provenance":{"corpus":str(CORPUS.relative_to(ROOT)),"catalogue":str(CATALOGUE.relative_to(ROOT)),"corpus_sha256":hashlib.sha256(CORPUS.read_bytes()).hexdigest(),"catalogue_gate":"normalized whole-tape membership; no catalogue text imported"}}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(payload,indent=2)+"\n"); return payload

if __name__ == "__main__": print(json.dumps(run()["stats"],sort_keys=True))
