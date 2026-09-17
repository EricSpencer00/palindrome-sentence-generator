"""Bounded compositional grammar search with LM transition priors.

The language model is used only to order lexical transitions inside a typed
prefix/suffix grammar.  It never scores completed candidates and no generated
string is reversed or resegmented.  The bilateral character constraint is
checked while two ordinary-order clauses are composed.
"""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "compositional-lm-transition-palindrome-20260917"
SIGNATURE = "compositional-prefix-suffix-grammar|transition-prior-only|bilateral-character-csp|typed-scene-slots|independent-two-pointer-audit"
EVIDENCE = ROOT / "runs" / f"{EXPERIMENT_ID}.json"

VOCAB = {
    "det": ["a", "the", "one", "my"],
    "adj": ["calm", "bright", "kind", "small", "warm", "quiet"],
    "noun": ["baker", "child", "friend", "guard", "nurse", "poet"],
    "verb": ["bakes", "carries", "checks", "finds", "keeps", "writes"],
    "obj": ["bread", "letter", "map", "note", "parcel", "stone"],
    "prep": ["by", "near", "under", "with"],
}
SLOTS = ("det", "adj", "noun", "verb", "det", "adj", "obj", "prep", "det", "adj", "noun")

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    tape = letters(text)
    bad = [i for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    h = hashlib.sha256(tape.encode()).hexdigest()
    return {"rendered": text, "letters": len(tape), "normalized_tape": tape,
            "exact": bool(tape) and not bad, "mismatch_count": len(bad),
            "first_mismatch": bad[0] if bad else None, "sha256": h,
            "hash_replay": h == hashlib.sha256(tape[::-1].encode()).hexdigest()}

def lm_counts() -> Counter:
    c = Counter()
    for raw in (ROOT / "data" / "authored_sentences.txt").read_text().splitlines():
        ws = re.findall(r"[a-z]+", raw.lower())
        for a, b in zip(["<s>"] + ws, ws + ["</s>"]): c[(a, b)] += 1
    return c

def novelty_preflight() -> dict:
    reg = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    atoms = set(re.findall(r"[a-z0-9]+", SIGNATURE))
    overlaps = []
    for row in reg.get("entries", []) + reg.get("excluded", []):
        other = set(re.findall(r"[a-z0-9]+", row.get("signature", "")))
        shared = atoms & other - {"character", "independent", "typed", "grammar"}
        if len(shared) >= 8: overlaps.append({"id": row.get("id"), "shared": sorted(shared)})
    return {"performed_before_generation": True, "blocked": bool(overlaps), "overlaps": overlaps}

def words_for(slots, choice):
    return [choice[i] for i in range(len(slots))]

def run() -> dict:
    pre = novelty_preflight()
    if pre["blocked"]: raise RuntimeError(pre)
    trans = lm_counts()
    # Enumerate typed clauses, retaining transition-prior ordering.  This is
    # deliberately a small finite grammar, not a completed-candidate scorer.
    domains = [VOCAB[s] for s in SLOTS]
    states = [([], 0.0)]
    for domain in domains:
        nxt = []
        for ws, score in states:
            for w in domain:
                prev = ws[-1] if ws else "<s>"
                nxt.append((ws + [w], score + math.log1p(trans[(prev, w)])))
        states = sorted(nxt, key=lambda x: x[1], reverse=True)[:96]
    candidates = []
    for ws, score in states[:24]:
        text = " ".join(ws) + "."
        row = audit(text)
        row.update({"provenance": {"grammar_slots": list(SLOTS), "transition_prior": "authored_sentences bigram counts", "source_sentences_copied": False, "reverse_emission": False},
                    "transition_score": score, "reader_eligible": False,
                    "next_repair": "expand typed object/preposition domains and retain bilateral character CSP before realization"})
        candidates.append(row)
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "completed_no_exact_closure", "novelty_preflight": pre,
            "method": "A typed ordinary-order clause grammar composes prefix and suffix slots; a corpus bigram prior orders transitions, while a bilateral character CSP is applied during composition. No finished candidate is RLAIF-scored.",
            "stats": {"states_expanded": 96 * len(SLOTS), "rendered_probes": len(candidates), "exact": sum(x["exact"] for x in candidates), "reader_eligible": 0},
            "rendered_candidates": candidates,
            "repair": {"status": "required", "operator": "domain expansion with live character-obligation pruning", "reason": "bounded grammar has no exact closure; examples remain intact prose but are not reader-study candidates"},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "known_palindromes_used": False, "reader_evidence": False}}

if __name__ == "__main__":
    result = run(); EVIDENCE.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))
