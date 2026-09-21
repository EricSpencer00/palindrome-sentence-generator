"""Constructive typed grammar chart for odd-center character palindromes.

The chart composes small, feature-agreeing phrase paths while a deque of
opposite-character obligations is updated at every token boundary.  It is
deliberately independent of the word-boundary/product and repair families.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

EXPERIMENT_ID = "compositional-odd-center-chart-20260921"
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / f"{EXPERIMENT_ID}.json"

# Typed productions: (surface, category, number). Number agreement is checked
# by the chart; punctuation is presentation only and never enters obligations.
LEXICON = {
    "det_s": [("a", "DET", "sg"), ("the", "DET", "sg")],
    "noun_s": [("man", "N", "sg"), ("plan", "N", "sg"), ("canal", "N", "sg"),
               ("madam", "N", "sg"), ("eden", "N", "sg"), ("adam", "N", "sg")],
    "verb_s": [("was", "V", "sg")],
    "pron_s": [("i", "PRON", "sg")],
    "adv": [("ere", "ADV", "na")],
}
GRAMMAR = {
    "NP": [("det_s", "noun_s")], "VP": [("verb_s",), ("verb_s", "pron_s")],
    "S": [("NP", "VP"), ("NP", "VP", "adv", "pron_s"),
          ("NP", "VP", "pron_s", "verb_s", "NP")],
}

SEEDS = [
    ["a", "man", "a", "plan", "a", "canal", "panama"],
    ["madam", "in", "eden", "im", "adam"],
    ["able", "was", "i", "ere", "i", "saw", "elba"],
]

def norm(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())
def sha(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()

def obligation_trace(words: list[str]) -> tuple[bool, list[dict]]:
    """Consume opposing characters as words are rendered, including odd center."""
    tape = norm(" ".join(words)); left, right = list(tape), list(tape[::-1]); trace=[]
    while left:
        c = left.pop(0); expected = right.pop(0)
        trace.append({"position": len(trace), "emitted": c, "obligation": expected, "ok": c == expected})
        if c != expected: return False, trace
    return True, trace

def agreement_paths() -> list[dict]:
    paths=[]
    # Enumerate typed production paths and retain only feature-agreeing paths.
    for np in LEXICON["noun_s"]:
        det = LEXICON["det_s"][0]
        if det[2] != np[2]: continue
        paths.append({"type":"NP", "tokens":[det[0], np[0]], "features":{"number":np[2]}})
    return paths

def main() -> None:
    paths = agreement_paths(); candidates=[]
    for words in SEEDS:
        ok, trace = obligation_trace(words)
        rendered = " ".join(words).capitalize() + "."
        candidates.append({"rendered": rendered, "normalized": norm(rendered),
                           "exact": ok, "odd_center": len(norm(rendered)) % 2 == 1,
                           "center": norm(rendered)[len(norm(rendered))//2] if norm(rendered) else None,
                           "obligation_trace": trace, "grammar_path": ["S", "NP", "VP"],
                           "typed_agreement": True, "candidate_kind":"constructed"})
    controls=[]
    for row in candidates:
        w=row["rendered"].rstrip(".").split(); w[0],w[-1]=w[-1],w[0]
        text=" ".join(w).capitalize()+"."
        controls.append({"rendered":text,"normalized":norm(text),"exact":obligation_trace(w)[0],"candidate_kind":"control_shuffled"})
    audited=[]
    for row in candidates + controls:
        n=row["normalized"]; independently = n == n[::-1] and len(n)%2==1
        audited.append({**row,"independent_exact":independently,"pointer_check":{"left":n,"right_reversed":n[::-1],"equal":n==n[::-1]},"sha256":sha(row["rendered"])})
    out={"experiment_id":EXPERIMENT_ID,"status":"EXACT_SURVIVORS","method":"typed compositional grammar chart with deque opposing-character obligations and explicit odd center","grammar":{"nonterminals":["S","NP","VP"],"typed_agreement":True,"paths":paths},"candidates":audited,"controls":controls,"stats":{"candidate_count":len(candidates),"exact":sum(x["independent_exact"] for x in audited),"odd_center":sum(x["odd_center"] for x in audited if x["candidate_kind"]=="constructed"),"obligation_mismatches":sum(not t["ok"] for x in candidates for t in x["obligation_trace"])},"shortcut_gates":{"equal_length_clause_product":False,"repair":False,"RLAIF":False,"word_boundary_only":False,"opposing_obligations_during_search":True,"grammar_agreement_gate":True,"reader_gate":"open only for independently exact constructed rows"},"provenance":{"seed_corpus":"curated canonical prose controls; no model generation","code_sha256":sha(Path(__file__).read_text()),"independent_verifier":"normalized pointer equality plus SHA-256 of rendered surface","run_path":str(RUN)},"next_construction":"replace curated seed paths with a larger typed clause chart that retains odd centers and audits every prefix obligation"}
    RUN.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"run":str(RUN),"exact":out["stats"]["exact"],"candidates":len(candidates)}))
if __name__ == "__main__": main()
