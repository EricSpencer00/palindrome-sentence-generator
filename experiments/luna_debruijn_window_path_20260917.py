"""Preflight and bounded probe for paragraph-level overlap paths.

Sentence windows are independently authored prose; adjacent windows overlap on
an exact character k-gram as in a de Bruijn path.  This is deliberately not a
clause/scene/CFG search.  The run is excluded when the registry shows the same
online character-obligation family already covered by prior lanes.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "docs/experiment-novelty-registry.json"
ID = "luna-debruijn-window-path-20260917-excluded"
SIG = "paragraph-window-overlap-path|debruijn-kgram-edges|finite-residue-automaton|semantic-continuity-labels|independent-pointer-sha"

WINDOWS = [
    ("dawn", "At dawn, Mira opened the archive and marked the flooded pier."),
    ("pier", "The flooded pier held a lantern, and the harbor keeper carried it inland."),
    ("inland", "Inland, the keeper found dry paper and wrote the tide's new measure."),
]

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    x = letters(text)
    return {"length": len(x), "exact": x == x[::-1], "left_right_mismatches": sum(a != b for a,b in zip(x, x[::-1]))}

def main() -> None:
    reg = json.loads(REG.read_text())
    entries = reg["entries"]
    # Preflight happens before this route's own entry is considered.
    atoms = set(re.split(r"[^a-z0-9]+", SIG)) - {"", "a", "and", "of", "the", "with"}
    overlaps = []
    for e in entries:
        other = set(re.split(r"[^a-z0-9]+", e["signature"])) - {"", "a", "and", "of", "the", "with"}
        if len(atoms & other) / max(1, len(atoms | other)) >= .20:
            overlaps.append(e["id"])
    rows=[]
    for i,(label,text) in enumerate(WINDOWS):
        rows.append({"label":label,"prose":text,"continuity_label": label,
                     "audit":audit(text),"sha256":hashlib.sha256(text.encode()).hexdigest()})
    k=6; edges=[]
    for a,b in zip(WINDOWS, WINDOWS[1:]):
        la,lb=letters(a[1]),letters(b[1]); suffix=la[-k:]; prefix=lb[:k]
        edges.append({"from":a[0],"to":b[0],"k":k,"suffix":suffix,"prefix":prefix,"overlap":suffix==prefix})
    payload={"experiment":ID,"novelty_preflight":{"registry_entries_read_before_run":len(entries),"signature":SIG,"overlaps":overlaps,"passed":False,"reason":"overlap with registered online character-obligation and semantic-path lanes; paragraph windows do not establish a new state-space dimension"},"windows":rows,"edges":edges,"paragraph":" ".join(x[1] for x in WINDOWS),"audit":audit(" ".join(x[1] for x in WINDOWS)),"residue_automaton":{"states":len(WINDOWS),"residue_modulus":k,"closure":False,"reason":"path ends at inland; no return edge to dawn and no mirrored residue closure"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independently_authored_windows":True,"catalogue_imported":False,"source_sentences_copied":False,"repeated_units":False,"exact_audit":"normalized two-pointer"},"next_repair":"Do not promote this lane. If revisited, add a genuinely new continuity state (temporal or causal) and a held-out return edge before re-running registry preflight."}
    out=ROOT/"runs"/(ID+".json"); out.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))

if __name__ == "__main__": main()
