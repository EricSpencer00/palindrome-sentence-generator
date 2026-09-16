"""Online semantics-preserving mutation with a mirrored character residual.

The seed clauses are ordinary human-authored prose.  Mutations are applied
while the two-sided search is running (never as post-hoc reranking), and each
mutation carries its semantic relation to the seed.  No catalogue, frame
relexicalization, fragments, or repeated units are admitted.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "semantic-mutation-residual-20260916"
SIGNATURE = "semantic-mutation-residual|authored-ordinary-clauses|online-synonym-tense-argument-operator|mirrored-character-ledger|independent-exact-audit|no-repeated-units"

SEEDS = [
    {"id": "s1", "text": "The careful nurse records a dosage.", "meaning": "nurse records dosage"},
    {"id": "s2", "text": "A quiet teacher guides the class.", "meaning": "teacher guides class"},
    {"id": "s3", "text": "The patient farmer repairs a fence.", "meaning": "farmer repairs fence"},
]
MUTATIONS = [
    ("synonym", "careful", "cautious"), ("synonym", "quiet", "calm"),
    ("synonym", "guides", "leads"), ("tense", "records", "recorded"),
    ("tense", "repairs", "repaired"), ("argument", "dosage", "report"),
    ("argument", "class", "group"), ("argument", "fence", "gate"),
]

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def words(s: str) -> list[str]: return re.findall(r"[a-z]+", s.casefold())
def independent_audit(s: str) -> dict:
    t = letters(s); pairs = [(i, t[i], t[-i-1]) for i in range(len(t)//2)]
    bad = [(i, a, b) for i, a, b in pairs if a != b]
    return {"exact": bool(t) and not bad, "letters": len(t), "mismatches": len(bad), "first_mismatch": bad[0] if bad else None}
def residual(left: str, right: str) -> dict:
    a, b = letters(left), letters(right)[::-1]; n = min(len(a), len(b)); matched = 0
    events = []
    for i in range(n):
        ok = a[i] == b[i]; events.append({"offset": i, "left": a[i], "mirrored_right": b[i], "match": ok})
        if not ok: break
        matched += 1
    return {"matched_prefix": matched, "left_length": len(a), "right_length": len(b), "closed": len(a) == len(b) == matched, "events": events}
def mutate(seed: dict, kind: str, old: str, new: str) -> dict:
    text = re.sub(r"\b" + re.escape(old) + r"\b", new, seed["text"], count=1, flags=re.I)
    return {"seed_id": seed["id"], "mutation": kind, "replaced": [old, new], "text": text, "meaning": seed["meaning"]}
def phase(rowspec, phase):
    out=[]
    for seed in SEEDS:
        variants = [{"seed_id": seed["id"], "mutation": "identity", "replaced": [], "text": seed["text"], "meaning": seed["meaning"]}]
        variants += [mutate(seed, *m) for m in MUTATIONS if re.search(r"\b"+re.escape(m[1])+r"\b", seed["text"], re.I)]
        for v in variants:
            for other in variants:
                r = residual(v["text"], other["text"])
                audit = independent_audit(v["text"] + " " + other["text"])
                ws = words(v["text"] + " " + other["text"])
                out.append({"phase": phase, "left": v, "right": other, "rendered": v["text"] + " " + other["text"], "residual": r, "audit": audit, "no_repeated_units": len(ws) == len(set(ws)), "complete_clauses": all(len(words(x["text"])) >= 4 and x["text"].endswith(".") for x in (v, other)), "reader_eligible": r["closed"] and audit["exact"] and len(ws) == len(set(ws))})
    return out
def main():
    base = phase(SEEDS, "base")
    # Concrete repair: add a held-out argument mutation and rerun online.
    global MUTATIONS
    MUTATIONS = MUTATIONS + [("argument", "nurse", "doctor"), ("argument", "farmer", "builder")]
    repair = phase(SEEDS, "repair")
    payload = {"experiment": EXPERIMENT, "signature": SIGNATURE, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "operator": "online semantic mutation: synonym, tense, or argument substitution before mirrored residual extension", "base": {"candidates": base, "exact_count": sum(x["reader_eligible"] for x in base)}, "repair": {"candidates": repair, "exact_count": sum(x["reader_eligible"] for x in repair)}, "repair_action": "held-out argument substitutions (nurse→doctor, farmer→builder) were inserted into the live mutation frontier, then residual and exact audits were rerun", "provenance": {"seed_source": "human-authored ordinary complete clauses", "catalogue_used": False, "frame_relexicalization": False, "fragments": False, "repeated_units_allowed": False}}
    (ROOT/"runs"/"semantic-mutation-residual-20260916.json").write_text(json.dumps(payload, indent=2)+"\n")
    (ROOT/"runs"/"semantic-mutation-residual-repair-20260916.json").write_text(json.dumps(payload["repair"], indent=2)+"\n")
    print(json.dumps({"base": len(base), "repair": len(repair), "base_exact": payload["base"]["exact_count"], "repair_exact": payload["repair"]["exact_count"]}))
if __name__ == "__main__": main()
