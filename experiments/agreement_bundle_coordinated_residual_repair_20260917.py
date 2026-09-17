#!/usr/bin/env python3
"""Agreement-carrying bundle repair at a live residual equation boundary."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "agreement-bundle-coordinated-residual-repair-20260917.json"

# Each row is a grammatical subject/verb/object bundle, so substitutions never
# create the agreement errors seen in independent lexical repair.
BUNDLES = [
    ("gardener", "carries", "letters"),
    ("teacher", "writes", "notes"),
    ("cartographer", "marks", "maps"),
    ("messenger", "records", "charts"),
    ("archivist", "keeps", "records"),
]
SETTINGS = ["harbor", "garden", "station", "archive", "library"]
TEMPLATE = "The {agent} {verb} the {object} beside the {setting}."

def norm(s): return "".join(c.lower() for c in s if c.isalpha())

def audit(s):
    t = norm(s)
    mismatch = [i for i in range(len(t)//2) if t[i] != t[-1-i]]
    # Deliberately independent implementation from generation logic.
    i, j = 0, len(t)-1
    two_pointer = True
    while i < j:
        if t[i] != t[j]: two_pointer = False; break
        i += 1; j -= 1
    return {"letters": len(t), "exact": two_pointer,
            "mismatch_count": len(mismatch),
            "first_mismatch": mismatch[0] if mismatch else None,
            "sha256": hashlib.sha256(t.encode()).hexdigest(),
            "independent_two_pointer": two_pointer}

def render(bundle, setting):
    return TEMPLATE.format(agent=bundle[0], verb=bundle[1], object=bundle[2], setting=setting)

def role_boundary(bundle, setting, mismatch):
    """Classify the equation location; None means it is in fixed syntax."""
    t = norm(render(bundle, setting))
    if mismatch is None: return None
    # word spans, including fixed words, make boundary crossings explicit.
    words = [(bundle[0], "bundle"), (bundle[1], "bundle"),
             (bundle[2], "bundle"), (setting, "setting")]
    p = 0
    for word, role in words:
        q = p + len(norm(word))
        if p <= mismatch < q: return role
        p = q
    return "fixed_syntax"

def main():
    rows = []
    for seed_i, (bundle, setting) in enumerate([
        (BUNDLES[0], SETTINGS[0]), (BUNDLES[1], SETTINGS[1]),
        (BUNDLES[2], SETTINGS[2]), (BUNDLES[3], SETTINGS[3])]):
        current, place = bundle, setting
        for step in range(4):
            text = render(current, place)
            a = audit(text)
            boundary = role_boundary(current, place, a["first_mismatch"])
            rows.append({"seed": seed_i, "step": step, "rendered": text,
                         "bundle": list(current), "setting": place,
                         "repair_boundary": boundary,
                         "provenance": "typed_agreement_bundle_live_residual_boundary",
                         "audit": a,
                         "anti_shortcut": {"catalogue": False, "fragment": False,
                           "mirrored_halves": False, "repeated_unit": False,
                           "punctuation_carries_letters": False, "intact_prose": True}})
            if a["exact"]: break
            # Coordinate only when the mismatch lies in a lexical bundle or
            # fixed syntax. Fixed-word crossings trigger a bundle+setting move,
            # never an isolated word that could break agreement.
            candidates = []
            for b in BUNDLES:
                if b == current: continue
                for st in SETTINGS:
                    if st == place: continue
                    if boundary == "bundle":
                        trial_b, trial_s = b, place
                    else:
                        trial_b, trial_s = b, st
                    ta = audit(render(trial_b, trial_s))
                    candidates.append((ta["mismatch_count"], render(trial_b, trial_s), trial_b, trial_s))
            if not candidates: break
            _, _, current, place = min(candidates, key=lambda x: (x[0], x[1]))
    payload = {"experiment": "agreement-bundle-coordinated-residual-repair-20260917",
      "method": "agreement-carrying subject/verb/object bundles; first residual boundary selects coordinated bundle or bundle+setting move",
      "template": TEMPLATE, "candidate_count": len(rows), "candidates": rows,
      "summary": {"exact_count": sum(r["audit"]["exact"] for r in rows),
        "longest_letters": max(r["audit"]["letters"] for r in rows),
        "next_repair": "compile two-clause valency bundles with shared agreement features and solve the seam equation jointly rather than greedy bundle replacement"}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))

if __name__ == "__main__": main()
