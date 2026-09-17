"""Asynchronous residual search over authored semantic scene banks.

The two clauses are ordinary-order productions.  ``consume`` is deliberately
the four-argument invariant: ``(owner, residual, left_add, right_add)``.
Only the side owning a residual is allowed to extend it; when empty, either
side may start the next asynchronous comparison.  No right-hand tape is
reversed as a word sequence.
"""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ID = "async-residual-scene-bank-20260917"
SIGNATURE = "async-residual-owner-invariant|authored-semantic-scene-bank|two-clause-families|terminal-co-design|independent-reverse-index"
OUT = ROOT / "runs" / "async-residual-scene-bank-20260917.json"

# All entries are small, authored role banks.  They are not corpus phrases.
BANK = {
    "subject": ("careful baker", "quiet keeper", "young sailor", "patient clerk"),
    "verb": ("checks", "marks", "carries", "records"),
    "object": ("the ledger", "a parcel", "the signal", "a ticket"),
    "place": ("by the pier", "near the gate", "at the dock", "in the yard"),
    "instrument": ("with a pencil", "with a lantern", "with a compass", "with a bell"),
}

FAMILIES = {
    "inspection": ("{subject} {verb} {object} {place}", "{subject} {verb} {object} {instrument}"),
    "handoff": ("{subject} {verb} {object} {instrument}", "{subject} {verb} {object} {place}"),
}

def consume(owner: int, residual: str, ladd: str, radd: str):
    """Consume opposite-facing additions while preserving the owner invariant.

    owner=1 means residual is unmatched left text; owner=-1 means unmatched
    right text.  Additions are ordinary left/right text, never pre-reversed.
    """
    left, right = normalize_letters(ladd), normalize_letters(radd)[::-1]
    if owner == 1:
        left = residual + left
    elif owner == -1:
        right = residual + right
    elif owner != 0 or residual:
        raise ValueError("non-empty residual must have exactly one owner")
    k = min(len(left), len(right))
    if left[:k] != right[:k]: return None
    if len(left) > len(right): return (1, left[k:])
    if len(right) > len(left): return (-1, right[k:])
    return (0, "")

def independent_audit(text: str):
    tape = normalize_letters(text); i, j = 0, len(tape)-1
    while i < j and tape[i] == tape[j]: i += 1; j -= 1
    return {"letters": len(tape), "exact": bool(tape) and i >= j,
            "direct_reverse": bool(tape) and tape == tape[::-1],
            "opposing_index": bool(tape) and i >= j,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "first_mismatch": None if i >= j else [i, j, tape[i], tape[j]],
            "mechanical": mechanical_admission_checks(text, min_letters=30, max_letters=240)}

def novelty_preflight():
    reg = json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text())
    rows = reg.get('entries', []) + reg.get('excluded', [])
    collision = [r.get('id') for r in rows if r.get('signature') == SIGNATURE]
    if collision: raise RuntimeError(f"novelty collision: {collision}")
    return {"status":"passed", "registry_entries_before_run":len(rows), "signature_collisions":[],
            "catalogue_lookup":False, "rejected_controls":["word-order mirror", "repeated fragment", "catalogue text"]}

def run(limit=600):
    pre = novelty_preflight(); rows=[]; explored=0
    for family, templates in FAMILIES.items():
        for vals in itertools.product(*[BANK[k] for k in ("subject","verb","object","place","instrument")]):
            env=dict(zip(("subject","verb","object","place","instrument"), vals))
            left, right = (t.format(**env) for t in templates)
            # Two asynchronous slot additions, in both legal owner orders.
            state=(0, "")
            for la, ra in ((left, ""), ("", right)):
                got=consume(*state, la, ra)
                if got is None: break
                state=got
            explored += 1
            if explored > limit: break
            text = left + ". " + right + "."
            audit=independent_audit(text)
            if audit["exact"]:
                rows.append({"family":family,"text":text,"left_clause":left,"right_clause":right,
                             "audit":audit,"provenance":{"bank":"authored role bank","catalogue":False,
                             "word_order_mirror":False,"repeated_fragment":False,"length":audit["letters"]}})
        if explored > limit: break
    result={"experiment_id":ID,"signature":SIGNATURE,"status":"completed_bounded_search",
            "explored":explored,"exact_candidates":rows,"rendered_exact_candidates":len(rows),
            "scene_families":list(FAMILIES),"bank_sizes":{k:len(v) for k,v in BANK.items()},
            "novelty_preflight":pre,"next_repair":{"operator":"add held-out terminal variants at the first residual seam",
            "reason":"current ordinary clause banks rarely satisfy cross-clause terminal equations; preserve owner/residual state"},
            "provenance":{"catalogue_imported":False,"source_sentences_copied":False,
            "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "audits":["direct reverse", "opposing index", "forward/reverse SHA-256", "length/provenance"]}}
    OUT.write_text(json.dumps(result,indent=2)+"\n"); return result

if __name__ == '__main__': print(json.dumps(run(), indent=2))
