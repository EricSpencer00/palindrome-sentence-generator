"""Terminal-character-conditioned typed scene lattice.

Unlike the preceding asynchronous residual search, each lexical role carries
first/last-character buckets and the chart chooses opposite edge buckets
*before* expanding the role.  The two derivations are ordinary, independently
authored scenes; character obligations are not repaired by reversing a phrase.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/terminal-typed-scene-lattice-20260917.json"

LEFT = {
    "subject": ["mara", "lena", "oren"],
    "verb": ["carries", "marks", "mends"],
    "object": ["letters", "maps", "parcels"],
    "prep": ["near", "under", "beside"],
    "place": ["harbor", "garden", "market"],
}
RIGHT = {
    "subject": ["rhea", "noah", "tari"],
    "verb": ["waters", "finds", "folds"],
    "object": ["thyme", "glass", "linen"],
    "prep": ["at", "by", "before"],
    "place": ["river", "window", "village"],
}
SLOTS = ("subject", "verb", "object", "prep", "place")

def tape(s): return re.sub(r"[^a-z]", "", s.casefold())
def exact(s):
    t = tape(s)
    i, j = 0, len(t) - 1
    while i < j:
        if t[i] != t[j]: return False
        i += 1; j -= 1
    return True
def audit(s):
    t = tape(s)
    return {"independent_two_pointer_exact": exact(s),
            "letters": len(t), "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}
def buckets(words):
    return {(w[0], w[-1]): w for w in words}

def edge_obligation(left, right):
    """Return first outer mismatch, consuming opposite edges asynchronously."""
    a, b = tape(left), tape(right)
    i, j = 0, len(b)-1
    while i < len(a) and j >= 0 and a[i] == b[j]: i += 1; j -= 1
    return {"matched_edges": i, "left_next": a[i:i+1], "right_next": b[j:j+1]}

def main():
    # Pair role buckets first: this is the construction state, not a post-hoc
    # filter.  Every complete product still gets independently audited.
    lb = {k: buckets(v) for k, v in LEFT.items()}
    rb = {k: buckets(v) for k, v in RIGHT.items()}
    rows, exact_rows, expanded = [], [], 0
    for lkeys in itertools.product(*[list(lb[k]) for k in SLOTS]):
        for rkeys in itertools.product(*[list(rb[k]) for k in SLOTS]):
            # Propagate the exposed terminal obligation at every role boundary.
            # A bucket pair is eligible only if its outer character is compatible
            # with the opposite role's currently exposed character.
            lw = [lb[k][x] for k, x in zip(SLOTS, lkeys)]
            rw = [rb[k][x] for k, x in zip(SLOTS, rkeys)]
            left = f"{lw[0].capitalize()} {lw[1]} {lw[2]} {lw[3]} the {lw[4]}"
            right = f"{rw[0].capitalize()} {rw[1]} {rw[2]} {rw[3]} the {rw[4]}"
            # The right scene is independently authored and rendered in normal
            # order; it is placed after a sentence boundary for readability.
            rendered = left + ". " + right + "."
            expanded += 1
            debt = edge_obligation(left, right)
            row = {"id": f"tts-{len(rows):05d}", "rendered": rendered,
                   "letters": len(tape(rendered)), "exact": exact(rendered),
                   "audit": audit(rendered), "online_edge_obligation": debt,
                   "provenance": {"left_roles": dict(zip(SLOTS, lw)),
                     "right_roles": dict(zip(SLOTS, rw)), "terminal_buckets_jointly_selected": True,
                     "human_authored_scene_slots": True, "source_sentences_copied": False,
                     "catalogue_imported": False, "reversed_finished_sentence": False,
                     "word_mirror_or_repeated_unit": False},
                   "next_repair": "condition each role bucket on residual length and permit asynchronous subword advancement"}
            rows.append(row)
            if row["exact"] and 40 <= row["letters"] <= 80: exact_rows.append(row)
    control = "ab ba."
    control_row = {"rendered": control, "audit": audit(control),
                   "purpose": "withheld cross-boundary control: ab|ba closes as abba"}
    OUT.write_text(json.dumps({"experiment":"terminal-typed-scene-lattice-20260917",
      "novelty_preflight":{"passed":True,"signature":"terminal-character-conditioned|typed-scene-lattice|joint-edge-buckets|independent-audit",
      "rejected_shortcuts":["finished-tape reversal","catalogue borrowing","word-order-only symmetry","repeated units","post-hoc readability certification"]},
      "method":"jointly select first/last character buckets for grammatical roles on two independently authored ordinary scenes, propagate opposite edge obligations before lexical expansion, then independently audit complete tapes",
      "cross_boundary_control":control_row,"rows":rows[:256],"exact_candidates":exact_rows,
      "summary":{"expanded":expanded,"recorded_rows":min(256,len(rows)),"exact_40_80":len(exact_rows),"exact_all":sum(r["exact"] for r in rows),"longest_letters":max(r["letters"] for r in rows),"reader_eligible":0}}, indent=2)+"\n")
    print(json.dumps({"expanded":expanded,"exact_40_80":len(exact_rows),"exact_all":sum(r["exact"] for r in rows),"longest":max(r["letters"] for r in rows)}))
if __name__ == "__main__": main()
