"""Independent semantic scene complete-pair diagnostic.

Unlike the earlier boundary diagnostic, both sides are authored from separate
lexical banks.  The first implementation rendered both clauses before checking
the character relation, so this file is retained only as a rejected diagnostic;
it is not evidence of a live construction method.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters

EXPERIMENT_ID = "independent-scene-overhang-lattice-v2-20260921"

# Independently authored, typed lexical choices.  The two banks deliberately
# have disjoint labels; lexical equality is allowed by chance, but no option
# is derived from another option's reverse spelling.
LEFT = {
    "subject": [("Ada", "singular"), ("Eve", "singular"), ("Otto", "singular")],
    "verb": [("sees", "transitive"), ("keeps", "transitive"), ("draws", "transitive")],
    "det": [("a", "singular"), ("the", "singular")],
    "noun": [("map", "singular"), ("note", "singular"), ("rune", "singular")],
}
RIGHT = {
    "subject": [("Ira", "singular"), ("Nora", "singular"), ("Liam", "singular")],
    "verb": [("reads", "transitive"), ("marks", "transitive"), ("holds", "transitive")],
    "det": [("a", "singular"), ("the", "singular")],
    "noun": [("book", "singular"), ("letter", "singular"), ("story", "singular")],
}
CENTERS = [(".", "declarative"), (";", "coordination"), ("?", "question")]

def clean(s: str) -> str:
    return normalize_letters(s)

def audit(text: str) -> dict:
    t = clean(text)
    r = t[::-1]
    return {"normalized": t, "letters": len(t),
            "two_pointer_exact": bool(t) and all(t[i] == t[-1-i] for i in range(len(t)//2)),
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(r.encode()).hexdigest()}

def shortcut_reasons(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    out = []
    if any(len(w) >= 3 and w == w[::-1] for w in words):
        out.append("self_palindromic_word_unit")
    ws = set(words)
    if any(len(w) >= 3 and w != w[::-1] and w[::-1] in ws for w in words):
        out.append("semordnilap_word_pair")
    return out

def consume(left: str, right: str, trace: list[dict]) -> tuple[bool, int]:
    """Audit a completed pair against the opposite character stream.

    Both clauses are already complete when this helper runs. The result is
    retained only to locate the first boundary failure.
    """
    # This reverse stream is explicitly post-render diagnostic bookkeeping.
    obligations = list(reversed(left))
    for pos, ch in enumerate(right):
        if not obligations or ch != obligations.pop(0):
            trace.append({"position": pos, "char": ch, "expected": obligations[0] if obligations else None,
                          "status": "mismatch"})
            return False, len(obligations)
        trace.append({"position": pos, "char": ch, "status": "consumed"})
    return True, len(obligations)

def scene_rows() -> list[dict]:
    rows = []
    for ls, ltag in LEFT["subject"]:
      for lv, lvtag in LEFT["verb"]:
       for ld, ldt in LEFT["det"]:
        for ln, lnt in LEFT["noun"]:
         if ldt != lnt: continue
         left = f"{ls} {lv} {ld} {ln}"
         for rs, rtag in RIGHT["subject"]:
          for rv, rvtag in RIGHT["verb"]:
           for rd, rdt in RIGHT["det"]:
            for rn, rnt in RIGHT["noun"]:
             if rdt != rnt: continue
             right = f"{rs} {rv} {rd} {rn}"
             for center, ctag in CENTERS:
              trace=[]
              # Center is independently chosen; punctuation contributes no
              # letters and is never used to alter the audit tape.
              center_ok, remaining = True, len(clean(left))
              # The complete-pair diagnostic records the failure boundary; it
              # is not a construction-time admission check.
              right_ok, remaining = consume(clean(left), clean(right), trace)
              rendered = f"{left}{center} {right}."
              a = audit(rendered)
              reasons = shortcut_reasons(rendered)
              rows.append({"rendered": rendered, "audit": a,
                "exact": a["two_pointer_exact"], "shortcut_rejections": reasons,
                "mechanically_admitted": a["two_pointer_exact"] and not reasons,
                "provenance": {"left_bank":"human_authored_scene_v2", "right_bank":"independent_human_authored_scene_v2",
                  "catalogue_imported":False, "reverse_derived_token":False, "finished_tape_reversed":True,
                  "construction_status":"rejected_post_render_pair_diagnostic",
                  "word_order_mirror":False, "center":center, "center_tag":ctag},
                "csp": {"live_obligation_check":False, "post_render_obligation_audit":True, "center_consumed":center_ok,
                  "right_consumed":right_ok, "remaining_obligations":remaining, "trace_prefix":trace[:12]},
                "reader_status":"not_run; only mechanically admitted rows enter reader gate"})
    return rows

def main() -> dict:
    rows = scene_rows()
    exact = [r for r in rows if r["exact"]]
    admitted = [r for r in exact if r["mechanically_admitted"]]
    controls = [{"rendered": r["rendered"], "kind":"intact_scene_control",
                 "shuffled":" ".join(r["rendered"].replace(".", "").split()[::-1]) + "."}
                for r in rows[:12]]
    result = {"experiment_id":EXPERIMENT_ID,
      "method":"independent semantic scene complete-pair diagnostic (post-render; rejected as live construction)",
      "stats":{"scene_pairs":len(rows), "exact_diagnostic":len(exact),
                "mechanically_admitted":len(admitted),
                "longest_exact_letters":max((r["audit"]["letters"] for r in exact),default=0),
                "longest_admitted_letters":max((r["audit"]["letters"] for r in admitted),default=0)},
      "candidates":sorted(rows,key=lambda r:-r["audit"]["letters"])[:200],
      "controls":controls,
      "independent_audit":["normalize_letters", "two-pointer character comparison", "SHA-256 forward/reverse"],
      "shortcut_gate":{"no_catalogue_import":True,"no_reverse_derived_tokens":True,
        "no_finished_tape_reverse":False,"post_render_pair_check":True,
        "no_word_order_mirror":True,
        "reader_gate":"closed; rejected post-render diagnostic"},
      "next_operator":"replace complete-pair checking with a slot-by-slot deque search: left slots forward, right slots from its final slot backward, with typed dependency/valency state"}
    return result

if __name__ == "__main__":
    out=main(); (ROOT/"runs"/(EXPERIMENT_ID+".json")).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out["stats"],sort_keys=True))
    for row in out["candidates"][:8]: print(row["rendered"])
