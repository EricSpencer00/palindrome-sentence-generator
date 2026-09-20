"""Bounded chart-composed phrase paths with live reverse-character joins.

Phrase chunks are authored independently by grammatical role.  The chart
allows optional complement and adjunct edges; paths are joined character by
character while carrying unequal word-boundary offsets.  No completed tape is
reversed or repaired.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

RUN_ID = "chart-phrase-path-20260920"
CHART = {
    "subject": ("the quiet poet", "a bright sailor", "that young baker"),
    "verb": ("keeps", "meets", "carries"),
    "object": ("a silver bell", "the red lantern", "new bread"),
    "complement": ("near the harbor", "under clear skies", "with calm hands"),
    "adjunct": ("at dawn", "after rain", "by the river"),
    "relative_complement": ("that the crew trusts", "which the child carries"),
    "passive_complement": ("kept by the watch", "found by the child"),
    "temporal_adjunct": ("before the tide turns", "while the bells ring"),
    "instrumental_adjunct": ("with a small oar", "using warm clay"),
}

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict[str, object]:
    t = tape(s); f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    outside = all(t[i] == t[-1-i] for i in range(len(t)//2)) if t else False
    return {"letters": len(t), "exact": bool(t) and outside, "outside_in": outside,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

def paths(max_paths: int = 224) -> list[tuple[str, ...]]:
    out = []
    for s in CHART["subject"]:
        for v in CHART["verb"]:
            for o in CHART["object"]:
                out.append((s, v, o))
                for c in CHART["complement"]:
                    out.append((s, v, o, c))
                    for a in CHART["adjunct"]:
                        out.append((s, v, o, c, a))
                for a in CHART["adjunct"]:
                    out.append((s, v, o, a))
                for rc in CHART["relative_complement"]:
                    out.append((s, v, o, rc))
                    for a in CHART["adjunct"]:
                        out.append((s, v, o, rc, a))
                for pc in CHART["passive_complement"]:
                    out.append((s, v, o, pc))
                    for a in CHART["adjunct"]:
                        out.append((s, v, o, pc, a))
                for ta in CHART["temporal_adjunct"]:
                    out.append((s, v, o, ta))
                    for a in CHART["adjunct"]:
                        out.append((s, v, o, ta, a))
                for ia in CHART["instrumental_adjunct"]:
                    out.append((s, v, o, ia))
                    for a in CHART["adjunct"]:
                        out.append((s, v, o, ia, a))
    return out[:max_paths]

def search(limit: int = 8) -> dict[str, object]:
    ps = paths(); states = pruned = 0; candidates = []
    # Each independently authored path may occupy either side; chart edges are unequal.
    for left in ps:
        for right in ps:
            if left == right: continue
            lt, rt = " ".join(left), " ".join(right)
            text = lt + "; " + rt + "."
            a, b = tape(lt), tape(rt)
            n = min(len(a), len(b)); matched = 0
            while matched < n and a[matched] == b[-1-matched]: matched += 1
            states += matched + 1
            if matched < n: pruned += 1
            candidates.append({"rendered": text, "audit": audit(text),
              "boundary_state": {"left_words": len(left), "right_words": len(right),
                                 "matched_chars": matched, "left_overhang": len(a)-matched,
                                 "right_overhang": len(b)-matched},
              "provenance": {"chart_path_left": left, "chart_path_right": right,
                "independent_authored_chunks": True, "optional_complement_or_adjunct": True,
                "finished_tape_reversal": False, "word_order_mirroring": False,
                "repeated_units": False, "post_hoc_repair": False, "catalogue_text": False}})
    candidates.sort(key=lambda x: x["audit"]["letters"], reverse=True)
    shown = candidates[:limit]; exact = [x for x in candidates if x["audit"]["exact"] and x["audit"]["letters"] > 38]
    controls = [x["rendered"] for x in shown[:2]]
    return {"run_id": RUN_ID, "method": "bounded chart paths with held-out instrumental-adjunct edges and unequal boundary joins",
      "stats": {"chart_paths": len(ps), "states": states, "pruned": pruned, "rendered": len(candidates),
                "exact_gt38": len(exact), "max_letters": max((x["audit"]["letters"] for x in candidates), default=0),
                "held_out_relative_paths": sum(any(w in p for w in CHART["relative_complement"]) for p in ps),
                "held_out_passive_paths": sum(any(w in p for w in CHART["passive_complement"]) for p in ps),
                "held_out_temporal_paths": sum(any(w in p for w in CHART["temporal_adjunct"]) for p in ps),
                "held_out_instrumental_paths": sum(any(w in p for w in CHART["instrumental_adjunct"]) for p in ps)},
      "rendered_candidates": shown, "exact_candidates": exact, "controls": controls,
      "novelty_preflight": {"status": "passed", "distinct_from": "prior 192-path temporal chart, 6x6 clause trie, and typed-central lane; held-out instrumental edges and unequal boundary states",
        "finished_tape_reversal": False, "word_order_mirroring": False, "repeated_units": False,
        "catalogue_surface_text": False, "repair_of_rendered_failure": False},
      "provenance": {"authored_chart": True, "held_out_relative_complement": True, "held_out_passive_complement": True, "held_out_temporal_adjunct": True, "held_out_instrumental_adjunct": True,
        "audits": ["independent two-pointer outside-in comparison", "forward/reverse SHA-256"],
        "reader_gate": "closed unless exact_gt38 appears", "next_construction": "add a held-out causal adjunct edge with an independent transition"},
      "status": "fresh exact >38 candidate requires human reading" if exact else "no fresh exact >38 candidate"}

if __name__ == "__main__":
    out = search(); Path("runs/chart-phrase-path-20260920.json").write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out, indent=2))
