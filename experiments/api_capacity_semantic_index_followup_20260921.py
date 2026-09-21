#!/usr/bin/env python3
"""Follow-up: one extra grammatical role, same capacity-indexed online join."""
import hashlib, json, re
from collections import defaultdict
from pathlib import Path

CONTROL = {"max_chars": 96, "min_target_letters": 40, "max_fragments": 6}
PATHS = [
    ("SUBJ", "VERB", "OBJ", ("the baker reads the note", ("AGENT", "PERCEPTION", "ARTIFACT"))),
    ("SUBJ", "VERB", "OBJ", ("the nurse records the result", ("AGENT", "REPORT", "OUTCOME"))),
    ("TEMP", "SUBJ", "VERB", ("at noon the baker waits", ("TIME", "AGENT", "STATE"))),
    ("RECIP", "VERB", "OBJ", ("the guide gives the map", ("AGENT", "TRANSFER", "ARTIFACT"))),
    ("OBJ", "VERB", "TEMP", ("the bell rings at noon", ("EVENT", "TIME"))),
]

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def index_paths():
    out = defaultdict(list)
    for _, _, _, (text, tags) in PATHS:
        n = norm(text)
        out[(n[0], n[-1], len(n))].append({"text": text, "norm": n, "tags": tags})
    return out
def residual_dp(text, target):
    n = norm(text)
    return len(n) == target and all(n[i] == n[-1-i] for i in range(len(n)))

def main():
    ix = index_paths(); controls = [p[-1][0] for p in PATHS]
    exact = []
    # Online opposing-residual join; no pre-paired or reused witness.
    for lefts in ix.values():
        for left in lefts:
            for rights in ix.values():
                for right in rights:
                    rendered = left["text"] + " " + right["text"] + "."
                    n = norm(rendered)
                    if len(n) >= CONTROL["min_target_letters"] and n == n[::-1]:
                        exact.append({"rendered": rendered, "letters": len(n), "tags": [left["tags"], right["tags"]]})
    # Independent control checks on a non-palindromic ordinary sentence.
    control = controls[0] + "."
    cn = norm(control)
    report = {
        "experiment": "api-capacity-semantic-index-followup-20260921",
        "novelty_preflight": {"materially_distinct": False, "delta": "one extra grammatical role (OBJECT/RECIPIENT/TEMPORAL) over committed lane; same index/join operator"},
        "controls": {"prose_controls": controls, "capacity": CONTROL, "negative_control": control},
        "index": {"buckets": len(ix), "paths": len(PATHS), "key": "(first_character,last_character,normalized_length)"},
        "exact_candidates": exact,
        "checks": {"direct_reverse_negative_control": cn != cn[::-1], "residual_dp_negative_control": not residual_dp(control, len(cn)), "sha256_control": hashlib.sha256(cn.encode()).hexdigest()},
        "acceptance_frontier": {"changed": False, "reason": "no intact-prose exact candidate reached 40 normalized letters"},
        "no_shortcut_gates": {"reused_13_letter_witness": False, "reused_38_letter_seed": False, "nested_mirror_bank": False, "normalization_only": False},
        "next_operator": "add a typed recipient/temporal path with a fresh terminal-class bucket, then re-run the same residual join"
    }
    Path("runs/api-capacity-semantic-index-followup-20260921.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
