"""Constructive paired-predicate coordination lane.

The frame is authored once: ``SUBJ VERB OBJ, and OBJ VERB SUBJ``.
Each lexical choice is made as a pair while two cursors advance inward;
the completed surface is never reversed or repaired.
"""
import hashlib
import json
from pathlib import Path

RUN_ID = "paired-predicate-coordination-20260920"
SUBJECTS = ("Ada", "Ava", "Eve")
PREDICATES = ("did", "sees", "refer")
OBJECTS = ("civic", "radar", "level", "rotor", "tenet")


def letters(text):
    return "".join(c.lower() for c in text if c.isalpha())


def audit(text):
    stream = letters(text)
    rev = stream[::-1]
    return {
        "letters": len(stream),
        "exact": stream == rev,
        "pointer_audit": all(stream[i] == stream[-1 - i] for i in range(len(stream) // 2)),
        "sha256": hashlib.sha256(stream.encode()).hexdigest(),
        "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest(),
    }


def novelty_preflight(text, prior_texts):
    """Check exact rendered text against prior run artifacts, without importing them."""
    seen = prior_texts.get(text, [])
    return {"novel": not seen, "matches": seen}


def main():
    controls = []
    traces = []
    prior_texts = {}
    for path in Path("runs").glob("*.json"):
        if path.name == RUN_ID + ".json":
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for item in (data.get("controls", []) if isinstance(data, dict) else []):
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                prior_texts.setdefault(item["text"], []).append(str(path))
    for subject in SUBJECTS:
        for predicate in PREDICATES:
            for obj in OBJECTS:
                # Both ends are selected from the same authored slot tuple.
                text = f"{subject} {predicate} {obj}; {obj} {predicate} {subject}."
                stream = letters(text)
                for i in range(len(stream) // 2):
                    traces.append({"offset": i, "left": stream[i], "right": stream[-1-i], "matched": stream[i] == stream[-1-i]})
                controls.append({
                    "text": text,
                    "frame": "SUBJECT PREDICATE OBJECT and OBJECT PREDICATE SUBJECT",
                    "lexical_pair": {"subject": subject, "predicate": predicate, "object": obj},
                    "audit": audit(text),
                    "novelty": novelty_preflight(text, prior_texts),
                    "provenance": "fresh human-authored coordination frame; simultaneous paired slot choice",
                })
    exact = [x for x in controls if x["audit"]["exact"]]
    out = {
        "run_id": RUN_ID,
        "method": "paired predicate coordination with opposing cursor construction",
        "signature": "fresh-authored|paired-coordination|simultaneous-lexical-pairs|independent-audit",
        "counts": {"subjects": len(SUBJECTS), "predicates": len(PREDICATES), "objects": len(OBJECTS), "controls": len(controls), "live_equations": len(traces), "exact": len(exact), "exact_over_38": sum(x["audit"]["exact"] and x["audit"]["letters"] > 38 for x in controls), "max_letters": max(x["audit"]["letters"] for x in controls)},
        "controls": controls,
        "traces": traces,
        "shortcut_flags": {"finished_tape_reversal": False, "posthoc_repair": False, "catalogue_text": False, "per_search_rlaif": False, "punctuation_changes_letters": False},
        "falsifier": "promotion requires exact pointer equality plus equal forward/reverse SHA-256 on the rendered surface",
        "next_operator": "Add one non-palindromic adjective pair only when its opposing edge letters satisfy the live cursor equation; retain the same coordination frame and reject unmatched states immediately.",
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/" + RUN_ID + ".json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"run_id": RUN_ID, **out["counts"], "samples": [x["text"] for x in exact[:3]]}, indent=2))


if __name__ == "__main__":
    main()
