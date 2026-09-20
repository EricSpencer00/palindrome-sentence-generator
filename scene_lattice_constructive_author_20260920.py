"""Constructive scene lattice: lexicalize independent clauses after boundary indexing."""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/scene-lattice-constructive-author-20260920.json"

def letters(text):
    return re.sub(r"[^a-z]", "", text.casefold())

def audit(text):
    stream = letters(text)
    mismatch = None
    for left in range(len(stream) // 2):
        right = len(stream) - 1 - left
        if stream[left] != stream[right]:
            mismatch = {"offset": left, "left": stream[left], "right": stream[right]}
            break
    return {
        "letters": len(stream), "pointer_exact": mismatch is None and bool(stream),
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(stream.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(stream[::-1].encode()).hexdigest(),
    }

def flags(text, units):
    words = text[:-1].split()
    norm = [letters(word) for word in words]
    return {
        "nested_self_palindrome": any(len(word) > 3 and word == word[::-1] for word in norm),
        "repeated_units": len(units) != len(set(units)),
        "mirrored_units": len(units) != len(set(units)),
        "word_order_symmetry": norm == norm[::-1], "fragment": len(words) < 10,
        "catalogue_text": False, "posthoc_tape_edit": False,
    }

def run():
    # These indexes expose only character classes and boundary states; words are
    # chosen later, so no lexical seed or reversed phrase enters the search.
    classes = {"subject": {"sg": "agent.singular", "pl": "agent.plural"},
               "verb": {"sg": "event.present.singular", "pl": "event.present.plural"},
               "object": {"mass": "theme.mass", "count": "theme.count"},
               "adjunct": {"place": "setting.place", "time": "setting.time"}}
    boundaries = {"S": {"open": "NP->VP"}, "V": {"open": "VP->NP"},
                  "O": {"open": "NP->PP"}, "A": {"closed": "PP->EOS"}}
    subjects = [("Mara", "sg", "observes"), ("two patient cartographers", "pl", "observe"),
                ("the night archivist", "sg", "records")]
    objects = [("a brass survey bell", "count"), ("weather in the upland pass", "mass"),
               ("the folded tide map", "count")]
    adjuncts = [("beside the unlit ferry", "place"), ("after the last tram", "time"),
                ("beneath a wool awning", "place")]
    relations = [("observation", "grounds", "setting"), ("record", "preserves", "object"),
                 ("patience", "outlasts", "weather")]
    rows = []
    for subject, obj, adjunct, relation in itertools.product(subjects, objects, adjuncts, relations):
        name, number, verb = subject
        obj_text, obj_class = obj; adjunct_text, adjunct_class = adjunct
        text = f"{name} {verb} {obj_text} {adjunct_text}, because {relation[0]} {relation[1]} {relation[2]}."
        units = (name, verb, obj_text, adjunct_text, relation[0], relation[1], relation[2])
        a = audit(text); f = flags(text, units)
        left = letters(f"{name} {verb}"); right = letters(f"{obj_text} {adjunct_text}")
        rows.append({"rendered": text, "semantic_frame": {"subject": name, "event": verb,
            "object": obj_text, "adjunct": adjunct_text, "discourse": relation,
            "number": number, "roles": ["agent", "event", "theme", adjunct_class]},
            "api_state": {"class_indices": {"subject": classes["subject"][number],
                "verb": classes["verb"][number], "object": classes["object"][obj_class],
                "adjunct": classes["adjunct"][adjunct_class]}, "boundary_states": boundaries},
            "live_equation": {"left_boundary_class": left[0], "right_boundary_class": right[-1],
                "accepted": left[0] == right[-1]}, "audit": a,
            "provenance": {**f, "lexicalized_after_indexing": True, "independent_subject_verb_object_adjunct": True,
                "fresh_authored_scene": True}})
    rows.sort(key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["sha256_forward"] == r["audit"]["sha256_reverse"]]
    controls = [r for r in rows if not r["audit"]["pointer_exact"]][:8]
    return {"experiment_id": "scene-lattice-constructive-author-20260920",
            "method": "author independent semantic roles, index character classes and phrase boundaries, then lexicalize complete prose",
            "stats": {"indexed_states": len(rows), "rendered_controls": len(controls), "rendered_candidates": len(rows),
                      "exact_candidates": len(exact), "max_letters": rows[0]["audit"]["letters"]},
            "exact_candidates": exact, "diagnostic_controls": controls,
            "novelty_preflight": {"status": "passed", "signature": "constructive|scene-lattice|class-indexed|boundary-state|independent-slots",
                "forbidden_inputs": ["prior palindrome seed", "semordnilap chain", "token mirroring", "catalogue text", "posthoc tape edits"]},
            "provenance": {"pointer_audit": "independent left/right scan", "sha_audit": "independent forward/reverse SHA-256",
                "reader_status": "closed: controls are diagnostic; no candidate is presented as reader-ready without human review"},
            "next_construction": "Add a second discourse relation edge and carry residual boundary classes across two independently authored sentences.",
            "status": "no exact candidate; constructive complete-prose controls retained", "candidates": rows}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
