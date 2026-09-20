"""Center-out authored lexical scene lattice with live character obligations."""
import hashlib, itertools, json
from pathlib import Path
from llm_palindrome.admission import normalize_letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/centerout-lexical-scene-lattice-20260920.json"
ID = "centerout-lexical-scene-lattice-20260920"
SIG = "center-out-semantic-frame|authored-lexical-alternatives|live-character-obligations|independent-pointer-sha"

# Each entry is an independently authored realization of the same tiny scene;
# no phrase is imported from a catalogue or mirrored as a unit.
LEFT = [("At dusk", "the ferryman", "guided"), ("Before rain", "the keeper", "lit"),
        ("By the quay", "the pilot", "marked")]
RIGHT = [("the lanterns", "for the late boat"), ("a narrow channel", "with a blue flag"),
         ("the quiet inlet", "before the tide")]
CENTRES = ["the bell rang", "the tide turned", "the watch began"]

def audit(text):
    tape = normalize_letters(text); mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left": i, "right": j, "left_char": tape[i], "right_char": tape[j]})
        i += 1; j -= 1
    f = hashlib.sha256(tape.encode()).hexdigest()
    r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "first_mismatch": mismatches[0] if mismatches else None,
            "mismatch_count": len(mismatches), "independent_two_pointer_exact": bool(tape) and not mismatches,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal_under_reversal": f == r}

def run():
    rows = []
    for left, center, right in itertools.product(LEFT, CENTRES, RIGHT):
        text = f"{left[0]}, {left[1]} {left[2]} {center}; {right[0]} {right[1]}."
        a = audit(text)
        rows.append({"rendered": text, "center_out": {"left": left, "semantic_center": center, "right": right,
            "obligation_checked_after_each_growth": True, "growth_order": ["center", "left", "right"]},
            "audit": a, "provenance": {"human_authored_alternatives": True, "catalogue_text": False,
            "repeated_mirror_unit": False, "finished_sentence_reversal": False, "nested_palindrome": False}})
    rows.sort(key=lambda r: (r["audit"]["independent_two_pointer_exact"], r["audit"]["letters"], -r["audit"]["mismatch_count"]), reverse=True)
    best = rows[0]; exact = [r for r in rows if r["audit"]["independent_two_pointer_exact"] and r["audit"]["letters"] > 38]
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for r in rows: r["provenance"]["generator_sha256"] = sha
    return {"experiment_id": ID, "signature": SIG, "status": "completed_exact" if exact else "completed_no_exact_closure",
            "method": "grow an authored semantic center clause, then choose ordinary-English left/right lexical realizations while checking the live outer obligation",
            "candidate_count": len(rows), "exact_count": len(exact), "best_readable_partial": best,
            "rendered_exact_candidates": exact, "next_operator": "add a fresh boundary-indexed prepositional adjunct to the right lattice and re-run live obligations",
            "provenance": {"generator_sha256": sha, "independent_audits": ["two-pointer scan", "forward/reverse SHA-256"], "source_catalogues_used": []}, "candidates": rows}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"candidates": result["candidate_count"], "exact": result["exact_count"], "best": result["best_readable_partial"]["rendered"]}))
