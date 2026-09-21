"""Bounded residual-attachment lane for the documented ``ne``, ``ni`` and ``rimda`` seams.

The lane authors complete clauses, then adds one typed attachment on each side;
it never copies or reverses the 38-letter seed and never admits self-palindromic
names.  The character audit is deliberately independent of construction.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "residual-attachment-slots-20260921.json"

SEED_LENGTH = 38
SEAMS = {
    "ne": ("the quiet archivist", "noted", "a narrow map"),
    "ni": ("a patient gardener", "marked", "the new plot"),
    "rimda": ("the careful cartographer", "revised", "an old atlas"),
}


def tape(text):
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text):
    letters = tape(text)
    mismatch = next((i for i in range(len(letters) // 2)
                     if letters[i] != letters[-1 - i]), None)
    forward = hashlib.sha256(letters.encode()).hexdigest()
    reverse = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {
        "letters": len(letters),
        "two_pointer_exact": mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def residual_trace(text):
    letters = tape(text)
    left, right = 0, len(letters) - 1
    pairs = []
    while left < right and letters[left] == letters[right]:
        pairs.append((letters[left], left, right))
        left += 1
        right -= 1
    return {
        "pairs_checked": len(pairs),
        "residual_before": letters[left:right + 1],
        "obligation": None if left >= right else (letters[left], letters[right]),
        "next_left_index": left,
        "next_right_index": right,
    }


def main():
    # A tiny, fixed product: one fresh authored sentence per documented seam.
    attachments = {
        "ne": ("who filed the map in spring", "while the bell sounded"),
        "ni": ("which the foreman praised at noon", "as the rain crossed the yard"),
        "rimda": ("that the survey team had annotated", "while distant gulls circled"),
    }
    old = set()
    for path in ROOT.glob("runs/*.json"):
        if path == RUN:
            continue
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                continue
            prior_rows = payload.get("rendered_candidates", payload.get("rows", []))
            if not isinstance(prior_rows, list):
                continue
            for row in prior_rows:
                if isinstance(row, dict) and row.get("rendered"):
                    old.add(tape(row["rendered"]))
        except (OSError, json.JSONDecodeError):
            continue

    rows = []
    for seam, (subject, verb, obj) in SEAMS.items():
        left_attach, right_attach = attachments[seam]
        rendered = f"{subject}, {left_attach}, {verb} {obj}; {right_attach}, the witness agreed."
        normalized = tape(rendered)
        trace = residual_trace(rendered)
        row = {
            "seam": seam,
            "rendered": rendered,
            "attachment": {"left": left_attach, "right": right_attach,
                            "type": "relative_clause_plus_temporal_PP"},
            "seed_control": {"seed_letters": SEED_LENGTH, "seed_wrapped": False,
                             "seed_copied": False, "self_palindromic_name": False},
            "novel_rendered": normalized not in old,
            "audit": audit(rendered),
            "live_residual": trace,
            "exact_admitted": trace["obligation"] is None,
            "provenance": {
                "fresh_complete_prose": True,
                "joint_character_obligations": True,
                "documented_live_seam": seam,
                "finished_tape_reversal": False,
                "post_hoc_repair": False,
                "catalogue_text": False,
                "source_sentences_copied": False,
            },
            "anti_shortcut_flags": {"wrapper_used": False, "seed_reused": False,
                                    "reversed_finished_sentence": False,
                                    "self_palindromic_name": False},
        }
        rows.append(row)

    exact = [r for r in rows if r["exact_admitted"] and r["novel_rendered"]]
    out = {
        "experiment_id": "residual-attachment-slots-20260921",
        "status": "completed_exact" if exact else "completed_no_exact_closure",
        "method": "bounded typed attachment on both sides of ne/ni/rimda live seams",
        "candidate_count": len(rows), "exact_count": len(exact),
        "rendered_candidates": rows,
        "stats": {"longest_letters": max(r["audit"]["letters"] for r in rows),
                  "seams_checked": list(SEAMS), "bounded_candidates": len(rows)},
        "novelty_preflight": {"all_rendered_new": all(r["novel_rendered"] for r in rows),
                              "fresh_complete_prose": True, "seed_copy_forbidden": True,
                              "self_palindromic_names_forbidden": True},
        "failure_and_repair": {"failure": "typed attachments cross the 38-letter seed but leave a live residual" if not exact else "none",
                                "residual": [r["live_residual"]["residual_before"] for r in rows],
                                "next_repair": "use seam-specific agreement inflection at the exposed residual, keeping both attachments typed"},
        "queue_row": {"lane": "residual_attachment_slots", "seams": list(SEAMS),
                      "priority": "next", "action": "agreement-inflection repair", "bounded": True},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer", "forward/reverse SHA-256"],
                       "shortcuts_excluded": True},
    }
    RUN.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "candidates": len(rows),
                      "exact": len(exact), "longest_letters": out["stats"]["longest_letters"]}))


if __name__ == "__main__":
    main()
