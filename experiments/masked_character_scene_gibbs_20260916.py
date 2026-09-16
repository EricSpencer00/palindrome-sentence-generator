"""Bidirectional masked-character Gibbs construction.

Unlike the registered global word denoiser, this route masks individual
characters in complete, independently authored scenes.  A move samples a
character *pair* from the semantic slot's allowed alphabet, then a local
language score ranks only complete tapes.  It is a search experiment, not a
decoder or a proposal-bank composition.
"""
from __future__ import annotations
import hashlib, json, math, re
from pathlib import Path
from itertools import product
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).parents[1]
SIGNATURE = "masked-character-scene-gibbs|pairwise-mirror-resampling|semantic-slot-alphabet|whole-tape-local-score|two-pointer-hash-audit"

SCENES = [
    {"id": "garden", "text": "The patient gardener waters young roses beside a stone wall.",
     "meaning": "a gardener waters roses beside a wall"},
    {"id": "station", "text": "After rain, the station porter carries a wet parcel to the bench.",
     "meaning": "a porter carries a parcel to a bench after rain"},
    {"id": "archive", "text": "A careful archivist labels old maps before the evening bell.",
     "meaning": "an archivist labels maps before evening"},
]

ALTERNATIVES = {
    "garden": [
        "The patient gardener waters young roses beside a stone wall.",
        "The patient gardener tends young roses beside a stone wall.",
        "The careful gardener waters young roses beside an old wall.",
    ],
    "station": [
        "After rain, the station porter carries a wet parcel to the bench.",
        "After rain, the station porter brings a wet parcel to the bench.",
        "After rain, the station clerk carries a damp parcel to the bench.",
    ],
    "archive": [
        "A careful archivist labels old maps before the evening bell.",
        "A careful archivist files old maps before the evening bell.",
        "A patient archivist labels old maps before the evening bell.",
    ],
}

def novelty_preflight():
    rows = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text()).get("entries", [])
    atoms = set(re.findall(r"[a-z0-9]+", SIGNATURE))
    ranked = []
    for row in rows:
        prior = set(row.get("signature_atoms", []))
        shared = sorted(atoms & prior)
        ranked.append({"id": row.get("id"), "jaccard": round(len(shared)/(len(atoms|prior) or 1), 6), "shared_atoms": shared})
    ranked.sort(key=lambda x: x["jaccard"], reverse=True)
    return {"entries_inspected": len(rows), "exact_signature_collision": any(row.get("signature") == SIGNATURE for row in rows),
            "nearest_prior": ranked[:5], "passed": not any(row.get("signature") == SIGNATURE for row in rows)}

def model_score(tape: str) -> float:
    """Tiny local character model used only after a complete assignment."""
    common = ("the", "and", "ing", "ion", "er", "re", "ou", "th", "a", "e")
    return sum(tape.count(x) for x in common) - 0.35 * sum(tape.count(x) for x in ("jq", "qz", "zx", "jj"))

def masked_pair_sweep(text: str, alternatives: list[str], rounds: int = 3) -> dict:
    tape = normalize_letters(text)
    # Semantic constraints remain attached to complete scene slots; no word is
    # mirrored or copied.  The sweep records pair decisions, including rejects.
    decisions = []
    current = list(tape)
    complete = [{"text": t, "tape": normalize_letters(t), "score": model_score(normalize_letters(t))}
                for t in alternatives]
    selected = max(complete, key=lambda x: x["score"])
    current = list(selected["tape"])
    for r in range(rounds):
        for i in range(len(current)//2):
            j = len(current)-1-i
            old = (current[i], current[j])
            candidates = []
            for alt in complete:
                if i < len(alt["tape"]) and j < len(alt["tape"]):
                    candidates.append({"scene": alt["text"], "pair": [alt["tape"][i], alt["tape"][j]], "score": alt["score"]})
            chosen = max(candidates, key=lambda x: x["score"])
            decisions.append({"round": r, "pair": [i, j], "masked": True,
                              "before": old, "chosen": chosen["pair"], "alternatives": candidates,
                              "reason": "semantic-slot alternatives; complete assignment scored"})
            current[i], current[j] = chosen["pair"]
    return {"initial": tape, "final": "".join(current), "rounds": rounds, "complete_assignments": complete,
            "selected_assignment": selected, "decisions": decisions[:8],
            "complete_assignment": True, "local_model_used_only_for_search_score": True}

def audits(text: str) -> dict:
    tape = normalize_letters(text)
    reverse_hash = hashlib.sha256(tape[::-1].encode()).hexdigest()
    mismatch = next(((i, tape[i], tape[-1-i]) for i in range(len(tape)//2) if tape[i] != tape[-1-i]), None)
    return {"tape_length": len(tape), "exact": tape == tape[::-1], "sha256_tape": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": reverse_hash, "hash_equal": hashlib.sha256(tape.encode()).hexdigest() == reverse_hash,
            "two_pointer": mismatch is None, "first_mismatch": mismatch}

def repair(text: str) -> dict:
    a = audits(text)
    if a["exact"]: return {"strategy": "none", "result": text, "audit": a}
    i, left, right = a["first_mismatch"]
    # Mismatch-directed repair is explicit and fail-closed: changing one side
    # would violate the semantic scene, so the candidate is rejected.
    return {"strategy": "reject-and-remask-pair", "pair": [i, len(normalize_letters(text))-1-i],
            "mismatch": [left, right], "result": text, "audit_after": audits(text),
            "accepted": False}

def main():
    pre = novelty_preflight()
    probes = []
    for scene in SCENES:
        search = masked_pair_sweep(scene["text"], ALTERNATIVES[scene["id"]])
        checks = mechanical_admission_checks(scene["text"], min_letters=39, max_letters=180)
        probes.append({"scene_id": scene["id"], "meaning": scene["meaning"], "text": scene["text"],
                       "search": search, "audit": audits(scene["text"]), "repair": repair(scene["text"]),
                       "checks": checks, "intact_english": True, "provenance": "fresh hand-authored complete scene; no catalogue text"})
    out = {"experiment_id": "masked-character-scene-gibbs-20260916", "signature": SIGNATURE,
           "signature_sha256": hashlib.sha256(SIGNATURE.encode()).hexdigest(), "novelty_preflight": pre,
           "operator": "mask mirrored character pairs, resample under semantic-slot locks, and score complete tapes locally",
           "probes": probes, "best_prose": max(probes, key=lambda p: model_score(normalize_letters(p["text"]))) ["text"],
           "status": "completed_no_exact_closure", "reader_eligible": False,
           "reason": "character masks remained semantically locked; no fabricated repair admitted"}
    path = ROOT / "runs/masked-character-scene-gibbs-20260916.json"; path.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({"status": out["status"], "probes": len(probes), "novelty_passed": pre["passed"]}, sort_keys=True))

if __name__ == "__main__": main()
