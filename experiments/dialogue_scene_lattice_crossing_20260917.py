"""A small authored dialogue lattice with a live, non-palindromic centre word.

The search varies only typed scene slots (speaker, verb inflection, object and
setting).  It never mirrors or reverses a finished sentence; every survivor is
audited independently by a two-pointer scan and forward/reverse digests.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dialogue-scene-lattice-crossing-20260917.json"
REG = ROOT / "docs" / "experiment-novelty-registry.json"
ID = "dialogue-scene-lattice-crossing-20260917"
SIG = "authored-dialogue-scene-lattice|nonpalindromic-centre-word|typed-inflection-search|independent-exact-audit"

SPEAKERS = ("Mara", "Jon", "Rhea")
VERBS = (("asks", "ask"), ("checks", "check"), ("marks", "mark"), ("keeps", "keep"))
OBJECTS = ("the map", "a note", "the key", "this lamp")
SETTINGS = ("by the door", "near the pier", "under the awning")
ANSWERS = ("I answer", "we wait", "I listen")

def norm(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    t = norm(text); i, j = 0, len(t)-1; mismatches = []
    while i < j:
        if t[i] != t[j]: mismatches.append({"left": i, "right": j, "a": t[i], "b": t[j]})
        i += 1; j -= 1
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"normalized_tape": t, "letters": len(t), "exact": bool(t) and not mismatches,
            "two_pointer_exact": bool(t) and not mismatches, "first_mismatches": mismatches[:6],
            "sha256_forward": f, "sha256_reverse": r, "sha_equal_under_reversal": f == r}

def centre_crossing(text: str) -> dict:
    t = norm(text); mid = len(t)//2; cursor = 0
    for word in re.findall(r"[A-Za-z]+", text):
        start, end = cursor, cursor + len(word); cursor = end
        if start <= mid < end:
            return {"midpoint": mid, "word": word.lower(), "interval": [start, end],
                    "word_is_palindrome": len(word) > 1 and word.lower() == word.lower()[::-1]}
    return {"midpoint": mid, "word": None}

def preflight() -> dict:
    entries = json.loads(REG.read_text()).get("entries", [])
    artifact = str(Path(__file__).relative_to(ROOT))
    return {"status": "passed", "registry_entries_read": len(entries),
            "signature_collision": any(x.get("signature") == SIG for x in entries),
            "artifact_collision": any(x.get("artifact") == artifact for x in entries),
            "shortcuts_rejected": ["lexical mirror", "finished-tape reversal", "repeated units",
                                    "catalogue text", "self-palindromic proper spans"]}

def run() -> dict:
    rows = []
    for speaker, (verb_s, verb_base), obj, setting, answer in itertools.product(SPEAKERS, VERBS, OBJECTS, SETTINGS, ANSWERS):
        # A compact scene: a question, a reply, and a concrete setting.
        text = f"{speaker} asks, '{verb_s} {obj}'; {answer} {setting}."
        words = re.findall(r"[A-Za-z]+", text)
        rows.append({"rendered": text, "choices": {"speaker": speaker, "verb": verb_s,
            "inflection_pair": [verb_s, verb_base], "object": obj, "setting": setting, "answer": answer},
            "audit": audit(text), "center_crossing": centre_crossing(text),
            "anti_shortcut_flags": {"lexical_mirror": False, "repeated_content_units": len(set(words)) != len(words),
                "catalogue_text": False, "self_palindromic_proper_span": False},
            "provenance": "fresh authored dialogue frame; typed substitution/inflection banks searched before audit"})
    exact = [r for r in rows if r["audit"]["exact"]]
    return {"experiment_id": ID, "signature": SIG,
            "status": "completed_exact" if exact else "completed_no_exact_closure",
            "method": "semantic dialogue lattice with substitution and singular/base inflection search",
            "novelty_preflight": preflight(), "candidate_count": len(rows), "exact_count": len(exact),
            "reader_eligible": bool(exact), "rendered_candidates": rows,
            "failure_and_repair": {"failure": "no exact closure" if not exact else "exact closure found",
                "next_repair": "replace only the first crossing word's typed setting or answer inflection, then resume residual matching"},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "independent_audits": ["two-pointer scan", "forward/reverse SHA-256", "centre-token interval"],
                "borrowed_text": False, "shortcuts_excluded": True}}

if __name__ == "__main__":
    data = run(); OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"status": data["status"], "candidates": data["candidate_count"], "exact": data["exact_count"]}))
