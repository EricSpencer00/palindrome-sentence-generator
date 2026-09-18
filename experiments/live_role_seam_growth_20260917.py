"""Live role-conditioned seam growth; obligations are consumed before prose is emitted."""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[a-z]+")
ROLES = {
    "agent": ["the fox", "a child", "the sailor", "a teacher", "the baker", "the nurse"],
    "action": ["finds", "reads", "marks", "opens", "carries", "helps"],
    "object": ["a den", "the book", "a map", "the door", "fresh bread", "the patient"],
    "place": ["at dawn", "by water", "near shore", "at noon", "to town", "in spring"],
}
CLAUSES = [
    ("the", "agent"), ("fox", "agent"), ("finds", "action"), ("a", "object"),
    ("den", "object"), ("at", "place"), ("dawn", "place"),
]

def tape(s): return "".join(WORD.findall(s.lower()))

def audit(s):
    t = tape(s); r = t[::-1]
    mismatches = [i for i, (a, b) in enumerate(zip(t, r)) if a != b]
    return {"letters": len(t), "exact": bool(t) and not mismatches,
            "two_pointer": bool(t) and all(t[i] == t[-1-i] for i in range(len(t)//2)),
            "sha_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha_reverse": hashlib.sha256(r.encode()).hexdigest(),
            "mismatches": mismatches[:12]}

def grow():
    # Each state contains actual lexical material on the left and a live suffix
    # obligation. Alternatives are role-compatible and tested at every offset.
    states = [{"words": [], "roles": [], "obligation": "", "tested": 0}]
    traces = []
    for offset, (required_word, role) in enumerate(CLAUSES):
        nxt = []
        for state in states:
            for phrase in ROLES[role]:
                words = phrase.split()
                left = " ".join(state["words"] + words)
                obligation = tape(left)[::-1]
                # A right-side role phrase must consume the current obligation;
                # no completed-tape mirroring is allowed.
                matches = [(rr, rp) for rr, bank in ROLES.items() for rp in bank
                           if obligation.startswith(tape(rp)) or tape(rp).startswith(obligation)]
                tested = state["tested"] + len(ROLES[role])
                if matches:
                    nxt.append({"words": state["words"] + words, "roles": state["roles"] + [role],
                                "obligation": obligation, "tested": tested})
                elif offset == 0:
                    traces.append({"offset": offset, "role": role, "left": left,
                                   "required_prefix": obligation[:16], "alternatives_tested": tested,
                                   "status": "dead_live_seam"})
        states = nxt[:256]
        if not states: break
    candidates = []
    # A candidate is only rendered if a live right phrase fully consumes the tape.
    for s in states:
        for bank in ROLES.values():
            for right in bank:
                text = " ".join(s["words"] + right.split())
                a = audit(text)
                if a["exact"] and len(set(WORD.findall(text))) == len(WORD.findall(text)):
                    candidates.append({"text": text, "audit": a, "provenance": "fresh role lattice"})
    return {"experiment_id": "live-role-seam-growth-20260917",
            "status": "completed_exact" if candidates else "quarantined_no_closure",
            "candidates": candidates, "traces": traces,
            "provenance": {"source": "fresh typed role phrases", "catalogue_imported": False,
                           "finished_mirroring": False, "independent_audits": ["two-pointer", "SHA-256"]},
            "novelty_preflight": {"duplicate_sweep": False, "method": "live role alternatives consume reverse obligations"},
            "failure_and_repair": {"next_repair": "add inflected role phrases whose full tape consumes the next obligation, then re-score attachment and agreement"}}

if __name__ == "__main__":
    out = grow(); (ROOT / "runs/live-role-seam-growth-20260917.json").write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out, indent=2))
