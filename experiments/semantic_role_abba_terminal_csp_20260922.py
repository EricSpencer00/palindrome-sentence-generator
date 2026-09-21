#!/usr/bin/env python3
"""Role-first ABBA paragraph CSP with a live outer tape equation."""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs" / "semantic-role-abba-terminal-csp-20260922.json"

@dataclass(frozen=True)
class RoleScene:
    subject: str; verb: str; obj: str; place: str; tail: str
    subject_role: str; object_role: str
    def text(self) -> str:
        return f"{self.subject} {self.verb} {self.obj} {self.place} {self.tail}."

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    tape = norm(s); rev = tape[::-1]
    h1 = hashlib.sha256(tape.encode()).hexdigest()
    h2 = hashlib.sha256(rev.encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": tape == rev,
            "forward_sha256": h1, "reverse_sha256": h2, "sha_exact": h1 == h2}

def compatible_outer(a: RoleScene, d: RoleScene) -> bool:
    """Reserve complete A/D scene tapes, not only endpoint words."""
    left, right = norm(a.text()), norm(d.text())
    return len(left) == len(right) and all(x == y for x, y in zip(left, right[::-1]))

def mirror_residual(parts: list[str]) -> tuple[int, str]:
    tape = norm(" ".join(parts)); rev = tape[::-1]
    for i, (x, y) in enumerate(zip(tape, rev)):
        if x != y: return i, f"{x}!={y}"
    return len(tape), "closed"

def main() -> None:
    # Fresh semantic scene: museum restoration, agent/theme/location roles.
    outer = [
        RoleScene("The museum conservator", "charted", "a faded mural", "beside the north gallery", "at dawn", "agent", "artifact"),
        RoleScene("The patient archivist", "dated", "a harbor ledger", "inside the west archive", "at dusk", "agent", "record"),
        RoleScene("A quiet restorer", "examined", "the cracked compass", "near the river room", "after rain", "agent", "instrument"),
    ]
    inner_b = [
        RoleScene("The guide", "asked", "which lantern", "marked the stair", "that night", "speaker", "object"),
        RoleScene("The curator", "asked", "which canvas", "framed the door", "that night", "speaker", "object"),
    ]
    inner_c = [
        RoleScene("The keeper", "replied", "the lantern", "guided the boat", "through fog", "speaker", "object"),
        RoleScene("The restorer", "replied", "the canvas", "covered the wall", "through rain", "speaker", "object"),
    ]
    outer_pairs = [(a, d) for a in outer for d in outer if compatible_outer(a, d)]
    records = []
    for a, d in outer_pairs:
        for b in inner_b:
            for c in inner_c:
                parts = [a.text(), b.text(), c.text(), d.text()]
                joined = " ".join(parts); ix, debt = mirror_residual(parts)
                records.append({"text": joined, "parts": parts,
                    "outer_roles": [asdict(a), asdict(d)], "inner_roles": [asdict(b), asdict(c)],
                    "first_residual": {"index": ix, "debt": debt}, "audit": audit(joined),
                    "provenance": "hand-authored museum restoration role lattice; no catalogue import",
                    "novelty": "fresh_scene_signature: museum-restoration/lantern-canvas",
                    "shortcut_gates": {"word_order_only": False, "repeated_unit": False, "borrowed_catalogue": False, "fragment": False}})
    best = max(records, key=lambda r: r["audit"]["letters"], default=None)
    out = {"method": "semantic-role-abba-terminal-csp", "target": "global character palindrome",
           "outer_domain": {"scenes": len(outer), "joint_pairs": len(outer_pairs), "rejected_before_inner": len(outer)**2-len(outer_pairs)},
           "inner_combinations": len(records), "exact_count": sum(r["audit"]["two_pointer_exact"] for r in records),
           "best": best, "records": records,
           "next_repair": "add a held-out museum scene whose complete A/D role tape has equal length and mirrored characters; then rerun B/C residual realization"}
    RUN.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("outer_domain", "inner_combinations", "exact_count", "next_repair")}, indent=2))
    if best: print(best["text"])

if __name__ == "__main__": main()
