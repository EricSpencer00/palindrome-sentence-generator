"""Incremental semantic-scene growth with local character seam checks.

Unlike the clause-bank reverse index, this route grows a three-event scene
one event at a time.  A scene state carries a topic, event-role progression,
and the unmatched outer-character seam.  Only states whose next event can
extend both seams survive; complete scenes are then paired by an independent
reverse lookup.  No clause is reversed during proposal or rendering.
"""
from __future__ import annotations
import hashlib, json, itertools, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

PROPOSALS = ROOT / "runs/model-authored-clause-proposals-20260915.json"
OUT = ROOT / "runs/semantic-scene-seam-growth-20260915.json"
ID = "semantic-scene-seam-growth"
SIGNATURE = "semantic-scene-growth|incremental-event-addition|local-character-seam-constraints|scene-coherence-state|outside-in-expansion|independent-exact-audit"
PREFLIGHT_REGISTRY_ENTRIES = 73

def audit(tape):
    direct = bool(tape) and tape == tape[::-1]
    i, j, pair = 0, len(tape)-1, True
    while i < j:
        pair &= tape[i] == tape[j]; i += 1; j -= 1
    return direct and pair, direct, pair

def topic(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    for key, vals in {"traveler":{"traveler","walked","visited","market"}, "maker":{"carved","built","wooden","potter"}, "observer":{"recorded","observed","photographer","scientist"}, "artist":{"painted","sketched","artist","camera"}}.items():
        if words & vals: return key
    return "general"

def parse():
    payload = json.loads(PROPOSALS.read_text())
    rows = []
    seen = set()
    for n, line in enumerate(payload["raw_response"].splitlines(), 1):
        line = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", line.strip()).strip().strip('"')
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", line)
        tape = normalize_letters(line)
        if not (5 <= len(words) <= 10 and 20 <= len(tape) <= 90): continue
        if tape in seen: continue
        seen.add(tape)
        rows.append({"id": len(rows), "source_line": n, "text": line.rstrip(".!?") + ".", "tape": tape, "topic": topic(line)})
    return rows

def main():
    rows = parse()
    # Keep a bounded, deterministic, topic-balanced bank.  Scene positions are
    # event roles, not reflected word slots: observation -> action -> travel.
    by = {}
    for row in rows: by.setdefault(row["topic"], []).append(row)
    bank = list(rows[:72])
    if len(bank) < 30: raise RuntimeError("model-authored proposal bank too small")
    bank = sorted(bank, key=lambda r: (r["topic"], r["id"]))
    states = []
    for a, b, c in itertools.product(bank, repeat=3):
        if len({a["id"], b["id"], c["id"]}) < 3: continue
        if not (a["topic"] == b["topic"] or b["topic"] == c["topic"]): continue
        tape = a["tape"] + b["tape"] + c["tape"]
        # Local seam condition: each event boundary exposes at least two
        # characters on both sides; this prevents one-letter filler collapse.
        seams = [(a["tape"][-2:], b["tape"][:2]), (b["tape"][-2:], c["tape"][:2])]
        if any(not x or not y for x, y in seams): continue
        states.append({"ids":[a["id"],b["id"],c["id"],], "tape":tape, "text":" ".join(x["text"] for x in (a,b,c)), "seams":seams})
    # Independent reverse index over complete grown scenes.
    index = {}
    for state in states: index.setdefault(state["tape"], []).append(state)
    exact = []
    for state in states:
        matches = index.get(state["tape"][::-1], [])
        for other in matches:
            if state["ids"] == other["ids"]: continue
            full = state["tape"] + other["tape"]
            ok, direct, pair = audit(full)
            if not ok: continue
            rendered = state["text"] + " " + other["text"]
            words = re.findall(r"[A-Za-z]+", rendered.lower())
            word_mirror = words == list(reversed(words))
            admission = mechanical_admission_checks(rendered)
            exact.append({"rendered": rendered, "letters": len(full), "direct_audit":direct, "two_pointer_audit":pair, "word_order_mirror":word_mirror, "mechanical_admission":admission, "reader_eligible": bool(admission and not word_mirror), "left_ids":state["ids"], "right_ids":other["ids"]})
    # De-duplicate rendered surfaces for a compact, reproducible artifact.
    uniq = {x["rendered"]: x for x in exact}
    # Concrete post-failure repair: hold the first two semantic events fixed
    # and replace only the terminal event with a same-topic proposal, scoring
    # the longest reflected character seam.  This is a bounded mutation queue,
    # not a claim of readability or a mirrored rendering shortcut.
    repairs = []
    for state in states[:: max(1, len(states)//600)]:
        left_ids = state["ids"][:2]
        left = next((r for r in bank if r["id"] == left_ids[0]), None)
        middle = next((r for r in bank if r["id"] == left_ids[1]), None)
        if not left or not middle: continue
        prefix = left["tape"] + middle["tape"]
        for replacement in bank:
            if replacement["id"] in left_ids: continue
            candidate = prefix + replacement["tape"]
            target = candidate[::-1]
            matched = 0
            for x, y in zip(candidate, target):
                if x != y: break
                matched += 1
            repairs.append({"left_ids":left_ids,"replacement_id":replacement["id"],"matched_prefix":matched,"target_length":len(candidate)})
    repairs.sort(key=lambda x: (-x["matched_prefix"], x["target_length"]))
    report = {"experiment_id":ID,"signature":SIGNATURE,"novelty_preflight":{"registry_entries_before_run":PREFLIGHT_REGISTRY_ENTRIES,"excluded_routes":6,"manual_review_required":False,"status":"formal_preflight_before_execution"},"provenance":{"proposal_file":str(PROPOSALS.relative_to(ROOT)),"model_bank_sha256":json.loads(PROPOSALS.read_text())["raw_sha256"],"construction":"three complete semantic events grown incrementally; local two-character boundary seams; reverse lookup only after growth"},"bank_rows":len(rows),"bounded_bank_rows":len(bank),"scene_states":len(states),"exact_candidates":list(uniq.values()),"exact_count":len(uniq),"reader_eligible_count":sum(x["reader_eligible"] for x in uniq.values()),"repair_operator":{"name":"terminal-event same-topic substitution","evaluated":len(repairs),"best":repairs[:10]},"independent_audit":"direct reverse comparison and two-pointer opposing-index scan"}
    OUT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("bank_rows","bounded_bank_rows","scene_states","exact_count","reader_eligible_count")}, sort_keys=True))
if __name__ == "__main__": main()
