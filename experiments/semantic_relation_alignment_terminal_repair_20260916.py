"""Terminal-span and odd-center repair for semantic relation alignment.

This is a bounded repair of ``semantic_relation_alignment_20260916``.  It
changes two declared state components: the terminal role can be an
independently lexicalized phrase span, and one unpaired character may be
consumed as an odd-length centre.  The relation edges and lexical-role
semantics remain the same.  Search is deterministic exhaustive recursion over
whole lexical units; it does not use a beam, MCTS, chart, CSP, ILP, or a
catalogue of palindromes.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EVIDENCE = ROOT / "runs" / "semantic-relation-alignment-terminal-repair-20260916.json"
EXPERIMENT_ID = "semantic-relation-alignment-terminal-repair-20260916"
SIGNATURE = (
    "semantic-relation-alignment|directed-event-edge-pairing|"
    "independent-role-lexicalization|boundary-synchronous-character-ledger|"
    "deterministic-exhaustive|terminal-compatible-phrase-spans|"
    "bounded-odd-center"
)

FRAME_EDGES = {
    "preserve_then_recover": ("preserves", "enables"),
    "signal_then_respond": ("signals", "elicits"),
    "plant_then_grow": ("plants", "causes"),
    "measure_then_adjust": ("measures", "guides"),
}

FRAMES = {
    "preserve_then_recover": {
        "left": ("det", "agent", "verb", "det", "artifact", "prep", "place"),
        "right": ("det", "agent_r", "verb_r", "det", "artifact_r", "prep", "terminal_span"),
    },
    "signal_then_respond": {
        "left": ("det", "agent_s", "verb_s", "det", "message", "prep", "place"),
        "right": ("det", "agent_a", "verb_a", "det", "message_a", "prep", "terminal_span"),
    },
    "plant_then_grow": {
        "left": ("det", "agent_p", "verb_p", "det", "crop", "prep", "place"),
        "right": ("det", "agent_g", "verb_g", "det", "crop_g", "prep", "terminal_span"),
    },
    "measure_then_adjust": {
        "left": ("det", "agent_m", "verb_m", "det", "quantity", "prep", "place"),
        "right": ("det", "agent_x", "verb_x", "det", "setting", "prep", "terminal_span"),
    },
}

# These pools are intentionally finite and independent on the two sides.  A
# terminal span is one lexical construction unit while retaining its internal
# word boundary in the rendered surface.
W = {
    "det": ("a", "an", "the", "our", "my", "one", "no"),
    "agent": ("baker", "farmer", "keeper", "maker", "pilot", "poet", "teacher", "worker"),
    "agent_r": ("clerk", "doctor", "editor", "nurse", "reader", "sailor", "worker"),
    "verb": ("keeps", "guards", "holds", "stores", "saves", "packs", "covers"),
    "verb_r": ("finds", "reads", "takes", "gets", "opens", "uses", "sees"),
    "artifact": ("map", "note", "record", "seed", "token", "letter", "paper", "book"),
    "artifact_r": ("answer", "memo", "message", "parcel", "signal", "story", "record", "token"),
    "prep": ("in", "on", "at", "by", "near", "over", "under", "from", "with"),
    "place": ("area", "villa", "cinema", "home", "harbor", "garden", "market", "office", "river", "road", "room", "tower", "camp", "city", "bay", "port", "field", "hall", "cave", "shore"),
    "terminal_span": ("area", "old arena", "near area", "at opera", "in plaza", "home port", "river bank", "garden wall", "market hall", "office tower", "city road", "harbor side"),
    "agent_s": ("caller", "child", "clerk", "farmer", "friend", "guide", "maker", "pilot", "poet", "teacher"),
    "verb_s": ("asks", "calls", "sends", "tells", "shows", "warns", "marks", "names"),
    "message": ("alarm", "answer", "note", "news", "signal", "story", "word", "warning", "letter"),
    "agent_a": ("artist", "caller", "child", "doctor", "friend", "guide", "keeper", "reader", "sailor", "teacher"),
    "verb_a": ("answers", "comes", "helps", "listens", "reads", "replies", "returns", "speaks", "waits"),
    "message_a": ("answer", "reply", "help", "news", "signal", "story", "word", "warning", "letter"),
    "agent_p": ("farmer", "gardener", "keeper", "maker", "worker", "child", "poet", "teacher"),
    "verb_p": ("sows", "plants", "grows", "tends", "keeps", "makes", "starts", "brings"),
    "crop": ("beans", "corn", "grain", "herbs", "seeds", "trees", "vines", "wheat"),
    "agent_g": ("farmer", "gardener", "keeper", "worker", "child", "maker", "teacher"),
    "verb_g": ("grows", "rises", "spreads", "thrives", "stands", "blossoms", "ripens"),
    "crop_g": ("beans", "crops", "flowers", "grain", "leaves", "plants", "trees", "vines"),
    "agent_m": ("doctor", "farmer", "maker", "pilot", "reader", "teacher", "worker", "writer"),
    "verb_m": ("tests", "checks", "counts", "marks", "reads", "rates", "weighs", "measures"),
    "quantity": ("cost", "depth", "height", "length", "weight", "width", "value", "rate"),
    "agent_x": ("doctor", "farmer", "maker", "pilot", "worker", "writer", "keeper", "teacher"),
    "verb_x": ("changes", "clears", "moves", "raises", "resets", "sets", "shifts", "tunes"),
    "setting": ("level", "angle", "balance", "height", "volume", "width", "value", "rate"),
}


def letters(value: str) -> str:
    return normalize_letters(value)


def _audit(left: tuple[str, ...], right: tuple[str, ...], edge: tuple[str, str]) -> dict:
    rendered = " ".join(left) + ". " + " ".join(right) + "."
    tape = normalize_letters(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=220)
    return {
        "rendered": rendered,
        "edge": list(edge),
        "left": list(left),
        "right": list(right),
        "letters": len(tape),
        "tape": tape,
        "exact": bool(tape) and tape == tape[::-1],
        "ledger_replay": bool(tape) and all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "checks": checks,
        "admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
    }


def search_frame(frame_name: str, spec: dict, state_budget: int = 100_000) -> tuple[list[dict], dict]:
    left_slots, right_slots = spec["left"], spec["right"]
    solutions: list[dict] = []
    stats = {"states": 0, "char_pairs": 0, "dead_char": 0, "lexical_assignments": 0, "terminal_span_trials": 0, "odd_center_trials": 0, "max_matched": 0, "budget_exhausted": False}

    def visit(li: int, ri: int, lp: int, rp: int, left: tuple[str, ...], right_rev: tuple[str, ...], trace: tuple[dict, ...]) -> None:
        stats["states"] += 1
        if stats["states"] > state_budget:
            stats["budget_exhausted"] = True
            return
        if len(solutions) >= 120:
            return
        # Both grammars have selected every slot and every selected unit is empty.
        if li == len(left_slots) and ri == len(right_slots) and lp == 0 and rp == 0:
            row = _audit(left, tuple(reversed(right_rev)), FRAME_EDGES[frame_name])
            if row["exact"]:
                row["trace"] = list(trace)
                solutions.append(row)
            return
        # Move across a completed lexical unit.
        if li < len(left_slots) and li < len(left) and lp == len(letters(left[-1])):
            visit(li + 1, ri, 0, rp, left, right_rev, trace)
            return
        if ri < len(right_slots) and ri < len(right_rev) and rp == len(letters(right_rev[-1])):
            visit(li, ri + 1, lp, 0, left, right_rev, trace)
            return
        # Select the opposite terminal unit as soon as a left unit exists.
        if ri < len(right_slots) and ri >= len(right_rev) and left:
            slot = right_slots[len(right_slots) - 1 - ri]
            options = W[slot]
            if slot == "terminal_span":
                stats["terminal_span_trials"] += len(options)
            for word in options:
                stats["lexical_assignments"] += 1
                visit(li, ri, lp, 0, left, right_rev + (word,), trace + ({"side": "right", "slot": slot, "unit": word, "edge": FRAME_EDGES[frame_name][1]},))
            return
        if li < len(left_slots) and li >= len(left):
            slot = left_slots[li]
            for word in W[slot]:
                stats["lexical_assignments"] += 1
                visit(li, ri, 0, rp, left + (word,), right_rev, trace + ({"side": "left", "slot": slot, "unit": word, "edge": FRAME_EDGES[frame_name][0]},))
            return
        if ri < len(right_slots) and ri >= len(right_rev):
            slot = right_slots[len(right_slots) - 1 - ri]
            for word in W[slot]:
                stats["lexical_assignments"] += 1
                visit(li, ri, lp, 0, left, right_rev + (word,), trace + ({"side": "right", "slot": slot, "unit": word, "edge": FRAME_EDGES[frame_name][1]},))
            return
        # If one side is complete, at most one character may remain as centre.
        if li >= len(left_slots) or ri >= len(right_slots):
            stats["odd_center_trials"] += 1
            if li >= len(left_slots) and ri < len(right_slots) and ri < len(right_rev):
                remaining = len(letters(right_rev[-1])) - rp
                if remaining == 1:
                    row = _audit(left, tuple(reversed(right_rev)), FRAME_EDGES[frame_name])
                    if row["exact"]:
                        row["trace"] = list(trace) + [{"center": letters(right_rev[-1])[::-1][rp]}]
                        solutions.append(row)
                return
            if ri >= len(right_slots) and li < len(left_slots) and li < len(left):
                remaining = len(letters(left[-1])) - lp
                if remaining == 1:
                    row = _audit(left, tuple(reversed(right_rev)), FRAME_EDGES[frame_name])
                    if row["exact"]:
                        row["trace"] = list(trace) + [{"center": letters(left[-1])[lp]}]
                        solutions.append(row)
                return
            return
        left_chars = letters(left[-1])
        right_chars = letters(right_rev[-1])[::-1]
        if lp >= len(left_chars) or rp >= len(right_chars):
            return
        if left_chars[lp] != right_chars[rp]:
            stats["dead_char"] += 1
            return
        stats["char_pairs"] += 1
        stats["max_matched"] = max(stats["max_matched"], sum(1 for item in trace if "char" in item) + 1)
        visit(li, ri, lp + 1, rp + 1, left, right_rev, trace + ({"char": left_chars[lp], "left_offset": lp, "right_offset": rp},))

    visit(0, 0, 0, 0, (), (), ())
    return solutions, stats


def novelty_preflight() -> dict:
    registry = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())["entries"]
    common = {"a", "an", "and", "after", "audit", "authoring", "before", "character", "complete", "construction", "constraints", "cross", "derived", "derivation", "english", "equation", "exact", "final", "full", "generation", "global", "grammar", "held", "heldout", "in", "independent", "join", "joint", "left", "lexical", "lexicalized", "of", "order", "out", "over", "paired", "parse", "parser", "repair", "residual", "reverse", "right", "sentence", "state", "surface", "tape", "the", "through", "to", "typed", "unit", "word", "words", "with"}
    atoms = lambda value: set(re.findall(r"[a-z0-9]+", value.lower())) - common
    current = atoms(SIGNATURE)
    nearest = []
    for row in registry:
        prior = atoms(row["signature"])
        score = len(current & prior) / len(current | prior)
        nearest.append({"id": row["id"], "jaccard": round(score, 6), "shared_atoms": sorted(current & prior)})
    nearest.sort(key=lambda row: (-row["jaccard"], row["id"]))
    return {"registry_entries": len(registry), "excluded_routes": len(json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text()).get("excluded", [])), "exact_signature_collision": any(row["signature"] == SIGNATURE for row in registry), "nearest_prior": nearest[:3]}


def run() -> dict:
    rows: list[dict] = []
    per_frame: dict[str, dict] = {}
    for frame_name, spec in FRAMES.items():
        found, stats = search_frame(frame_name, spec)
        rows.extend(found)
        per_frame[frame_name] = stats
    admitted = [row for row in rows if row["admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_repair_no_reader_promotion",
        "repair_of": "semantic-relation-alignment-20260916",
        "method": "Same directed event-edge relation state as the base route, repaired with independently lexicalized terminal phrase spans and a bounded odd-length centre. The exhaustive boundary ledger and final mechanical audit are unchanged.",
        "novelty_preflight": novelty_preflight(),
        "config": {"frames": FRAME_EDGES, "min_letters": 39, "terminal_span_count": len(W["terminal_span"]), "odd_center": "at most one unmatched character, rechecked by full tape reversal", "state_budget_per_frame": 100000, "search_control": "deterministic exhaustive recursion; no beam, MCTS, chart, CSP, ILP, or catalogue input"},
        "stats": {"frames": len(FRAMES), "rows": len(rows), "exact": sum(row["exact"] for row in rows), "admitted": len(admitted), "max_letters": max((row["letters"] for row in rows), default=0), "per_frame": per_frame},
        "rows": rows[:120],
        "provenance": {"catalogue_imported": False, "source_sentences_copied": False, "reader_evidence": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }


def main() -> None:
    if EVIDENCE.exists():
        raise SystemExit(f"refusing to overwrite existing output: {EVIDENCE}")
    payload = run()
    EVIDENCE.parent.mkdir(exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": payload["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
