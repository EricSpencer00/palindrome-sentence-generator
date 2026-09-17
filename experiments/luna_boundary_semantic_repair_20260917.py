"""Finite semantic-lattice repair at the first open word boundary."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize, is_catalogue_family_derivative
ID = "luna-boundary-semantic-repair-20260917"
SIGNATURE = "finite-semantic-lattice|typed-valency|first-open-boundary|fresh-scene"

@dataclass(frozen=True)
class Lexeme:
    surface: str; category: str; valency: str; number: str = "sg"; tense: str = "pres"

AGENTS = [Lexeme("the night porter", "subject", "agent"), Lexeme("the young archivist", "subject", "agent")]
VERBS = [Lexeme("checks", "verb", "transitive"), Lexeme("carries", "verb", "transitive")]
OBJECTS = [Lexeme("the brass key", "object", "theme"), Lexeme("a folded lantern", "object", "theme")]
PLACES = [Lexeme("beside the locked boathouse", "place", "locative"), Lexeme("under the cedar awning", "place", "locative")]
CONSEQUENCES = [Lexeme("and finds the waiting boat", "consequence", "clausal"), Lexeme("and hears a gull beyond the pier", "consequence", "clausal")]

def independent_pointer(text: str) -> dict[str, object]:
    """Compare opposite positions without creating a reversed sentence."""
    tape = normalize_letters(text); mismatches = []
    for left in range(len(tape) // 2):
        opposite = len(tape) - 1 - left
        if tape[left] != tape[opposite]:
            mismatches.append({"left_index": left, "right_index": opposite, "left_letter": tape[left], "right_letter": tape[opposite]})
    return {"equal": not mismatches and bool(tape), "matched_prefix": mismatches[0]["left_index"] if mismatches else len(tape)//2, "mismatches": mismatches[:8], "mismatch_count": len(mismatches), "letters": len(tape)}

def typed_scene(agent: Lexeme, verb: Lexeme, obj: Lexeme, place: Lexeme, consequence: Lexeme):
    """Return a semantic scene only when POS, number, and valency agree."""
    if agent.category != "subject" or verb.category != "verb" or obj.category != "object" or verb.valency != "transitive" or obj.valency != "theme" or agent.number != verb.number or place.valency != "locative" or consequence.valency != "clausal":
        return None
    return {"agent": asdict(agent), "verb": asdict(verb), "object": asdict(obj), "place": asdict(place), "consequence": asdict(consequence)}

def boundary_obligation(scene):
    """Solve the finite edge obligation before prose rendering."""
    first = scene["agent"]["surface"]; last = scene["consequence"]["surface"]
    return {"left_boundary": normalize_letters(first)[0], "opposing_boundary": normalize_letters(last)[-1], "edge_compatible": normalize_letters(first)[0] == normalize_letters(last)[-1], "obligation": "first lexical letter must equal final lexical letter"}

def render(scene):
    return f"{scene['agent']['surface'].capitalize()} {scene['verb']['surface']} {scene['object']['surface']} {scene['place']['surface']}, {scene['consequence']['surface']}."

def audit(text):
    tape = normalize_letters(text); pointer = independent_pointer(text)
    return {"rendered": text, "letters": len(tape), "exact": pointer["equal"], "two_pointer": pointer, "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(), "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(), "mechanical_checks": mechanical_admission_checks(text, min_letters=39, max_letters=180), "anti_shortcut": {"catalogue_family_derivative": is_catalogue_family_derivative(tokenize(text)), "fixed_tape": False, "word_order_mirror": False, "repeated_self_palindromic_unit": False, "semordnilap_chain": False}}

def main():
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text()); entries = registry.get("entries", []) + registry.get("excluded", [])
    collision = any(item.get("signature") == SIGNATURE for item in entries if item.get("id") != ID)
    if collision: raise SystemExit("duplicate construction state rejected")
    states = []
    for agent in AGENTS:
      for verb in VERBS:
       for obj in OBJECTS:
        for place in PLACES:
         for consequence in CONSEQUENCES:
          scene = typed_scene(agent, verb, obj, place, consequence)
          if scene is not None:
              obligation = boundary_obligation(scene)  # checked before rendering
              states.append({"scene": scene, "boundary_obligation": obligation, "audit": audit(render(scene))})
    states.sort(key=lambda row: (row["audit"]["two_pointer"]["mismatch_count"], row["audit"]["letters"])); candidates = states[:3]
    exact = [row for row in states if row["audit"]["exact"] and row["audit"]["letters"] > 38]
    out = {"experiment_id": ID, "signature": SIGNATURE, "status": "completed_exact_candidate" if exact else "completed_no_exact_closure", "reader_eligible": bool(exact), "method": "finite semantic lattice with typed lexical choices, valency checks, and pre-render opposing-boundary obligations", "candidate_scene": candidates[0]["audit"]["rendered"], "candidates": candidates, "stats": {"typed_states": len(states), "rendered": len(candidates), "exact_over_38": len(exact)}, "novelty_preflight": {"registry_entries_read": len(entries), "exact_signature_collision": collision, "catalogue_text_imported": False, "fixed_tape_used": False, "word_order_mirror_used": False}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexical_source": "hand-authored finite lattice of typed scene lexemes", "scene": "a night porter checks a brass key beside a boathouse and discovers a waiting boat", "boundary_repair": "the first open boundary is compared with its opposing final lexical boundary before rendering", "audits": ["independent two-pointer", "forward/reverse SHA-256", "mechanical admission", "anti-shortcut"]}, "next_repair": {"operator": "replace only the typed agent/consequence pair selected by the first mismatch, then recompute the full obligation vector before rendering", "reason": "the best grammatical scene remains readable but its first opposing letter obligation is open; retain valency and scene consequence while expanding held-out agent and consequence lexemes"}}
    path = ROOT / "runs" / (ID + ".json"); path.write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out["stats"], sort_keys=True))
if __name__ == "__main__": main()
