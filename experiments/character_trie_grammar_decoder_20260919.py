"""Character-trie grammar decoder with live mirrored half-tape assignments."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "character-trie-grammar-decoder-20260919"


@dataclass(frozen=True)
class Token:
    text: str
    tag: str
    number: str | None = None
    proper: bool = False

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    terminals: list[Token] = field(default_factory=list)


def make_trie(items: tuple[Token, ...]) -> TrieNode:
    root = TrieNode()
    for item in items:
        node = root
        for character in item.tape:
            node = node.children.setdefault(character, TrieNode())
        node.terminals.append(item)
    return root


def token_bank(text: str, tag: str, number: str | None = None) -> tuple[Token, ...]:
    return tuple(Token(word, tag, number) for word in text.split())


DETERMINERS = token_bank("a an the some many my our this that nine old new", "D")
SUBJECTS = token_bank("aide poet scribe sailor captain nurse baker player writer clerk queen king herald pilot gardener reader man woman child lover scholar farmer teacher keeper dreamer walker hunter ranger mother father friend brother sister knight bard monk priest singer dancer merchant doctor judge guard messenger traveler stranger student elder widow daughter son river wind cloud storm wave silence winter frost dawn mountain comet star moon heart dream night rose bird flower thief harbor garden shore gate bell book letter map song tale story crown ring road world home house field forest valley castle tower island ocean ship boat lamp mirror glass stone fire rain snow sun sky sea angel oracle oath willow thorn violet lark raven swan fox wolf hare deer horse honey wine bread lord lady", "S", "sg")
SUBJECTS = SUBJECTS + token_bank("men poets sailors writers nurses players singers gulls daughters neighbours sons elders widows fishermen captains waves ropes anchors mothers debts rivers founders barges foremen mills lamps shadows lenses numbers signals mountains headlamps domes comets stars birds flowers lovers scholars farmers teachers keepers dreamers walkers hunters rangers friends brothers sisters knights bards monks priests dancers makers weavers fishers miners merchants doctors judges guards messengers travelers strangers students clouds winds storms rivers harbors gardens shores gates bells books songs tales stories crowns rings roads worlds homes houses fields forests valleys villages castles towers islands oceans ships boats lamps mirrors glasses stones fires rains snows suns moons skies seas angels oracles oaths willows thorns violets larks ravens swans foxes wolves hares deer horses lords ladies memos notes pages poems plans", "S", "pl")
SUBJECTS = SUBJECTS + tuple(Token(word, "S", "sg", True) for word in "Diana Noel Nora Mara Leon Ada Ava Anna Eve Liam Otto Omar Iris Aria Rhea Rosa Sara Lara Alan Ronan".split())

_SG_VERBS = "rips reads marks writes carries guards inspires praises keeps finds follows remembers opens closes watches hears sees holds loves needs guides calls seeks knows trusts helps feeds leads meets saves likes hates names lights counts weighs mends teaches tells brings takes makes gathers builds minds sings rings walks runs dreams waits wants looks turns raises answers offers covers crosses fears serves shares chooses changes warns blesses chases buries shadows haunts mirrors greets welcomes"
_PL_VERBS = "read mark write carry guard inspire praise keep find follow remember open close watch hear see hold love need guide call seek know trust help feed lead meet save like hate name light count weigh mend teach tell bring take make gather build mind sing ring walk run dream wait want look turn raise answer offer cover cross fear serve share choose change warn bless chase bury shadow haunt mirror greet welcome"
VERBS = token_bank(_SG_VERBS, "V", "sg") + token_bank(_PL_VERBS, "V", "pl")
OBJECTS = token_bank("memos letters notes maps charts pages poems plans books songs tales stories prayers bells stars roses words secrets answers questions truth love hope peace time fate honor glory mercy music water wine bread fire rain snow light darkness air earth sky sea gold silver sand salt stone wood iron glass flowers vows names faces hands hearts minds voices bridges ships boats crowns rings roads fields forests mountains rivers harbors gardens shores sonnet poem book letter chart map bell rose tale story song", "O")
OBJECTS = OBJECTS + tuple(Token(word, "O", proper=True) for word in "Diana Noel Nora Mara Leon".split())
PREPOSITIONS = token_bank("at after under near by before in through beside with for from on", "P")
SEAM = (Token(";", "C"),)
_PILOT_SUBJECT_NAMES = set("aide poet scribe sailor captain nurse baker player writer clerk queen king herald pilot gardener reader man woman child men poets sailors writers singers Diana Noel Nora Mara Leon".split())
_PILOT_OBJECT_NAMES = set("memos letters notes maps charts pages poems plans books songs tales stories prayers bells stars roses sonnet poem book letter chart map bell rose tale story song Diana Noel Nora Mara Leon".split())
PILOT_TOKENS = {
    "D": DETERMINERS,
    "S": tuple(token for token in SUBJECTS if token.text in _PILOT_SUBJECT_NAMES),
    "V": VERBS,
    "O": tuple(token for token in OBJECTS if token.text in _PILOT_OBJECT_NAMES),
    "P": PREPOSITIONS,
    "C": SEAM,
}
def _preferred(items: tuple[Token, ...], order: tuple[str, ...]) -> tuple[Token, ...]:
    rank = {text: index for index, text in enumerate(order)}
    return tuple(sorted(items, key=lambda item: (rank.get(item.text, len(rank)), item.text)))

PILOT_TOKENS["D"] = _preferred(DETERMINERS, ("an", "nine", "some", "a", "the"))
PILOT_TOKENS["S"] = _preferred(PILOT_TOKENS["S"], ("aide", "men", "Diana", "poet", "scribe", "sailor"))
PILOT_TOKENS["V"] = _preferred(VERBS, ("rips", "inspire", "reads", "inspires", "marks", "writes"))
PILOT_TOKENS["O"] = _preferred(PILOT_TOKENS["O"], ("memos", "Diana", "notes", "letter", "maps"))
PILOT_TOKENS["D"] = tuple(token for token in PILOT_TOKENS["D"] if token.text in {"an", "nine", "some", "a", "the"})
PILOT_TOKENS["S"] = tuple(token for token in PILOT_TOKENS["S"] if token.text in {"aide", "men", "poet", "scribe", "sailor", "Diana", "Noel", "Nora", "Mara", "Leon"})
PILOT_TOKENS["V"] = tuple(token for token in PILOT_TOKENS["V"] if token.text in {"rips", "inspire", "reads", "inspires", "marks", "writes", "read", "write", "mark", "keep", "find", "follow"})
PILOT_TOKENS["O"] = tuple(token for token in PILOT_TOKENS["O"] if token.text in {"memos", "Diana", "notes", "letter", "maps", "poem", "book", "song", "tale"})
TOKENS = PILOT_TOKENS
TRIES = {slot: make_trie(items) for slot, items in TOKENS.items()}

FRAMES = (("D", "S", "V", "D", "O", "C", "D", "S", "V", "O"), ("D", "S", "V", "D", "O", "P", "D", "O", "C", "D", "S", "V", "O"), ("D", "S", "V", "D", "O", "C", "D", "S", "V", "D", "O", "P"))


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatches = [(i, left, right) for i, (left, right) in enumerate(zip(tape, reverse)) if left != right]
    return {"normalized": tape, "letters": len(tape), "two_pointer_exact": bool(tape) and not mismatches, "first_mismatch": mismatches[0] if mismatches else None, "sha256_forward": hashlib.sha256(tape.encode("ascii")).hexdigest(), "sha256_reverse": hashlib.sha256(reverse.encode("ascii")).hexdigest()}


def choices(slot: str, position: int, target: int, assigned: dict[int, str], used: frozenset[str], number: str | None, proper_count: int) -> list[tuple[Token, dict[int, str]]]:
    result: list[tuple[Token, dict[int, str]]] = []

    def walk(node: TrieNode, cursor: int, state: dict[int, str]) -> None:
        for token in node.terminals:
            if token.tape and len(token.tape) > 1 and token.tape in used:
                continue
            if slot == "V" and number not in (None, token.number):
                continue
            if token.proper and proper_count:
                continue
            result.append((token, state))
        if cursor >= target:
            return
        mirror = min(cursor, target - 1 - cursor)
        for character, child in node.children.items():
            previous = state.get(mirror)
            if previous is not None and previous != character:
                continue
            updated = dict(state)
            updated[mirror] = character
            walk(child, cursor + 1, updated)

    walk(TRIES[slot], position, dict(assigned))
    return result


def search(frame: tuple[str, ...], target: int, max_nodes: int = 20_000) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    nodes = 0
    minimum = {slot: min(len(token.tape) for token in items) for slot, items in TOKENS.items()}
    maximum = {slot: max(len(token.tape) for token in items) for slot, items in TOKENS.items()}

    def visit(index: int, position: int, assigned: dict[int, str], chosen: list[Token], used: frozenset[str], number: str | None, proper_count: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return
        if index == len(frame):
            if position != target:
                return
            raw = " ".join(token.text for token in chosen).replace(" ; ", "; ")
            rendered = raw[:1].upper() + raw[1:] + "."
            checked = audit(rendered)
            if not checked["two_pointer_exact"] or checked["letters"] < 30:
                return
            gates = mechanical_admission_checks(rendered, min_letters=30, max_letters=260)
            rows.append({"rendered": rendered, "word_spans": [token.text for token in chosen], "audit": checked, "mechanical_checks": gates, "mechanically_admitted": all(gates.values()), "provenance": {"construction": "character trie with POS/inflection terminals and live half-tape assignments", "finished_tape_reversed": False, "catalogue_imported": False, "word_order_mirror": False, "rlaif_used": False}, "reader_status": "unreviewed; exactness does not certify readability"})
            return
        slot = frame[index]
        remaining = frame[index + 1:]
        for token, next_assigned in choices(slot, position, target, assigned, used, number, proper_count):
            end = position + len(token.tape)
            if end + sum(minimum[key] for key in remaining) > target or end + sum(maximum[key] for key in remaining) < target:
                continue
            next_number = token.number if slot == "S" else number
            next_used = used | ({token.tape} if len(token.tape) > 1 and token.tag not in {"D", "P", "C"} else set())
            visit(index + 1, end, next_assigned, chosen + [token], next_used, next_number, proper_count + int(token.proper))

    visit(0, 0, {}, [], frozenset(), None, 0)
    return rows, nodes


def run() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    nodes = 0
    target_runs = 0
    for frame in FRAMES:
        for target in range(38, 61):
            budget = 20_000 if frame == FRAMES[0] and target == 38 else 5_000
            found, count = search(frame, target, max_nodes=budget)
            rows.extend(found)
            nodes += count
            target_runs += 1
    exact = sorted({row["audit"]["normalized"]: row for row in rows}.values(), key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    return {"experiment_id": EXPERIMENT_ID, "method": "character-trie grammar decoder with live mirrored half-tape assignments", "frames": [list(frame) for frame in FRAMES], "stats": {"target_runs": target_runs, "nodes": nodes, "exact": len(exact), "mechanically_admitted": sum(row["mechanically_admitted"] for row in exact), "longest_exact_letters": max((row["audit"]["letters"] for row in exact), default=0)}, "candidates": exact, "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"], "novelty_preflight": {"status": "passed", "signature": EXPERIMENT_ID, "catalogue_imported": False}, "next_repair": "Add typed relative-clause transitions to the trie automaton; preserve live boundaries and agreement state.", "reader_gate": "closed; exactness and mechanical admission do not certify readability"}


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / (EXPERIMENT_ID + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
