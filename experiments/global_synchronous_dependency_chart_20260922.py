"""Globally intersect two finite-clause dependency parses with a palindrome.

This is the successor to the retired local ``r=s`` stack.  It does not start
from a seed, a completed phrase, or a finished tape.  Two independently
directed dependency automata choose Brown-attested, WordNet-backed terminals
through character tries.  The left clause advances in reading order while
the right clause advances from its outside edge.  Every chart transition
therefore consumes the live character debt before either clause is complete.

The packed chart key carries both partial parses (heads, open valencies,
agreement, tense, and discourse referents), both token-boundary masks, the
character owner/residual/cursors, and global lemma freshness.  Paths may pack
only when all of those future-relevant values agree.  Complete strings are
rendered only after both finite clauses close and every central mechanical
gate is checked.  A zero-result run records an exhausted-chart obstruction,
not a proposed fragment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (  # noqa: E402
    mechanical_admission_checks,
    normalize_letters,
)


EXPERIMENT_ID = "global-synchronous-dependency-chart-20260922"
DEFAULT_OUTPUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
SIGNATURE = (
    "global-synchronous-dependency-chart|dual-finite-clause-dependency-automata|"
    "brown-wordnet-character-tries|packed-live-residual|dual-boundary-masks|"
    "discourse-referents"
)
WORD = re.compile(r"^[a-z]+$")
TRANSITIVE_FRAMES = frozenset(
    {5, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24, 25, 26, 30, 31}
)
NOUN_TAGS = frozenset({"nn", "nns"})
VERB_TAGS = frozenset({"vb", "vbz", "vbd"})
CONTENT_SLOTS = frozenset({"subject", "verb", "object"})


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def tag_base(raw: str) -> str:
    return raw.casefold().split("-", 1)[0].split("+", 1)[0].rstrip("*")


def wordnet_lines(wordnet_zip: Path, member: str):
    with zipfile.ZipFile(wordnet_zip) as archive:
        with archive.open(member) as handle:
            for raw in handle:
                yield raw.decode("utf-8").rstrip("\n")


def load_exceptions(wordnet_zip: Path, pos: str) -> dict[str, tuple[str, ...]]:
    values = {}
    for line in wordnet_lines(wordnet_zip, f"wordnet/{pos}.exc"):
        fields = line.split()
        if fields:
            values[fields[0]] = tuple(fields[1:])
    return values


def load_wordnet(wordnet_zip: Path) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[str, ...]], dict]:
    """Load verb frames and noun domains directly from the WordNet archive."""
    lexnames = {}
    for line in wordnet_lines(wordnet_zip, "wordnet/lexnames"):
        fields = line.split()
        if len(fields) >= 2:
            lexnames[int(fields[0])] = fields[1]

    noun_domains: dict[str, set[str]] = defaultdict(set)
    for line in wordnet_lines(wordnet_zip, "wordnet/data.noun"):
        if not line or line.startswith("  "):
            continue
        fields = line.split("|", 1)[0].split()
        count = int(fields[3], 16)
        cursor = 4
        domain = lexnames[int(fields[1])]
        for _ in range(count):
            noun_domains[fields[cursor].casefold()].add(domain)
            cursor += 2

    verb_frames: dict[str, set[int]] = defaultdict(set)
    for line in wordnet_lines(wordnet_zip, "wordnet/data.verb"):
        if not line or line.startswith("  "):
            continue
        fields = line.split("|", 1)[0].split()
        count = int(fields[3], 16)
        cursor = 4
        lemmas = []
        for _ in range(count):
            lemmas.append(fields[cursor].casefold())
            cursor += 2
        pointer_count = int(fields[cursor])
        cursor += 1 + 4 * pointer_count
        if cursor >= len(fields):
            continue
        frame_count = int(fields[cursor])
        cursor += 1
        per_word: dict[int, set[int]] = defaultdict(set)
        for _ in range(frame_count):
            if cursor + 2 >= len(fields) or fields[cursor] != "+":
                break
            frame, word_number = int(fields[cursor + 1]), int(fields[cursor + 2], 16)
            per_word[word_number].add(frame)
            cursor += 3
        for index, lemma in enumerate(lemmas, 1):
            verb_frames[lemma].update(per_word.get(0, ()))
            verb_frames[lemma].update(per_word.get(index, ()))

    return (
        {lemma: tuple(sorted(frames)) for lemma, frames in verb_frames.items()},
        {lemma: tuple(sorted(domains)) for lemma, domains in noun_domains.items()},
        {
            "noun_lemmas": len(noun_domains),
            "verb_lemmas": len(verb_frames),
            "wordnet_zip_sha256": hashlib.sha256(wordnet_zip.read_bytes()).hexdigest(),
        },
    )


def iter_brown_tokens(corpus_dir: Path):
    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file() or path.name in {"README", "CONTENTS", "cats.txt"}:
            continue
        for line_number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            for token_index, field in enumerate(line.split()):
                if "/" not in field:
                    continue
                raw_word, raw_tag = field.rsplit("/", 1)
                word, tag = raw_word.casefold(), tag_base(raw_tag)
                if WORD.fullmatch(word) and not tag.startswith(("np", "fw")):
                    yield word, tag, f"{path.name}:{line_number}:{token_index}"


def noun_lemma(word: str, exceptions: dict[str, tuple[str, ...]], domains: dict[str, tuple[str, ...]]) -> str:
    candidates = exceptions.get(word, ()) + (
        word,
        word[:-1] if word.endswith("s") else "",
        word[:-2] if word.endswith("es") else "",
    )
    return next((item for item in candidates if item and item in domains), word)


def verb_lemma(word: str, exceptions: dict[str, tuple[str, ...]], frames: dict[str, tuple[int, ...]]) -> str:
    candidates = exceptions.get(word, ()) + (
        word,
        word[:-1] if word.endswith("s") else "",
        word[:-2] if word.endswith("ed") else "",
        word[:-1] if word.endswith("ed") else "",
    )
    return next((item for item in candidates if item and item in frames), word)


@dataclass(frozen=True)
class Lexeme:
    surface: str
    lemma: str
    slot: str
    number: str
    tense: str
    person: str
    valency: str
    semantic_types: tuple[str, ...]
    brown_count: int
    brown_source: str
    wordnet_frames: tuple[int, ...] = ()

    @property
    def tape(self) -> str:
        return normalize_letters(self.surface)

    @property
    def content(self) -> bool:
        return self.slot in CONTENT_SLOTS


def build_inventory(corpus_dir: Path, wordnet_zip: Path, *, per_slot: int) -> tuple[dict[str, tuple[Lexeme, ...]], dict]:
    """Build a fixed, frequency-ranked Brown/WordNet terminal domain."""
    frames, noun_domains, wn_stats = load_wordnet(wordnet_zip)
    noun_exc = load_exceptions(wordnet_zip, "noun")
    verb_exc = load_exceptions(wordnet_zip, "verb")
    counts: Counter[tuple[str, str]] = Counter()
    first = {}
    corpus_digest = hashlib.sha256()
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file():
            corpus_digest.update(path.name.encode() + b"\0" + path.read_bytes())
    for word, tag, source in iter_brown_tokens(corpus_dir):
        counts[(word, tag)] += 1
        first.setdefault((word, tag), source)

    slots: dict[str, list[Lexeme]] = defaultdict(list)
    # Articles are grammar terminals but must still be frequent Brown tokens.
    for article in ("a", "the"):
        count = sum(value for (word, _tag), value in counts.items() if word == article)
        source = next((first[key] for key in first if key[0] == article), "")
        if count:
            slots["determiner"].append(Lexeme(article, article, "determiner", "", "", "", "", (), count, source))

    noun_rows: dict[tuple[str, str], Lexeme] = {}
    for (word, tag), count in counts.items():
        if tag not in NOUN_TAGS or count < 4 or not (2 <= len(word) <= 10):
            continue
        lemma = noun_lemma(word, noun_exc, noun_domains)
        domains = noun_domains.get(lemma, ())
        if not domains:
            continue
        number = "plural" if tag == "nns" else "singular"
        row = Lexeme(word, lemma, "noun", number, "", "third", "", domains, count, first[(word, tag)])
        key = (word, number)
        if key not in noun_rows or count > noun_rows[key].brown_count:
            noun_rows[key] = row
    ranked_nouns = sorted(noun_rows.values(), key=lambda row: (-row.brown_count, row.surface, row.number))
    # Subjects are restricted to plausibly agentive domains. Objects use the
    # same common inventory without an animacy restriction.
    animate = {"noun.animal", "noun.group", "noun.person"}
    subjects = [replace(row, slot="subject") for row in ranked_nouns if animate.intersection(row.semantic_types)]
    objects = [replace(row, slot="object") for row in ranked_nouns]
    slots["subject"].extend(subjects[:per_slot])
    slots["object"].extend(objects[:per_slot])

    verb_rows: dict[tuple[str, str, str], Lexeme] = {}
    for (word, tag), count in counts.items():
        if tag not in VERB_TAGS or count < 4 or not (2 <= len(word) <= 10):
            continue
        lemma = verb_lemma(word, verb_exc, frames)
        licensed = frames.get(lemma, ())
        if not set(licensed).intersection(TRANSITIVE_FRAMES):
            continue
        number = "singular" if tag == "vbz" else ("any" if tag == "vbd" else "plural")
        tense = "past" if tag == "vbd" else "present"
        row = Lexeme(word, lemma, "verb", number, tense, "third", "transitive", (), count,
                     first[(word, tag)], licensed)
        key = (word, number, tense)
        if key not in verb_rows or count > verb_rows[key].brown_count:
            verb_rows[key] = row
    slots["verb"].extend(sorted(verb_rows.values(), key=lambda row: (-row.brown_count, row.surface))[:per_slot])

    inventory = {name: tuple(rows) for name, rows in slots.items()}
    stats = {
        **wn_stats,
        "brown_raw_sha256": corpus_digest.hexdigest(),
        "per_slot_cap": per_slot,
        "terminals": {name: len(rows) for name, rows in inventory.items()},
        "brown_token_types": len(counts),
        "minimum_brown_count": 4,
    }
    return inventory, stats


class CharacterTrie:
    """Trie over one grammar slot in the direction exposed to the chart."""

    def __init__(self, terminals: tuple[Lexeme, ...], *, reverse: bool):
        self.children: list[dict[str, int]] = [dict()]
        self.terminals: list[list[Lexeme]] = [[]]
        for terminal in terminals:
            tape = terminal.tape[::-1] if reverse else terminal.tape
            node = 0
            for char in tape:
                following = self.children[node].get(char)
                if following is None:
                    following = self._new_node()
                    self.children[node][char] = following
                node = following
            self.terminals[node].append(terminal)

    def _new_node(self) -> int:
        self.children.append({})
        self.terminals.append([])
        return len(self.children) - 1

    def _descendants(self, node: int):
        yield from self.terminals[node]
        for child in self.children[node].values():
            yield from self._descendants(child)

    def compatible(self, residual: str) -> tuple[Lexeme, ...]:
        """Return words whose exposed tape is prefix-comparable to residual."""
        if not residual:
            return tuple(self._descendants(0))
        node = 0
        rows = []
        for index, char in enumerate(residual):
            rows.extend(self.terminals[node])  # word ended before residual
            following = self.children[node].get(char)
            if following is None:
                return tuple(rows)
            node = following
            if index == len(residual) - 1:
                rows.extend(self._descendants(node))  # residual ended first or together
        # A surface can occur with multiple Brown tag analyses. Preserve them;
        # dependency features, not spelling alone, distinguish chart edges.
        return tuple(rows)


@dataclass(frozen=True)
class ClausePlan:
    plan_id: str
    slots: tuple[str, ...]
    subject_determiner: str
    object_determiner: str


PLANS = (
    ClausePlan("definite-subject-indefinite-object", ("determiner", "subject", "verb", "determiner", "object"), "the", "a"),
    ClausePlan("indefinite-subject-definite-object", ("determiner", "subject", "verb", "determiner", "object"), "a", "the"),
)


@dataclass(frozen=True)
class ParseState:
    plan_id: str
    cursor: int
    subject_head: str = ""
    subject_number: str = ""
    predicate_head: str = ""
    predicate_number: str = ""
    tense: str = ""
    object_head: str = ""
    open_valencies: tuple[str, ...] = ("subject", "finite_predicate", "object")
    discourse_referents: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    agreement_resolved: bool = False
    valency_resolved: bool = False


def initial_parse(plan: ClausePlan, *, reverse: bool) -> ParseState:
    return ParseState(plan.plan_id, len(plan.slots) - 1 if reverse else 0)


def plan_by_id(plan_id: str) -> ClausePlan:
    return next(plan for plan in PLANS if plan.plan_id == plan_id)


def parse_complete(state: ParseState, *, reverse: bool) -> bool:
    plan = plan_by_id(state.plan_id)
    cursor_done = state.cursor < 0 if reverse else state.cursor == len(plan.slots)
    return (
        cursor_done
        and not state.open_valencies
        and state.agreement_resolved
        and state.valency_resolved
        and bool(state.tense)
        and bool(state.subject_head and state.predicate_head and state.object_head)
    )


def next_slot(state: ParseState, *, reverse: bool) -> str | None:
    if parse_complete(state, reverse=reverse):
        return None
    plan = plan_by_id(state.plan_id)
    if state.cursor < 0 or state.cursor >= len(plan.slots):
        return None
    return plan.slots[state.cursor]


def agreement(subject_number: str, verb_number: str) -> bool:
    return bool(subject_number) and verb_number in {subject_number, "any"}


def advance_parse(state: ParseState, terminal: Lexeme, *, reverse: bool) -> ParseState | None:
    slot = next_slot(state, reverse=reverse)
    if slot is None or terminal.slot != slot:
        return None
    plan = plan_by_id(state.plan_id)
    if slot == "determiner":
        required = plan.subject_determiner if state.cursor == 0 else plan.object_determiner
        if terminal.surface != required:
            return None
    values = asdict(state)
    values["cursor"] = state.cursor - 1 if reverse else state.cursor + 1
    refs = list(state.discourse_referents)
    if slot == "subject":
        values["subject_head"] = terminal.lemma
        values["subject_number"] = terminal.number
        refs.append(("subject", terminal.lemma, terminal.semantic_types))
    elif slot == "verb":
        if terminal.valency != "transitive" or not set(terminal.wordnet_frames).intersection(TRANSITIVE_FRAMES):
            return None
        values["predicate_head"] = terminal.lemma
        values["predicate_number"] = terminal.number
        values["tense"] = terminal.tense
    elif slot == "object":
        values["object_head"] = terminal.lemma
        refs.append(("object", terminal.lemma, terminal.semantic_types))
    values["discourse_referents"] = tuple(sorted(refs))
    open_values = []
    if not values["subject_head"]:
        open_values.append("subject")
    if not values["predicate_head"]:
        open_values.append("finite_predicate")
    if not values["object_head"]:
        open_values.append("object")
    values["open_valencies"] = tuple(open_values)
    values["agreement_resolved"] = (
        agreement(values["subject_number"], values["predicate_number"])
        if values["subject_head"] and values["predicate_head"] else False
    )
    # The exact finite surface features are retained separately in the chart
    # witness and revalidated at completion; reverse parsing may see the verb
    # before the subject.  A failed known comparison is pruned immediately.
    if slot == "verb" and state.subject_number and not agreement(state.subject_number, terminal.number):
        return None
    if slot == "subject" and state.predicate_head and not agreement(terminal.number, state.predicate_number):
        return None
    values["valency_resolved"] = bool(values["predicate_head"] and values["object_head"])
    return ParseState(**values)


@dataclass(frozen=True)
class ChartState:
    left: ParseState
    right: ParseState
    owner: str
    residual: str
    left_cursor: int
    right_cursor: int
    left_boundary_mask: int
    right_boundary_mask: int
    used_content_lemmas: frozenset[str]


@dataclass
class ChartCell:
    left_tokens: tuple[Lexeme, ...]
    right_tokens_reverse: tuple[Lexeme, ...]
    derivations: int = 1
    alternate_backpointers: int = 0


def feature_valid(tokens: tuple[Lexeme, ...]) -> bool:
    subject = next((row for row in tokens if row.slot == "subject"), None)
    verb = next((row for row in tokens if row.slot == "verb"), None)
    obj = next((row for row in tokens if row.slot == "object"), None)
    return bool(
        subject and verb and obj and agreement(subject.number, verb.number)
        and verb.tense in {"present", "past"}
        and verb.valency == "transitive"
        and set(verb.wordnet_frames).intersection(TRANSITIVE_FRAMES)
    )


def discourse_connection(left: ParseState, right: ParseState) -> dict:
    left_refs = [row for row in left.discourse_referents if row[0] == "object"]
    right_refs = [row for row in right.discourse_referents if row[0] == "subject"]
    bridges = []
    for lrole, llemma, ltypes in left_refs:
        for rrole, rlemma, rtypes in right_refs:
            shared = sorted(set(ltypes).intersection(rtypes))
            if shared:
                bridges.append({"left_role": lrole, "left_lemma": llemma,
                                "right_role": rrole, "right_lemma": rlemma,
                                "shared_wordnet_domains": shared})
    return {"connected": bool(bridges), "bridges": bridges}


def boundary_positions(mask: int) -> list[int]:
    return [index for index in range(1, mask.bit_length()) if mask & (1 << index)]


def render_clause(tokens: tuple[Lexeme, ...]) -> str:
    return " ".join(row.surface for row in tokens).capitalize() + "."


def independent_audit(rendered: str) -> dict:
    tape = normalize_letters(rendered)
    left, right = 0, len(tape) - 1
    comparisons = 0
    mismatch = None
    while left < right:
        comparisons += 1
        if tape[left] != tape[right]:
            mismatch = {"left": left, "right": right, "left_character": tape[left], "right_character": tape[right]}
            break
        left += 1
        right -= 1
    return {
        "letters": len(tape),
        "normalized_tape": tape,
        "two_pointer_exact": mismatch is None,
        "comparisons": comparisons,
        "first_mismatch": mismatch,
        "sha256_forward": sha256_text(tape),
        "sha256_reverse": sha256_text(tape[::-1]),
    }


def audit_candidate(state: ChartState, cell: ChartCell) -> dict:
    left_tokens = cell.left_tokens
    right_tokens = tuple(reversed(cell.right_tokens_reverse))
    surface_for_audit = f"{render_clause(left_tokens)} {render_clause(right_tokens)}"
    audit = independent_audit(surface_for_audit)
    central = mechanical_admission_checks(surface_for_audit, min_letters=39, max_letters=160)
    connection = discourse_connection(state.left, state.right)
    left_complete = parse_complete(state.left, reverse=False) and feature_valid(left_tokens)
    right_complete = parse_complete(state.right, reverse=True) and feature_valid(right_tokens)
    terminal_cursor = state.left_cursor
    shared = set(boundary_positions(state.left_boundary_mask)).intersection(
        boundary_positions(state.right_boundary_mask)
    )
    internal_shared = sorted(position for position in shared if position != terminal_cursor)
    opposite_segmentation = (
        state.left_boundary_mask != state.right_boundary_mask and not internal_shared
    )
    gates = {
        "minimum_letters_exclusive_38": audit["letters"] > 38,
        "independent_two_pointer_exact": audit["two_pointer_exact"],
        "independent_sha_exact": audit["sha256_forward"] == audit["sha256_reverse"],
        "left_complete_finite_dependency_parse": left_complete,
        "right_complete_finite_dependency_parse": right_complete,
        "agreement_and_tense": feature_valid(left_tokens) and feature_valid(right_tokens),
        "closed_valencies": not state.left.open_valencies and not state.right.open_valencies,
        "connected_discourse_referents": connection["connected"],
        "different_token_segmentation": opposite_segmentation,
        "complementary_boundary_masks": not internal_shared,
        "common_brown_wordnet_terminals": all(row.brown_count >= 4 for row in (*left_tokens, *right_tokens)),
        "central_mechanical_admission": all(central.values()),
        "no_seed_or_posthoc_repair": True,
    }
    accepted = all(gates.values())
    row = {
        "left_tokens": [asdict(row) for row in left_tokens],
        "right_tokens": [asdict(row) for row in right_tokens],
        "left_parse": asdict(state.left),
        "right_parse": asdict(state.right),
        "character_state": {"owner": state.owner, "residual": state.residual,
                            "left_cursor": state.left_cursor, "right_cursor": state.right_cursor},
        "boundary_masks": {"left_bitset": state.left_boundary_mask,
                           "right_bitset": state.right_boundary_mask,
                           "left_positions": boundary_positions(state.left_boundary_mask),
                           "right_positions_from_outer_edge": boundary_positions(state.right_boundary_mask),
                           "forbidden_internal_shared": internal_shared},
        "discourse": connection,
        "independent_audit": audit,
        "central_admission": central,
        "admission_gates": gates,
        "accepted": accepted,
        "packed_derivations": cell.derivations,
    }
    # Rejected complete cells remain reproducible through their token records
    # and audit hashes, but only a row that has already passed every gate is
    # promoted to a rendered candidate.
    if accepted:
        row["rendered"] = surface_for_audit
    else:
        row["rejected_surface_withheld"] = True
    return row


def preflight(registry_path: Path) -> dict:
    tokens = frozenset(SIGNATURE.split("|"))
    entries = json.loads(registry_path.read_text()).get("entries", []) if registry_path.exists() else []
    scored = []
    exact = []
    for entry in entries:
        other = frozenset(str(entry.get("signature", "")).split("|"))
        if entry.get("signature") == SIGNATURE:
            exact.append(entry.get("id"))
        if other:
            score = len(tokens & other) / len(tokens | other)
            if score:
                scored.append({"id": entry.get("id"), "jaccard": round(score, 4),
                               "shared": sorted(tokens & other)})
    scored.sort(key=lambda row: (-row["jaccard"], str(row["id"])))
    return {
        "status": "blocked" if exact else "passed",
        "signature": SIGNATURE,
        "registry_read": registry_path.exists(),
        "registry_entries_checked": len(entries),
        "exact_signature_collisions": exact,
        "closest_five": scored[:5],
        "bounded_comparisons": len(entries),
        "distinction": (
            "global dual finite-clause dependency state and both token masks are chart keys; "
            "the retired r=s family carried only a local residual stack"
        ),
    }


def best_obstruction(
    dead: list[dict], tries: dict[tuple[str, str], CharacterTrie], *, exhausted: bool
) -> dict:
    if not dead:
        return {"kind": "empty_chart", "domain_exhausted": exhausted}
    dead.sort(key=lambda row: (-row["matched_cursor"], len(row["state"]["residual"]), row["reason"]))
    row = dead[0]
    state = row.pop("_chart_state")
    side = "right" if state.owner == "left" else "left"
    parse = state.right if side == "right" else state.left
    slot = next_slot(parse, reverse=side == "right")
    best = None
    if slot:
        for terminal in tries[(side, slot)].compatible(""):
            exposed = terminal.tape[::-1] if side == "right" else terminal.tape
            cursor = 0
            for expected, observed in zip(state.residual, exposed):
                if expected != observed:
                    break
                cursor += 1
            key = (cursor, terminal.brown_count, terminal.surface)
            if best is None or key > best[0]:
                best = (key, terminal, exposed)
    row["next_required_side"] = side
    row["next_required_slot"] = slot
    row["best_terminal_probe"] = None if best is None else {
        "terminal": asdict(best[1]),
        "exposed_tape": best[2],
        "matching_prefix_characters": best[0][0],
        "expected_residual": state.residual,
        "expected_next_character": state.residual[best[0][0]:best[0][0] + 1],
        "observed_next_character": best[2][best[0][0]:best[0][0] + 1],
    }
    row["domain_exhausted"] = exhausted
    row["interpretation"] = (
        "No trie terminal for the required dependency slot can consume the live residual "
        "without violating character equality, syntax, agreement, freshness, or the two boundary masks."
    )
    return row


def search(
    inventory: dict[str, tuple[Lexeme, ...]], *, max_states: int
) -> dict:
    tries = {
        (side, slot): CharacterTrie(rows, reverse=side == "right")
        for slot, rows in inventory.items()
        for side in ("left", "right")
    }
    queue = deque()
    packed: dict[ChartState, ChartCell] = {}
    for left_plan in PLANS:
        for right_plan in PLANS:
            state = ChartState(initial_parse(left_plan, reverse=False), initial_parse(right_plan, reverse=True),
                               "", "", 0, 0, 0, 0, frozenset())
            packed[state] = ChartCell((), ())
            queue.append(state)
    complete = []
    dead = []
    stats = Counter()
    rejection = Counter()

    while queue and len(packed) < max_states:
        state = queue.popleft()
        cell = packed[state]
        stats["states_expanded"] += 1
        left_done = parse_complete(state.left, reverse=False)
        right_done = parse_complete(state.right, reverse=True)
        if left_done and right_done:
            if not state.owner and not state.residual and state.left_cursor == state.right_cursor:
                complete.append((state, cell))
                stats["complete_chart_cells"] += 1
            else:
                rejection["complete_with_character_debt"] += 1
            continue

        sides = ("right",) if state.owner == "left" else (("left",) if state.owner == "right" else ("left",))
        advanced = 0
        local_reason = Counter()
        for side in sides:
            reverse = side == "right"
            parse = state.right if reverse else state.left
            if parse_complete(parse, reverse=reverse):
                local_reason["required_side_parse_already_complete"] += 1
                continue
            slot = next_slot(parse, reverse=reverse)
            if slot is None:
                local_reason["no_legal_dependency_slot"] += 1
                continue
            candidates = tries[(side, slot)].compatible(state.residual)
            stats["trie_queries"] += 1
            stats["trie_terminals_returned"] += len(candidates)
            if not candidates:
                local_reason["trie_prefix_obstruction"] += 1
            for terminal in candidates:
                stats["terminal_attempts"] += 1
                if terminal.content and terminal.lemma in state.used_content_lemmas:
                    rejection["global_lemma_freshness"] += 1
                    local_reason["global_lemma_freshness"] += 1
                    continue
                next_parse = advance_parse(parse, terminal, reverse=reverse)
                if next_parse is None:
                    rejection["dependency_or_agreement"] += 1
                    local_reason["dependency_or_agreement"] += 1
                    continue
                exposed = terminal.tape[::-1] if reverse else terminal.tape
                if state.residual:
                    common = min(len(state.residual), len(exposed))
                    if state.residual[:common] != exposed[:common]:
                        raise AssertionError("trie returned a non-prefix-comparable terminal")
                    if len(state.residual) > len(exposed):
                        owner, residual = state.owner, state.residual[common:]
                    elif len(exposed) > len(state.residual):
                        owner, residual = side, exposed[common:]
                    else:
                        owner, residual = "", ""
                else:
                    owner, residual = side, exposed
                left_cursor = state.left_cursor + (len(terminal.tape) if side == "left" else 0)
                right_cursor = state.right_cursor + (len(terminal.tape) if side == "right" else 0)
                left_mask = state.left_boundary_mask | ((1 << left_cursor) if side == "left" else 0)
                right_mask = state.right_boundary_mask | ((1 << right_cursor) if side == "right" else 0)
                next_left = next_parse if side == "left" else state.left
                next_right = next_parse if side == "right" else state.right
                complete_after = parse_complete(next_left, reverse=False) and parse_complete(next_right, reverse=True)
                shared = left_mask & right_mask
                terminal_bit = (1 << left_cursor) if complete_after and left_cursor == right_cursor else 0
                if shared & ~terminal_bit:
                    rejection["complementary_boundary_mask"] += 1
                    local_reason["complementary_boundary_mask"] += 1
                    continue
                if not residual and not complete_after:
                    rejection["intermediate_character_closure"] += 1
                    local_reason["intermediate_character_closure"] += 1
                    continue
                used = state.used_content_lemmas | ({terminal.lemma} if terminal.content else set())
                next_state = ChartState(next_left, next_right, owner, residual, left_cursor, right_cursor,
                                        left_mask, right_mask, frozenset(used))
                next_left_tokens = cell.left_tokens + ((terminal,) if side == "left" else ())
                next_right_tokens = cell.right_tokens_reverse + ((terminal,) if side == "right" else ())
                prior = packed.get(next_state)
                if prior is None:
                    packed[next_state] = ChartCell(next_left_tokens, next_right_tokens, cell.derivations)
                    queue.append(next_state)
                    advanced += 1
                    stats["packed_cells_created"] += 1
                else:
                    prior.derivations += cell.derivations
                    prior.alternate_backpointers += 1
                    stats["packed_reconvergences"] += 1
                    advanced += 1
        if not advanced:
            dominant = local_reason.most_common(1)[0][0] if local_reason else "no_transition"
            dead.append({
                "reason": dominant,
                "matched_cursor": min(state.left_cursor, state.right_cursor),
                "state": {
                    "owner": state.owner,
                    "residual": state.residual,
                    "left_cursor": state.left_cursor,
                    "right_cursor": state.right_cursor,
                    "left_boundary_positions": boundary_positions(state.left_boundary_mask),
                    "right_boundary_positions": boundary_positions(state.right_boundary_mask),
                    "left_parse": asdict(state.left),
                    "right_parse": asdict(state.right),
                },
                "witness": {
                    "left_tokens": [row.surface for row in cell.left_tokens],
                    "right_tokens_reverse": [row.surface for row in cell.right_tokens_reverse],
                },
                "local_rejections": dict(local_reason),
                "_chart_state": state,
            })
    exhausted = not queue
    audited = [audit_candidate(state, cell) for state, cell in complete]
    audited.sort(key=lambda row: (
        not row["accepted"],
        -row["independent_audit"]["letters"],
        row.get("rendered", ""),
    ))
    accepted = [row for row in audited if row["accepted"]]
    return {
        "stats": {**stats, "packed_cells": len(packed), "dead_cells": len(dead),
                  "complete_cells": len(complete), "audited_candidates": len(audited),
                  "accepted_candidates": len(accepted)},
        "rejections": dict(rejection),
        "domain_exhausted": exhausted,
        "cap_reached": not exhausted,
        "audited_candidates": audited,
        "accepted_candidates": accepted,
        "obstruction": None if accepted else best_obstruction(dead, tries, exhausted=exhausted),
    }


def run(corpus_dir: Path, wordnet_zip: Path, *, per_slot: int = 64, max_states: int = 500_000) -> dict:
    started = time.monotonic()
    novelty = preflight(ROOT / "docs" / "experiment-novelty-registry.json")
    if novelty["status"] != "passed":
        raise RuntimeError(f"preflight collision: {novelty['exact_signature_collisions']}")
    inventory, inventory_stats = build_inventory(corpus_dir, wordnet_zip, per_slot=per_slot)
    result = search(inventory, max_states=max_states)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "decision": (
            "Can a genuinely global packed chart jointly construct two complete finite dependency clauses "
            "from common Brown/WordNet terminals while matching all characters online and passing every "
            "central admission gate above 38 letters?"
        ),
        "novelty_preflight": novelty,
        "acceptance_gate": {
            "minimum_letters_exclusive": 38,
            "two_complete_finite_dependency_parses": True,
            "agreement_tense_and_transitive_valency": True,
            "connected_discourse_referents": True,
            "exact_two_pointer_and_sha": True,
            "different_opposite_segmentation": True,
            "complementary_token_boundary_masks": True,
            "all_central_mechanical_checks": True,
        },
        "fixed_domain": {
            "clause_plans": [asdict(plan) for plan in PLANS],
            "per_slot_frequency_cap": per_slot,
            "max_packed_states": max_states,
            "proper_names": False,
            "fragments": False,
            "completed_phrase_banks": False,
            "catalogue_text": False,
            "repeated_units": False,
            "seed_recovery": False,
            "post_hoc_repair": False,
            "lexical_widening_after_run": False,
            "retired_local_residual_stack": False,
        },
        "state_contract": {
            "parse_state": ["open_valencies", "subject_head", "predicate_head", "object_head",
                            "agreement_resolved", "tense", "discourse_referents"],
            "character_state": ["owner", "residual", "left_cursor", "right_cursor"],
            "boundary_state": ["left_boundary_mask", "right_boundary_mask"],
            "packing_key_excludes": ["rendered_text", "completed_phrase", "posthoc_score"],
            "terminal_access": "slot-specific forward/reverse character tries",
        },
        "inventory": inventory_stats,
        "search": result,
        "verdict": (
            "accepted exact connected English candidate above 38 letters"
            if result["accepted_candidates"] else
            "precise exhausted packed-chart obstruction"
            if result["domain_exhausted"] else
            "bounded chart cap reached without promotion"
        ),
        "provenance": {
            "host": os.uname().nodename,
            "python": sys.version.split()[0],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "brown_source": str(corpus_dir),
            "wordnet_source": str(wordnet_zip),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "every_complete_cell_rendered_and_independently_audited": True,
            "all_nonterminal_text_retained_only_as_obstruction_witness": True,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["result_sha256"] = sha256_text(canonical)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brown", required=True, type=Path)
    parser.add_argument("--wordnet", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-slot", type=int, default=64)
    parser.add_argument("--max-states", type=int, default=500_000)
    args = parser.parse_args()
    payload = run(args.brown, args.wordnet, per_slot=args.per_slot, max_states=args.max_states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"inventory": payload["inventory"], "stats": payload["search"]["stats"],
                      "domain_exhausted": payload["search"]["domain_exhausted"],
                      "accepted": len(payload["search"]["accepted_candidates"]),
                      "obstruction": payload["search"]["obstruction"],
                      "elapsed_seconds": payload["provenance"]["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
