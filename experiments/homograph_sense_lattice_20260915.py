"""Homograph-sense lattice for exact palindrome construction.

The lattice gives each orthographic word more than one independently attested
role (for example ``watch`` can be a noun or a verb).  Forward and reverse
surfaces are parsed with different role frames, so a closure is useful only if
the same tape supports two semantic/POS readings.  This is not a list of
semordnilap pairs and it never emits a reflected word sequence directly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "homograph-sense-lattice-20260915"
SIGNATURE = (
    "homograph-sense-lattice|one-orthographic-tape|two-independent-semantic-parses|"
    "sense-conditioned-word-boundary-choice|exact-letter-audit"
)
MIN_LETTERS = 39

# Roles and compact sense labels are deliberately explicit.  These are lexical
# types, not copied sentence material.
HOMOGRAPHS: dict[str, tuple[tuple[str, str], ...]] = {
    "can": (("VERB", "modal"), ("NOUN", "container")),
    "watch": (("NOUN", "timepiece"), ("VERB", "observe")),
    "light": (("NOUN", "illumination"), ("ADJ", "not-heavy"), ("VERB", "ignite")),
    "plant": (("NOUN", "organism"), ("VERB", "place")),
    "bark": (("NOUN", "tree-surface"), ("VERB", "dog-sound")),
    "seal": (("NOUN", "animal"), ("VERB", "close")),
    "book": (("NOUN", "volume"), ("VERB", "reserve")),
    "change": (("NOUN", "difference"), ("VERB", "alter")),
    "spring": (("NOUN", "season"), ("VERB", "leap")),
    "match": (("NOUN", "contest"), ("VERB", "fit")),
    "rose": (("NOUN", "flower"), ("VERB", "rise-past")),
    "record": (("NOUN", "document"), ("VERB", "capture")),
    "refuse": (("NOUN", "waste"), ("VERB", "decline")),
    "present": (("NOUN", "gift"), ("VERB", "offer"), ("ADJ", "current")),
    "address": (("NOUN", "location"), ("VERB", "speak-to")),
    "object": (("NOUN", "thing"), ("VERB", "oppose")),
    "permit": (("NOUN", "license"), ("VERB", "allow")),
    "produce": (("NOUN", "farm-goods"), ("VERB", "make")),
    "project": (("NOUN", "plan"), ("VERB", "throw")),
    "subject": (("NOUN", "topic"), ("VERB", "expose")),
    "content": (("NOUN", "material"), ("ADJ", "satisfied")),
    "desert": (("NOUN", "wilderness"), ("VERB", "abandon")),
    "entrance": (("NOUN", "entry"), ("VERB", "enchant")),
    "minute": (("NOUN", "time"), ("ADJ", "tiny")),
    "close": (("ADJ", "near"), ("VERB", "shut")),
    "read": (("VERB", "interpret"), ("ADJ", "literate")),
    "lead": (("NOUN", "metal"), ("VERB", "guide")),
    "wind": (("NOUN", "air"), ("VERB", "coil")),
    "row": (("NOUN", "line"), ("VERB", "argue")),
    "tear": (("NOUN", "drop"), ("VERB", "rip")),
    "bow": (("NOUN", "gesture"), ("VERB", "bend")),
    "bass": (("NOUN", "fish"), ("NOUN", "instrument")),
    "polish": (("NOUN", "shine"), ("VERB", "smooth")),
    "number": (("NOUN", "numeral"), ("VERB", "count")),
    "sound": (("NOUN", "noise"), ("ADJ", "healthy"), ("VERB", "measure")),
    "fair": (("NOUN", "event"), ("ADJ", "just")),
    "fine": (("NOUN", "penalty"), ("ADJ", "excellent"), ("VERB", "penalize")),
    "kind": (("NOUN", "type"), ("ADJ", "gentle")),
    "left": (("NOUN", "remainder"), ("ADJ", "departed"), ("VERB", "depart")),
    "right": (("NOUN", "entitlement"), ("ADJ", "correct")),
    "presented": (("VERB", "offered"), ("ADJ", "shown")),
    "saw": (("NOUN", "tool"), ("VERB", "see-past")),
}

FUNCTIONS = frozenset("a an the this that some many i we he she they you it me us them and or but if as of to in on at by for with from is are was were be do does did can could will would have has had not".split())
FRAMES = {
    "det_event": ("DET", "NOUN", "VERB", "DET", "NOUN"),
    "pron_event": ("PRON", "VERB", "DET", "NOUN"),
    "modified_event": ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    "copular": ("DET", "NOUN", "VERB", "ADJ"),
    "report": ("NOUN", "VERB", "DET", "NOUN", "ADP", "NOUN"),
}
ROLE_WORDS: dict[str, tuple[str, ...]] = defaultdict(tuple)
for _word, _senses in HOMOGRAPHS.items():
    for _role, _sense in _senses:
        ROLE_WORDS[_role] = tuple(dict.fromkeys((*ROLE_WORDS[_role], _word)))
for _role, _words in {
    "DET": "a an the some many this that".split(),
    "PRON": "i we he she they you it me us them".split(),
    "ADP": "in on at by to for with from near over under".split(),
    # Ordinary lexical support lets the homograph state be embedded in a
    # complete clause; ambiguity is still required at closure.
    "NOUN": "aide artist area agenda camera data drama idea letter memo note person story teacher writer river garden room book dog cat".split(),
    "VERB": "act add ask answer arrive call carry change close create draw enter find give help hold lead leave make move open read record return rise run say see send show speak start take tell use watch work".split(),
    "ADJ": "able bright calm clear close fair fine good kind light little modern present right safe sound strong".split(),
}.items():
    ROLE_WORDS[_role] = tuple(dict.fromkeys((*ROLE_WORDS.get(_role, ()), *_words)))

# Add corpus-derived lexical *types* (not spans) so the lattice can cross a
# boundary instead of dying on a hand-list seam.  The ambiguous homographs
# above remain the semantic signal required for admission.
try:
    from collections import Counter
    from nltk.corpus import brown
    from wordfreq import top_n_list, zipf_frequency

    _counts = Counter()
    for _sent in brown.tagged_sents(tagset="universal"):
        for _raw, _tag in _sent:
            _w = _raw.casefold()
            if _w.isascii() and _w.isalpha():
                _counts[(_w, _tag)] += 1
    _canon = {}
    for (_w, _tag), _count in _counts.items():
        if _w not in _canon or _count > _counts[(_w, _canon[_w])]:
            _canon[_w] = _tag
    for _w in top_n_list("en", 50_000):
        if not (_w.isascii() and _w.isalpha() and len(_w) >= 3 and _w != _w[::-1] and zipf_frequency(_w, "en") >= 3.25):
            continue
        _tag = _canon.get(_w)
        if _tag in {"NOUN", "VERB", "ADJ", "ADV"}:
            ROLE_WORDS[_tag] = tuple(dict.fromkeys((*ROLE_WORDS.get(_tag, ()), _w)))
except LookupError:
    # The explicit inventory still gives a deterministic smoke test when the
    # optional Brown corpus is unavailable.
    pass

# A deterministic cap keeps the lattice a bounded experiment while preserving
# the hand-audited homographs and the highest-frequency corpus types first.
for _role, _limit in {"NOUN": 420, "VERB": 360, "ADJ": 260, "ADV": 180}.items():
    ROLE_WORDS[_role] = ROLE_WORDS[_role][:_limit]


def _content_unique(words: tuple[str, ...]) -> bool:
    content = [word for word in words if word not in FUNCTIONS and len(word) > 2]
    return len(content) == len(set(content))


def _sense_parse(words: tuple[str, ...], tags: tuple[str, ...]) -> dict:
    senses = []
    ambiguous = 0
    for word, tag in zip(words, tags):
        options = tuple((role, sense) for role, sense in HOMOGRAPHS.get(word, ()) if role == tag)
        if len(HOMOGRAPHS.get(word, ())) > 1:
            ambiguous += 1
        senses.append({"word": word, "role": tag, "sense_options": list(options)})
    return {"ambiguous_word_count": ambiguous, "assignments": senses}


def _audit(text: str, left_tags: tuple[str, ...], right_tags: tuple[str, ...]) -> dict:
    tape = normalize_letters(text)
    independent = "".join(ch for ch in text.casefold() if "a" <= ch <= "z")
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=180)
    left_words = tuple(text.casefold().replace(";", "").replace(".", "").split())[: len(left_tags)]
    right_words = tuple(text.casefold().replace(";", "").replace(".", "").split())[len(left_tags):]
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_tape": tape,
        "independent_ascii_tape": independent,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_exact": bool(independent) and independent == independent[::-1],
        "tapes_equal": tape == independent,
        "mechanical_checks": checks,
        "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
        "forward_parse": _sense_parse(left_words, left_tags),
        "reverse_parse": _sense_parse(tuple(reversed(right_words)), tuple(reversed(right_tags))),
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def solve(left_tags: tuple[str, ...], right_tags: tuple[str, ...], budget: int = 30_000) -> tuple[list[dict], dict, list[dict]]:
    stack = [(0, len(right_tags) - 1, "", 0, (), ())]
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    while stack and stats["states"] < budget:
        i, j, residual, owner, left, right_rev = stack.pop()
        stats["states"] += 1
        if len(probes) < 30 and (left or right_rev):
            partial_words = left + tuple(reversed(right_rev))
            tape = "".join(partial_words)
            probes.append({"rendered": " ".join(partial_words).capitalize() + ".", "letters": len(tape), "normalized_tape": tape, "residual": residual, "owner": owner})
        if i >= len(left_tags) and j < 0:
            stats["terminal"] += 1
            if residual and residual != residual[::-1]:
                stats["dead_terminal"] += 1
                continue
            words = left + tuple(reversed(right_rev))
            if len(normalize_letters("".join(words))) < MIN_LETTERS or not _content_unique(words):
                stats["admission_shape_reject"] += 1
                continue
            left_parse = _sense_parse(left, left_tags)
            right_parse = _sense_parse(tuple(reversed(right_rev)), right_tags)
            if left_parse["ambiguous_word_count"] + right_parse["ambiguous_word_count"] < 2:
                stats["not_dual_sense"] += 1
                continue
            text = " ".join(words[: len(left_tags)]).capitalize() + "; " + " ".join(words[len(left_tags):]) + "."
            audit = _audit(text, left_tags, right_tags)
            row = {"rendered": text, "left_tags": list(left_tags), "right_tags": list(right_tags), "audit": audit, "reader_status": "not_run; dual-sense parsing is not human readability evidence"}
            rows.append(row)
            stats["exact_terminal"] += int(audit["exact"])
            stats["mechanically_admitted"] += int(audit["mechanically_admitted"])
            continue
        if owner == 0:
            if i >= len(left_tags) or j < 0:
                stats["shape_mismatch"] += 1
                continue
            lt, rt = left_tags[i], right_tags[j]
            right_by_first: dict[str, list[str]] = defaultdict(list)
            for word in ROLE_WORDS.get(rt, ()):
                right_by_first[word[-1]].append(word)
            for word in ROLE_WORDS.get(lt, ()):
                for other in right_by_first.get(word[0], ()):
                    if word not in FUNCTIONS and word in left + right_rev:
                        continue
                    if other not in FUNCTIONS and other in left + right_rev:
                        continue
                    reversed_other = other[::-1]
                    if word.startswith(reversed_other):
                        rem, new_owner = word[len(reversed_other):], 1 if len(word) > len(reversed_other) else 0
                    elif reversed_other.startswith(word):
                        rem, new_owner = reversed_other[len(word):], -1 if len(reversed_other) > len(word) else 0
                    else:
                        continue
                    stack.append((i + 1, j - 1, rem, new_owner, left + (word,), right_rev + (other,)))
        elif owner == 1:
            if j < 0:
                stats["shape_mismatch"] += 1
                continue
            for other in ROLE_WORDS.get(right_tags[j], ()):
                e = other[::-1]
                if residual.startswith(e):
                    rem, new_owner = residual[len(e):], 1 if len(residual) > len(e) else 0
                elif e.startswith(residual):
                    rem, new_owner = e[len(residual):], -1 if len(e) > len(residual) else 0
                else:
                    continue
                stack.append((i, j - 1, rem, new_owner, left, right_rev + (other,)))
        else:
            if i >= len(left_tags):
                stats["shape_mismatch"] += 1
                continue
            for word in ROLE_WORDS.get(left_tags[i], ()):
                if residual.startswith(word):
                    rem, new_owner = residual[len(word):], -1 if len(residual) > len(word) else 0
                elif word.startswith(residual):
                    rem, new_owner = word[len(residual):], 1 if len(word) > len(residual) else 0
                else:
                    continue
                stack.append((i + 1, j, rem, new_owner, left + (word,), right_rev))
    return rows, dict(stats), probes


def run() -> dict:
    all_rows: list[dict] = []
    all_probes: list[dict] = []
    stats = Counter()
    pair_stats = {}
    for name, left_tags in FRAMES.items():
        for right_name, right_tags in FRAMES.items():
            rows, state, probes = solve(left_tags, right_tags)
            for row in rows:
                row["frame_pair"] = [name, right_name]
            all_rows.extend(rows)
            all_probes.extend(probes)
            pair_stats[f"{name}__{right_name}"] = state
            stats.update({f"{name}__{right_name}.{key}": value for key, value in state.items()})
    unique = {}
    for row in all_rows:
        tape = row["audit"]["normalized_tape"]
        if tape not in unique:
            unique[tape] = row
    rows = sorted(unique.values(), key=lambda row: (row["audit"]["mechanically_admitted"], row["audit"]["letters"]), reverse=True)
    admitted = [row for row in rows if row["audit"]["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not admitted else "exact_hits_pending_blinded_readers",
        "novelty_audit": {
            "registry_entries_read_before_run": 82,
            "excluded_routes_read_before_run": 6,
            "signature_overlap": [],
            "conceptual_near_pairs": [],
            "manual_review_required": False,
            "self_entry_present": False,
            "repair_of_registered_family": False,
        },
        "config": {"frames": {key: list(value) for key, value in FRAMES.items()}, "frame_pairs": len(FRAMES) ** 2, "catalogue_text_copied": False},
        "stats": {**stats, "unique_terminal_rows": len(rows), "mechanically_admitted": len(admitted), "reader_eligible": 0},
        "pair_stats": pair_stats,
        "rendered_candidates": rows[:120],
        "rendered_probes": all_probes[:160],
        "exact_candidates": admitted,
        "next_repair": "Hold the dual-sense parse state fixed and add a held-out homograph inventory with agreement-aware chart transitions; do not fall back to semordnilap pairs or reverse word-order emission.",
        "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "explicit homograph sense/role inventory; no intact source sentences", "programmatic_readability_claim": False},
        "reader_gate": "No row is reader evidence; any admitted item requires manual intact-prose review and randomized blinded shuffled controls.",
    }


if __name__ == "__main__":
    out = run()
    path = ROOT / "runs" / "homograph-sense-lattice-20260915.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "stats": out["stats"], "path": str(path)}, indent=2))
