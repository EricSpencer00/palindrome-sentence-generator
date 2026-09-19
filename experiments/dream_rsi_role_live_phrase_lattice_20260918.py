"""Dream-RSI role-carrying phrase lattice.

This is the concrete repair after the role-agnostic model phrase-bank lane:
the model must propose a balanced inventory of clause-role phrases, and the
search carries those role labels alongside the exact character overhang.  A
child is rejected immediately when either displayed half can no longer be a
complete subject/verb clause prefix or suffix.  The model proposes lexical
material only; the host trie, grammar chart, and independent audits decide
what survives.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.bigram import BigramModel
from llm_palindrome.generate import build_vocab
from llm_palindrome.phrases import build_inventory
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.sentence_plan import SentencePlan
from llm_palindrome.syntax import brown_tables
from experiments.dream_rsi_model_guided_span_resynthesis_20260918 import audit


EXPERIMENT = "dream-rsi-role-live-phrase-lattice-20260918"
HOST = "http://127.0.0.1:11434"
MODEL = "gpt-oss:20b"
ROLES = ("subject", "verb_phrase", "object", "adjunct")
PHRASE_RE = re.compile(r"[a-z]+(?: [a-z]+){1,6}")

PROMPT = (
    "Generate exactly 32 distinct original ordinary-English phrase units in "
    'JSON only: {"items":[{"text":"...","role":"subject|verb_phrase|object|adjunct"}]}. '
    "Return eight items of each role, in mixed order. Subject units are "
    "natural noun phrases, verb_phrase units are finite verb phrases, object "
    "units are natural objects, and adjunct units are time/place/manner "
    "phrases. Each unit has two to seven lowercase ASCII words, no punctuation, "
    "no quotations, no famous palindrome, no list, and no fragmentary word salad. "
    "The units should combine into ordinary complete clauses about reading, "
    "archives, weather, cooking, travel, or daily work."
)


def request_json(body: dict) -> dict:
    request = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)


def parse_items(raw: str) -> tuple[list[dict], str | None]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return [], "no_json_object"
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        return [], f"json_error:{exc.msg}"
    values = payload.get("items")
    if not isinstance(values, list):
        return [], "items_not_list"
    out, seen = [], set()
    for value in values:
        if not isinstance(value, dict):
            continue
        raw_text = str(value.get("text", "")).casefold().strip()
        # Do not silently turn punctuation-bearing model output into a new
        # phrase: provenance must name exactly what entered the trie.
        if not re.fullmatch(r"[a-z]+(?: [a-z]+){1,6}", raw_text):
            continue
        text = raw_text
        role = str(value.get("role", "")).casefold().strip()
        if role not in ROLES or not PHRASE_RE.fullmatch(text) or text in seen:
            continue
        seen.add(text)
        out.append({"text": text, "role": role})
    return out, None


def model_phrase_bank() -> tuple[list[dict], dict]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": "low",
        "options": {"temperature": 0.82, "num_predict": 2600, "seed": 2026091802},
    }
    response = request_json(body)
    raw = response.get("message", {}).get("content", "")
    items, parse_error = parse_items(raw)
    return items, {
        "raw_reply": raw,
        "parse_error": parse_error,
        "model": MODEL,
        "prompt": PROMPT,
    }


def words(units: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(word for unit in units for word in unit.split())


def role_path_possible(units: tuple[str, ...], role_map: dict[str, str]) -> bool:
    """Keep tagged units in a clause-compatible order.

    Untagged corpus words/phrases are wildcards.  The finite role sequence is
    intentionally permissive about optional objects and multiple adjuncts but
    never permits a verb phrase before a subject or an object before a verb.
    """
    seen = [role_map[unit] for unit in units if unit in role_map]
    if not seen:
        return True
    allowed = {
        "subject": {"subject", "verb_phrase", "adjunct"},
        "verb_phrase": {"verb_phrase", "object", "adjunct"},
        "object": {"object", "adjunct"},
        "adjunct": {"adjunct"},
    }
    return all(next_role in allowed[role] for role, next_role in zip(seen, seen[1:]))


class LiveRoleGrammar:
    def __init__(self, plan: SentencePlan, role_map: dict[str, str]):
        self.plan = plan
        self.role_map = role_map
        self.rejected = 0
        self.role_rejected = 0
        self.syntax_rejected = 0

    def __call__(self, left: tuple[str, ...], right: tuple[str, ...]) -> bool:
        left_words, right_words = words(left), words(right)
        if not role_path_possible(left, self.role_map) or not role_path_possible(right, self.role_map):
            self.role_rejected += 1
            self.rejected += 1
            return False
        ok = self.plan.suffix_possible(left_words) and self.plan.prefix_possible(right_words)
        if not ok:
            self.syntax_rejected += 1
            self.rejected += 1
        return ok


def run(*, seeds: int = 4, min_letters: int = 80, max_letters: int = 180) -> dict:
    items, model_record = model_phrase_bank()
    phrases = [item["text"] for item in items]
    role_map = {item["text"]: item["role"] for item in items}
    role_counts = Counter(item["role"] for item in items)
    table, shapes, _ = brown_tables(3, 9)
    plan = SentencePlan(table, shapes, min_words=4, max_words=9)
    local_vocab = build_vocab(16000)
    inventory = build_inventory(str(ROOT / "data" / "count_2w.txt"), vocab=local_vocab, top_n=12000, min_count=3)
    units = list(dict.fromkeys(local_vocab + inventory + phrases))
    tries = WordTries(units)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=set(local_vocab))
    scorer = CoherentScorer(
        bigrams,
        freq_weight=0.10,
        length_weight=0.14,
        phrase_weight=4.0,
        long_bonus=1.2,
        short_penalty=2.0,
        unit_bonus={phrase: 7.0 for phrase in phrases},
    )
    live = LiveRoleGrammar(plan, role_map)
    records = []
    for seed in range(seeds):
        units_out = beam_search(
            tries,
            scorer,
            min_letters=min_letters,
            max_steps=220,
            beam_width=180,
            candidate_limit=500,
            per_parent=8,
            seed=seed,
            diversity=1.25,
            max_word_uses=2,
            allow_state=live,
        )
        if not units_out:
            continue
        text = " ".join(units_out)
        checks = mechanical_admission_checks(text, min_letters=min_letters, max_letters=max_letters)
        expanded = words(units_out)
        records.append({
            "seed": seed,
            "rendered": text,
            "letters": len(normalize_letters(text)),
            "units": units_out,
            "expanded_words": expanded,
            "model_role_units_used": [unit for unit in units_out if unit in role_map],
            "audit": audit(text),
            "mechanical_checks": checks,
            "mechanically_admitted": all(checks.values()),
            "grammar_halves": {
                "left_suffix_possible": plan.suffix_possible(expanded),
                "right_prefix_possible": plan.prefix_possible(expanded),
            },
            "reader_status": "human-unreviewed",
            "provenance": {
                "fresh_model_phrase_bank": True,
                "role_labels_carried_live": True,
                "model_phrase_count": len(phrases),
                "catalogue_imported": False,
                "finished_tape_reversed": False,
                "human_readability_certified": False,
            },
        })
    exact = [row for row in records if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {
        "experiment": EXPERIMENT,
        "method": "Dream-RSI balanced role-tagged phrase bank plus live clause-prefix/suffix grammar and exact character lattice",
        "model": model_record,
        "phrase_items": items,
        "role_counts": dict(role_counts),
        "records": records,
        "fresh_exact_closures": exact,
        "mechanically_admitted": admitted,
        "stats": {
            "model_phrases": len(phrases),
            "role_counts": dict(role_counts),
            "seeds": seeds,
            "closures": len(records),
            "exact": len(exact),
            "mechanically_admitted": len(admitted),
            "longest_letters": max((row["letters"] for row in records), default=0),
            "live_rejections": live.rejected,
            "role_rejections": live.role_rejected,
            "syntax_rejections": live.syntax_rejected,
        },
        "reader_gate": {
            "status": "closed",
            "reason": "No exact row is reader-eligible until it clears the strict mechanical gate and a blinded intact/shuffled study.",
            "programmatic_metrics_are_diagnostic": True,
        },
        "next_repair": {
            "operator": "replace corpus-shape clause charts with a typed hand-authored subject/verb/object automaton and retain the role labels at every mirrored boundary",
            "reason": "the live role lattice rejects fragments, but Brown clause shapes are too sparse for long exact crossings",
            "reader_test": "only a strict exact survivor enters randomized intact/shuffled blinded rating",
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "catalogue_imported": False,
            "human_readability_certified": False,
        },
    }


if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        directory.mkdir(exist_ok=True)
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))
    for row in sorted(payload["fresh_exact_closures"], key=lambda item: -item["letters"]):
        print(f"{row['letters']} letters | {row['rendered']}")
