"""Audit rendered palindrome proposals without pretending to certify readability.

The report is intentionally candidate-first: every row keeps its rendered text,
source run, provenance, letter count, two independent exact-tape checks, the
shared mechanical gate, and transparent local-order diagnostics.  The Brown
bigram and word-frequency values can reject obvious debris or prioritize a
reader packet; they never establish that a proposal is readable.  A proposal
can only become reader material after it passes every mechanical gate and is
placed in an intact-prose versus shuffled-control study with blinded readers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]

WORD_RE = re.compile(r"[A-Za-z]+")


def normalize(text: str) -> str:
    return "".join(WORD_RE.findall(text.lower()))


def sentence_join(left: str, right: str) -> str:
    """Render two authored halves without manufacturing duplicate stops."""
    def finish(value: str) -> str:
        return value.rstrip().rstrip(".!?") + "."

    return f"{finish(left)} {finish(right)}"


def iter_rows(payload: object, source: str, _context_provenance: object = None) -> Iterable[dict]:
    """Yield rendered rows from the repository's append-only run formats."""
    if isinstance(payload, dict):
        context_provenance = _context_provenance or (
            payload.get("method")
            or payload.get("provenance")
            or payload.get("experiment")
            or payload.get("experiment_id")
            or payload.get("signature")
        )
        for key in (
            "rendered_probes",
            "rendered_candidates_and_probes",
            "rendered_candidates",
            "best_rendered_candidates",
            "rendered_rows",
            "repair_candidates",
            "instantiated_outputs",
            "repair_outputs",
            "failed_outputs",
            "exact_outputs",
            "records",
            "rows",
            "candidates",
            "probes",
            "repair",
            "rejected",
            "closed_leads",
            "mechanically_admitted_leads",
            "admitted",
            "attempts",
            # Ten-lane constructive audits keep their ordinary prose under
            # lane-specific names rather than flattening it into a generic
            # ``rows`` field.  Preserve those rows in the common report.
            "candidate_prose",
            "heldout_repairs",
            "lanes",
        ):
            values = payload.get(key)
            if isinstance(values, list):
                for row in values:
                    if isinstance(row, dict):
                        text = row.get("rendered") or row.get("text") or row.get("best_prose") or row.get("candidate")
                        # Character-LM tape resegmentation stores the two
                        # independently read surfaces as left/right fields.
                        # Joining them here keeps the rendered control visible
                        # without treating a failed resegmentation as exact.
                        if not text and isinstance(row.get("left"), str) and isinstance(row.get("right_resegmented"), str):
                            text = sentence_join(row["left"], row["right_resegmented"])
                        if not text and isinstance(row.get("audit"), dict):
                            text = row["audit"].get("rendered") or row["audit"].get("text")
                        if not text and isinstance(row.get("left"), str) and isinstance(row.get("right"), str):
                            text = sentence_join(row["left"], row["right"])
                        if isinstance(text, str) and text.strip():
                            item = {"source_run": source, **row, "rendered": text}
                            if context_provenance and "provenance" not in item:
                                item["provenance"] = context_provenance
                            yield item
        # A bounded repair artifact may keep its best base and held-out rows
        # as single dictionaries rather than list-valued phase fields.  Walk
        # those explicit surfaces so the shared aggregate cannot omit a
        # complete rendered repair.
        for key in ("repaired", "heldout", "bounded_repair", "second_bounded_repair"):
            row = payload.get(key)
            if isinstance(row, dict):
                text = row.get("rendered") or row.get("text") or row.get("best_prose")
                if isinstance(text, str) and text.strip():
                    item = {"source_run": source, **row, "rendered": text}
                    if context_provenance and "provenance" not in item:
                        item["provenance"] = context_provenance
                    yield item
        # Some append-only experiments keep base and repair phases as nested
        # objects.  Walk those phases so the diagnostic report cannot silently
        # omit their rendered probes.
        for phase in ("base", "repair"):
            nested = payload.get(phase)
            if isinstance(nested, dict):
                yield from iter_rows(nested, f"{source}#{phase}", context_provenance)
        # Bounded single-call authoring probes keep one captured model output
        # under ``captured_output`` rather than manufacturing a list wrapper.
        # Surface that row explicitly so repair evidence enters the same
        # candidate-first aggregate instead of disappearing from the report.
        captured = payload.get("captured_output")
        if isinstance(captured, dict):
            text = captured.get("rendered") or captured.get("text")
            if isinstance(text, str) and text.strip():
                item = {"source_run": source, **captured, "rendered": text}
                if context_provenance and "provenance" not in item:
                    item["provenance"] = context_provenance
                yield item
        # Newer bilateral constructors keep their single complete realization
        # under ``candidate``.  Treat it as a rendered row so the shared audit
        # cannot silently drop an otherwise fully provenance-backed probe.
        candidate = payload.get("candidate")
        if isinstance(candidate, dict):
            text = candidate.get("rendered") or candidate.get("text")
            if isinstance(text, str) and text.strip():
                item = {"source_run": source, **candidate, "rendered": text}
                if context_provenance and "provenance" not in item:
                    item["provenance"] = context_provenance
                yield item
        elif isinstance(candidate, str) and candidate.strip():
            # A compact constructive lane may retain one authored surface as
            # a top-level string.  It is still independently audited here.
            yield {
                "source_run": source,
                "rendered": candidate,
                "provenance": context_provenance,
                "lane": payload.get("experiment") or payload.get("experiment_id"),
            }
        # Compact character-level lanes may keep their one rendered surface
        # under ``rendered_prose`` rather than wrapping it in a candidate
        # object.  Preserve that prose in the common audit instead of
        # silently dropping a valid lane artifact.
        rendered_prose = payload.get("rendered_prose")
        if isinstance(rendered_prose, str) and rendered_prose.strip():
            yield {
                "source_run": source,
                "rendered": rendered_prose,
                "provenance": context_provenance,
                "lane": payload.get("experiment") or payload.get("experiment_id"),
            }
        # A few older artifacts store one candidate at the top level.
        text = payload.get("rendered") or payload.get("text")
        if isinstance(text, str) and text.strip():
            item = {"source_run": source, **payload, "rendered": text}
            if context_provenance and "provenance" not in item:
                item["provenance"] = context_provenance
            yield item
    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                text = row.get("rendered") or row.get("text")
                if isinstance(text, str) and text.strip():
                    yield {"source_run": source, **row, "rendered": text}


@lru_cache(maxsize=1)
def _brown_model():
    try:
        from experiments.audit_programmatic_readability import BrownBigramModel

        return BrownBigramModel.from_brown()
    except Exception:
        return None


def deterministic_shuffle_gain(words: list[str], seed: int, shuffles: int) -> float | None:
    """Return Brown order gain against shuffles, or ``None`` without Brown."""
    model = _brown_model()
    if model is None:
        return None
    observed = model.score(words)
    if observed is None:
        return None
    rng = random.Random(seed)
    baseline = []
    for _ in range(shuffles):
        shuffled = list(words)
        rng.shuffle(shuffled)
        score = model.score(shuffled)
        if score is not None:
            baseline.append(score)
    return observed - (sum(baseline) / len(baseline)) if baseline else None


def audit_row(row: dict, *, seed: int, shuffles: int) -> dict:
    text = row["rendered"]
    tape = normalize(text)
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    exact = bool(tape) and tape == tape[::-1]
    independent = bool(tape) and hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(tape[::-1].encode()).hexdigest()
    from llm_palindrome.admission import mechanical_admission_checks

    checks = mechanical_admission_checks(text, min_letters=39, max_letters=1000)
    diagnostics = {
        "brown_order_gain_vs_shuffle": deterministic_shuffle_gain(tokens, seed, shuffles),
        "mean_zipf_frequency": (sum(zipf_frequency(token, "en") for token in tokens) / len(tokens)) if tokens else None,
        "word_count": len(tokens),
        "repeated_word_rate": (1 - len(set(tokens)) / len(tokens)) if tokens else None,
        "punctuation_segments": len([segment for segment in re.split(r"[.!?]+", text) if WORD_RE.search(segment)]),
    }
    failures = sorted(key for key, value in checks.items() if not value)
    return {
        "rendered": text,
        "source_run": row["source_run"],
        "provenance": row.get("provenance") or row.get("method") or row.get("source") or "unspecified",
        "repair": row.get("repair"),
        "letters": len(tape),
        "exact_letter_palindrome": exact,
        "independent_sha256_exact": independent,
        "mechanical_checks": checks,
        "failed_checks": failures,
        "diagnostics_not_readability": diagnostics,
        "reader_next_test": (
            "Not reader-eligible until every mechanical check passes; if promoted, "
            "freeze this intact rendering with a matched word-shuffle control and "
            "randomized blinded rater order."
        ),
    }


def audit(paths: Iterable[Path], *, seed: int = 20260915, shuffles: int = 16) -> dict:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        for row in iter_rows(payload, str(path)):
            key = (row["source_run"], row["rendered"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(audit_row(row, seed=seed, shuffles=shuffles))
    route_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        route_rows[row["source_run"]].append(row)
    route_summary = []
    for route, route_items in sorted(route_rows.items()):
        zipf_values = [
            item["diagnostics_not_readability"]["mean_zipf_frequency"]
            for item in route_items
            if item["diagnostics_not_readability"]["mean_zipf_frequency"] is not None
        ]
        brown_values = [
            item["diagnostics_not_readability"]["brown_order_gain_vs_shuffle"]
            for item in route_items
            if item["diagnostics_not_readability"]["brown_order_gain_vs_shuffle"] is not None
        ]
        route_summary.append({
            "source_run": route,
            "rows": len(route_items),
            "exact_count": sum(item["exact_letter_palindrome"] for item in route_items),
            "mechanically_admitted_count": sum(
                all(item["mechanical_checks"].values()) for item in route_items
            ),
            "min_letters": min(item["letters"] for item in route_items),
            "max_letters": max(item["letters"] for item in route_items),
            "mean_zipf_frequency": sum(zipf_values) / len(zipf_values) if zipf_values else None,
            "mean_brown_order_gain_vs_shuffle": (
                sum(brown_values) / len(brown_values) if brown_values else None
            ),
        })
    return {
        "status": "diagnostic_not_human_readability_result",
        "method": {
            "exactness": "normalized ASCII tape equality plus independent SHA-256 reversal check",
            "mechanical_gate": "llm_palindrome.admission.mechanical_admission_checks",
            "local_order": "Brown add-alpha bigram gain against deterministic same-word shuffles",
            "seed": seed,
            "shuffles": shuffles,
        },
        "limits": [
            "Programmatic diagnostics do not certify grammar, coherent meaning, or human readability.",
            "Rows remain reader-ineligible until all mechanical checks pass and a blinded study is run.",
        ],
        "runs": [str(path) for path in paths],
        "candidate_count": len(rows),
        "exact_count": sum(row["exact_letter_palindrome"] for row in rows),
        "mechanically_admitted_count": sum(all(row["mechanical_checks"].values()) for row in rows),
        "route_summary": route_summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--shuffles", type=int, default=16)
    args = parser.parse_args()
    if args.shuffles < 2:
        parser.error("--shuffles must be at least 2")
    report = audit(args.runs, seed=args.seed, shuffles=args.shuffles)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("candidate_count", "exact_count", "mechanically_admitted_count")}, indent=2))


if __name__ == "__main__":
    main()
