#!/usr/bin/env python3
"""Rebuild a small, independently checked audit of selected week results.

The audit is intentionally bounded to named examples. It makes no claim that
the selected list is exhaustive or contains a global length maximum.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "2bdd7df301e185e67480f4cc3a0dbec43589b90b"
OUTPUT_JSON = ROOT / "paper/week_results.json"
OUTPUT_TABLE = ROOT / "paper/week_results_table.tex"
OUTPUT_EXAMPLES = ROOT / "paper/week_examples.tex"
OUTPUT_REPRESENTATIVE = ROOT / "paper/representative_output.tex"

# SHA-256 over the exact bytes of each tracked source file at SNAPSHOT.
SOURCE_SHA256 = {
    "runs/seed-np-cross-role-intersection-20260922.json": "606d7b9e3150a8e9ea845e0c669a741ebfe10bb58a18593a66c6bc17310f4ff3",
    "runs/overhang-growth-from-240-20261001.json": "83c755c8855f0260ccc0e75c428a473b1f38f2ef9fa0c70bdb0043ea93ea495a",
    "runs/incumbent-560-outer-causal-scene-20261002.json": "c2d520dd84f17164da62f691aab95cba57d351cd660ed611ac444fdb6679dc13",
    "runs/luna6-nora-aron-live-residual-growth-20260923.json": "54775109bbb8cfda710cffae3cb5345ad2e1729d0b324e0482c61a98bf1e4758",
    "runs/luna6-god-dog-live-residual-growth-20260923.json": "5f2df550ad7d4d919a4f42750b6679f9010cb94253c1baf1e86911c615822823",
    "runs/incumbent-568-live-event-chain-insertion-20260923.json": "621d4e8c9fdfc6bbc4217527c120b28854762a81d94c1a01be47e6bd76e0cabd",
    "runs/incumbent-672-discourse-linked-reverse-chain-20260922.json": "51435f9d8bacb1b2378663932a9c9261e4a9fab9eb46b43c5ecfee03e45fc878",
    "runs/repair-640-mirrored-shell-cycle-20260923.json": "4e9174e68397d9519cfa90cdb12dcbaa87a2a93bf4fa0d9a9f9e7ef0cab73f49",
    "runs/repair-686-mixed-predicate-shell-cycle-20260923.json": "77c38810fea972550a396ced140dd534f4004ac327fd2a7bdcec96ecbf2b9317",
    "runs/repair-736-center-lexical-event-path-20260923.json": "424d9d5fa3a4f8a450f149e3feb2d239f6116b4fe99c60e3328df58fda367a2d",
}

SELECTIONS = [
    {
        "id": "38-control",
        "source": "runs/seed-np-cross-role-intersection-20260922.json",
        "key": "incumbent_oracle.rendered",
        "mechanism": "inherited 38-letter reference/control; this audit does not establish authorship",
        "lineage": "loaded unchanged as incumbent_oracle; origin and authorship are not established by this audit",
        "reader_status": "not studied in a project reader study",
        "caveat": "Reference/control only; not evidence that this project generated a readable palindrome or that 38 is a maximum.",
    },
    {
        "id": "54-np-candidate",
        "source": "runs/seed-np-cross-role-intersection-20260922.json",
        "key": "rows[0].rendered",
        "mechanism": "bounded live-residual noun-phrase grammar intersection over an inherited open frame",
        "lineage": "retains the aide/rips/nine/memos and some/men/inspire/Diana frame; adds derived NP material without reflecting a closed span",
        "reader_status": "not run; frozen 24-rater packet has no responses",
        "caveat": "Mechanically admitted and flagged for study; unusual compounds and intended meaning still require reader evaluation.",
    },
    {
        "id": "498-overhang",
        "source": "runs/overhang-growth-from-240-20261001.json",
        "key": "rows[0].rendered",
        "mechanism": "bounded center-out overhang search from an inherited 240-letter seed",
        "lineage": "search-produced growth row; parent_artifact and growth_over_parent are recorded in the row",
        "reader_status": "not run",
        "caveat": "Exact construction with explicitly rough, unreviewed generated boundary syntax; not a readability example.",
    },
    {
        "id": "568-pinned",
        "source": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "key": "rows[id=outer-causal-scene-568-working-incumbent].rendered",
        "mechanism": "pinned working incumbent; authored outer-macro replacement with a causal scene",
        "lineage": "parent is the 560-letter central-event bridge",
        "reader_status": "not run",
        "caveat": "Working-length incumbent, not the longest selected result; proper palindromic spans, repeated units, and rough prose are documented debt.",
    },
    {
        "id": "616-nora-aron",
        "source": "runs/luna6-nora-aron-live-residual-growth-20260923.json",
        "key": "rendered_full_tape",
        "mechanism": "authored reciprocal-clause insertion at a live partial-word seam",
        "lineage": "separate direct child of the 568 pinned incumbent",
        "reader_status": "not run",
        "caveat": "Repeated reciprocal shell, self-palindromic token sees, and whole-token reversal pairs in the insertion are recorded debt.",
    },
    {
        "id": "630-god-dog",
        "source": "runs/luna6-god-dog-live-residual-growth-20260923.json",
        "key": "rendered_full_text",
        "mechanism": "authored six-clause insertion at a live partial-word seam",
        "lineage": "separate direct child of the 568 pinned incumbent",
        "reader_status": "not run",
        "caveat": "The inserted surface contains dog/god reversal and repeated identity frames; no readability claim.",
    },
    {
        "id": "640-event-chain",
        "source": "runs/incumbent-568-live-event-chain-insertion-20260923.json",
        "key": "candidate.rendered",
        "mechanism": "authored left event chain; bounded typed chart chooses a connected opposing resegmentation under a live residual",
        "lineage": "direct child of the 568 pinned incumbent",
        "reader_status": "not run",
        "caveat": "Repeated predicate and inherited rough discourse remain; exactness does not certify readability.",
    },
    {
        "id": "672-reverse-chain",
        "source": "runs/incumbent-672-discourse-linked-reverse-chain-20260922.json",
        "key": "rows[id=discourse-linked-reverse-chain-672].rendered",
        "mechanism": "bounded relation-chain search with per-clause reverse-character filtering",
        "lineage": "direct comparison child of the 568 pinned incumbent",
        "reader_status": "not run; saved AI review did not consider the full tape reader-worthy",
        "caveat": "Four-beat chains are schematic/repetitive; the run reports one accepted exact path from 9,273 examined states.",
    },
    {
        "id": "686-shell-cycle",
        "source": "runs/repair-640-mirrored-shell-cycle-20260923.json",
        "key": "candidate.rendered",
        "mechanism": "authored left event cycle; bounded deterministic grammar chart chooses opposing shell resegmentation",
        "lineage": "640 → 686 → 736 → 752 descendant branch from the 568 incumbent",
        "reader_status": "not run",
        "caveat": "Repeated sees-cycle and inherited repetition/rough discourse remain.",
    },
    {
        "id": "736-mixed-cycle",
        "source": "runs/repair-686-mixed-predicate-shell-cycle-20260923.json",
        "key": "candidate.rendered",
        "mechanism": "authored left mixed-predicate cycle; bounded deterministic grammar chart chooses opposing resegmentation",
        "lineage": "686 → 736 → 752 descendant branch from the 568 incumbent",
        "reader_status": "not run",
        "caveat": "Predicate repetition and inherited rough discourse remain; this is construction evidence.",
    },
    {
        "id": "752-center-path",
        "source": "runs/repair-736-center-lexical-event-path-20260923.json",
        "key": "candidate.rendered",
        "mechanism": "authored left two-event open path; bounded role chart chooses the connected opposing resegmentation",
        "lineage": "736 → 752; upstream branch is 568 → 640 → 686 → 736 → 752",
        "reader_status": "not run",
        "caveat": "Strongest exact-length endpoint in this selected audit; a central reversed event pair is a proper palindromic span, and prose/repetition debt remains.",
    },
]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
SENTENCE_RE = re.compile(r"[^.!?]*[.!?](?:[\"”’']*)|[^.!?]+$")


def git_bytes(revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def resolve_key(data: Any, key: str) -> Any:
    if key == "incumbent_oracle.rendered":
        return data["incumbent_oracle"]["rendered"]
    if key == "rows[0].rendered":
        return data["rows"][0]["rendered"]
    match = re.fullmatch(r"rows\[id=(.+)\]\.rendered", key)
    if match:
        row_id = match.group(1)
        return next(row["rendered"] for row in data["rows"] if row.get("id") == row_id)
    value: Any = data
    for part in key.split("."):
        value = value[part]
    return value


def ascii_letters(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text).lower()


def outside_in_exact(normalized: str) -> bool:
    left, right = 0, len(normalized) - 1
    while left < right:
        if normalized[left] != normalized[right]:
            return False
        left += 1
        right -= 1
    return True


def raw_text_outside_in_exact(text: str) -> bool:
    """Compare surviving ASCII letters directly, without building a tape."""
    left, right = 0, len(text) - 1
    while left < right:
        while left < right and not (text[left].isascii() and text[left].isalpha()):
            left += 1
        while left < right and not (text[right].isascii() and text[right].isalpha()):
            right -= 1
        if left < right and text[left].lower() != text[right].lower():
            return False
        left += 1
        right -= 1
    return True


def metrics(text: str) -> dict[str, Any]:
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    trigrams = Counter(tuple(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2)))
    trigram_total = sum(trigrams.values())
    trigram_excess = sum(count - 1 for count in trigrams.values() if count > 1)
    sentences = [part.strip() for part in SENTENCE_RE.findall(text) if part.strip()]
    sentence_keys = [" ".join(sentence.split()).casefold() for sentence in sentences]
    sentence_counts = Counter(sentence_keys)
    duplicate_sentences = sum(count - 1 for count in sentence_counts.values() if count > 1)
    return {
        "word_count": len(tokens),
        "unique_lowercase_word_count": len(set(tokens)),
        "repeated_trigram_excess_occurrences": trigram_excess,
        "trigram_occurrences": trigram_total,
        "repeated_trigram_rate": (trigram_excess / trigram_total) if trigram_total else 0.0,
        "sentence_count": len(sentences),
        "duplicate_sentence_count": duplicate_sentences,
    }


def escape_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "“": "``",
        "”": "''",
        "‘": "`",
        "’": "'",
    }
    return "".join(replacements.get(char, char) for char in text)


def make_lineage_replay(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe normalized-tape changes; this is not replay of generation."""
    by_id = {row["id"]: row for row in records}
    chain = ["568-pinned", "640-event-chain", "686-shell-cycle",
             "736-mixed-cycle", "752-center-path"]
    stages = []
    for parent_id, child_id in zip(chain, chain[1:]):
        parent = by_id[parent_id]["normalized_ascii_letters"]
        child = by_id[child_id]["normalized_ascii_letters"]
        matcher = SequenceMatcher(None, parent, child, autojunk=False)
        edits = [
            {"parent_span_half_open": [i1, i2],
             "old": parent[i1:i2], "new": child[j1:j2]}
            for tag, i1, i2, j1, j2 in matcher.get_opcodes()
            if tag != "equal"
        ]
        stages.append({
            "parent_id": parent_id,
            "child_id": child_id,
            "parent_letters": len(parent),
            "child_letters": len(child),
            "parent_sha256": hashlib.sha256(parent.encode("ascii")).hexdigest(),
            "child_sha256": hashlib.sha256(child.encode("ascii")).hexdigest(),
            "edits": edits,
        })
    return {
        "representation": "deterministic normalized-character diff; replays saved tapes, not candidate-generation procedures",
        "diff": "Python difflib.SequenceMatcher, autojunk=False; coordinates refer to the parent tape",
        "stages": stages,
    }


def build() -> dict[str, Any]:
    records = []
    for selection in SELECTIONS:
        source_path = selection["source"]
        source_file = ROOT / source_path
        snapshot_bytes = git_bytes(SNAPSHOT, source_path)
        current_bytes = source_file.read_bytes()
        expected_source_hash = SOURCE_SHA256[source_path]
        snapshot_source_hash = hashlib.sha256(snapshot_bytes).hexdigest()
        current_source_hash = hashlib.sha256(current_bytes).hexdigest()
        if snapshot_source_hash != expected_source_hash:
            raise AssertionError(f"Pinned source SHA mismatch at snapshot: {source_path}")
        if current_source_hash != expected_source_hash or current_bytes != snapshot_bytes:
            raise AssertionError(f"Current source differs from frozen tracked file: {source_path}")
        data = json.loads(current_bytes)
        surface = resolve_key(data, selection["key"])
        normalized = ascii_letters(surface)
        raw_exact = raw_text_outside_in_exact(surface)
        normalized_exact = bool(normalized) and outside_in_exact(normalized)
        if not raw_exact or not normalized_exact:
            raise AssertionError(f"Selected candidate failed independent raw-text or normalized outside-in check: {selection['id']}")
        if normalized != normalized[::-1]:
            raise AssertionError(f"Selected candidate failed full normalized reverse equality: {selection['id']}")
        digest = hashlib.sha256(normalized.encode("ascii")).hexdigest()
        source_claim = None
        if selection["id"] == "38-control":
            source_claim = data["incumbent_oracle"]["audit"]["sha256_forward"]
        elif selection["id"] == "54-np-candidate":
            source_claim = data["rows"][0]["independent_exact_audit"]["sha256_forward"]
        elif selection["id"] == "498-overhang":
            source_claim = data["rows"][0]["audit"]["sha256_forward"]
        elif selection["id"] == "568-pinned":
            source_claim = next(row["audit"]["sha256_forward"] for row in data["rows"] if row.get("id") == "outer-causal-scene-568-working-incumbent")
        elif selection["id"] == "672-reverse-chain":
            source_claim = next(row["independent_audit"]["sha256_forward"] for row in data["rows"] if row.get("id") == "discourse-linked-reverse-chain-672")
        elif selection["id"] == "616-nora-aron":
            source_claim = data["full_audit"]["forward_sha256"]
        elif selection["id"] == "630-god-dog":
            source_claim = data["validation"]["forward_sha256"]
        else:
            source_claim = data["independent_audit"]["sha256_forward"]
        if digest != source_claim:
            raise AssertionError(f"Independent hash does not match saved audit: {selection['id']}")
        records.append(
            {
                "id": selection["id"],
                "surface": surface,
                "normalized_ascii_letters": normalized,
                "letters": len(normalized),
                "normalized_sha256": digest,
                "independently_exact": True,
                "normalization": "re.sub('[^A-Za-z]', '', surface).lower()",
                "raw_text_outside_in_check": "passed",
                "normalized_full_reverse_equality": "passed",
                "metrics": metrics(surface),
                "mechanism": selection["mechanism"],
                "lineage": selection["lineage"],
                "reader_status": selection["reader_status"],
                "caveat": selection["caveat"],
                "source": {
                    "git_revision": SNAPSHOT,
                    "path": source_path,
                    "json_key": selection["key"],
                    "file_sha256": current_source_hash,
                    "tracked_at_snapshot": True,
                },
            }
        )

    return {
        "title": "Selected exact palindrome construction results",
        "snapshot": SNAPSHOT,
        "selection_note": "Named exemplars selected for audit; not an exhaustive scan and not a global maximum claim. Date-like suffixes in inherited run filenames are experiment identifiers, not asserted run dates.",
        "human_evidence": "No project human reader study has run for any selected row. The 38-letter item is an inherited reference/control whose authorship is not established by this audit. The 54-letter candidate's 24-rater packet is frozen but has no responses.",
        "definitions": {
            "normalization": "Lowercase ASCII letters only: re.sub('[^A-Za-z]', '', surface).lower().",
            "exactness": "Two distinct implementations in audit_week_results.py are required: a raw-text outside-in scan skips every character outside ASCII A-Z/a-z and compares lowercase surviving letters; separately, the re.sub-normalized tape must equal its full reversal. Its SHA-256 must also match the source artifact.",
            "word_count": "Regex [A-Za-z]+(?:'[A-Za-z]+)* over the full surface; straight ASCII apostrophes remain internal to words.",
            "unique_word_count": "Number of distinct case-folded tokens from the word-count regex.",
            "repeated_trigram_rate": "Excess token-trigram occurrences (each occurrence after the first for a repeated trigram) divided by all token-trigram windows across the full token sequence; sentence boundaries do not reset the sequence.",
            "duplicate_sentence_count": "Number of sentence occurrences after the first that exactly repeat an earlier case-insensitive sentence after whitespace normalization; sentence boundaries are terminal .?! punctuation, with immediately following quote marks attached, and punctuation is retained for duplicate comparison.",
        },
        "lineage_tape_replay": make_lineage_replay(records),
        "results": records,
    }


def render_table(records: list[dict[str, Any]]) -> str:
    labels = {
        "568-pinned": "Parent", "640-event-chain": "Edit 1",
        "686-shell-cycle": "Edit 2", "736-mixed-cycle": "Edit 3",
        "752-center-path": "Edit 4",
    }
    lineage_ids = list(labels)
    selected = {row["id"]: row for row in records}
    header = [
        r"\begin{table}[t]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Stage & $L$ & $\Delta L$ & Words/types & Rep. 3g & Dup. sent. \\",
        r"\midrule",
    ]
    rows = []
    previous_letters = None
    for record_id in lineage_ids:
        row = selected[record_id]
        m = row["metrics"]
        delta = "---" if previous_letters is None else f"+{row['letters'] - previous_letters}"
        repeat = f"{100 * m['repeated_trigram_rate']:.1f}\\%"
        rows.append(f"{labels[record_id]} & {row['letters']} & {delta} & "
                    f"{m['word_count']}/{m['unique_lowercase_word_count']} & {repeat} & "
                    f"{m['duplicate_sentence_count']} " + r"\\")
        previous_letters = row["letters"]
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Five exact tapes in the saved lineage. Types are distinct lowercase words. Repeat 3g is excess frequency of repeated three-token windows; duplicate sentences are exact case-insensitive repeats. These are not fluency measures.}",
        r"\label{tab:lineage}",
        r"\end{table}",
        "",
    ]
    return "\n".join(header + rows + footer)


def render_examples(records: list[dict[str, Any]]) -> str:
    selected = {row["id"]: row for row in records}
    order = ["752-center-path", "568-pinned", "630-god-dog", "672-reverse-chain"]
    out = ["% Generated by paper/audit_week_results.py from frozen tracked JSON."]
    for record_id in order:
        row = selected[record_id]
        out.extend(
            [
                r"\par\medskip",
                r"\Needspace{10\baselineskip}",
                f"\\noindent\\textbf{{{escape_tex(record_id)} ({row['letters']} letters)}}\\par",
                f"\\noindent Normalized SHA-256: \\path{{{row['normalized_sha256']}}}\\par",
                r"\begin{quote}\small\raggedright",
                escape_tex(row["surface"]),
                r"\end{quote}",
            ]
        )
    return "\n".join(out) + "\n"


def render_representative_output(records: list[dict[str, Any]]) -> str:
    """Render the audited longest lineage endpoint, with its quality status."""
    row = next(record for record in records if record["id"] == "752-center-path")
    if row["letters"] != 752 or not row["independently_exact"]:
        raise AssertionError("Representative endpoint must remain the audited exact 752-letter tape")
    return "\n".join(
        [
            "% Generated by paper/audit_week_results.py from frozen tracked JSON.",
            r"\par\medskip",
            r"\noindent\textbf{752-letter lineage endpoint (exact; rough prose, not reader-validated)}\par",
            f"{{\\raggedright\\noindent Normalized SHA-256: \\path{{{row['normalized_sha256']}}}\\par}}",
            r"\begin{quote}\small\raggedright",
            escape_tex(row["surface"]),
            r"\end{quote}",
            "",
        ]
    )


def main() -> None:
    result = build()
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_TABLE.write_text(render_table(result["results"]), encoding="utf-8")
    OUTPUT_EXAMPLES.write_text(render_examples(result["results"]), encoding="utf-8")
    OUTPUT_REPRESENTATIVE.write_text(render_representative_output(result["results"]), encoding="utf-8")
    print(json.dumps({"snapshot": SNAPSHOT, "rows": len(result["results"]), "outputs": [str(p.relative_to(ROOT)) for p in (OUTPUT_JSON, OUTPUT_TABLE, OUTPUT_EXAMPLES, OUTPUT_REPRESENTATIVE)]}))


if __name__ == "__main__":
    main()
