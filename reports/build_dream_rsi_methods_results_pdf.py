#!/usr/bin/env python3
"""Build the Dream-RSI methods/results report.

The report is deliberately evidence-first: exact rows that fail the
shortcut-free gate are shown as diagnostics, never as readable successes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/dream-rsi-methods-results-20260918.pdf"


def ascii_text(value: object) -> str:
    text = str(value)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def esc(value: object) -> str:
    text = ascii_text(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def para(value: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(esc(value), style)


def short_id(identifier: str) -> str:
    text = re.sub(r"-2026091[678]$", "", identifier)
    text = text.replace("-", " ")
    return text[:82]


def short_signature(signature: str) -> str:
    text = ascii_text(signature).replace("|", " / ")
    return text[:155]


def short_result(status: object) -> str:
    if not status:
        return "No run status recorded; preflight idea only."
    text = ascii_text(status)
    text = text.replace("; next repair is", "; next:")
    return text[:235]


def choose_methods(registry: dict) -> list[dict]:
    entries = registry.get("entries", [])
    eligible = [
        entry
        for entry in entries
        if entry.get("status") and not str(entry.get("id", "")).endswith("-excluded")
    ]
    # Preserve the registry's progression while guaranteeing that the newest
    # Dream-RSI constructions appear in the matrix.
    selected: list[dict] = []
    seen: set[str] = set()
    for entry in eligible[:40] + eligible[-20:]:
        identifier = str(entry.get("id", ""))
        if identifier and identifier not in seen:
            selected.append(entry)
            seen.add(identifier)
        if len(selected) == 50:
            break
    return selected


def load_json(path: str) -> dict:
    with (ROOT / path).open() as handle:
        return json.load(handle)


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = landscape(letter)
    canvas.setStrokeColor(colors.HexColor("#d7dee8"))
    canvas.setLineWidth(0.5)
    canvas.line(0.48 * inch, height - 0.42 * inch, width - 0.48 * inch, height - 0.42 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#617083"))
    canvas.drawString(0.52 * inch, height - 0.31 * inch, "DREAM-RSI METHODS AND RESULTS | 2026-09-18")
    canvas.drawRightString(width - 0.52 * inch, 0.29 * inch, f"Page {doc.page}")
    canvas.drawString(0.52 * inch, 0.29 * inch, "Exactness is necessary; programmatic scores never certify readability.")
    canvas.restoreState()


def build_pdf() -> None:
    registry = load_json("docs/experiment-novelty-registry.json")
    dual = load_json("runs/dream-rsi-dual-boundary-model-authoring-20260918.json")
    discourse = load_json("runs/dream-rsi-discourse-ellipsis-20260918.json")

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=27,
        textColor=colors.HexColor("#102a43"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#486581"),
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#102a43"),
        spaceBefore=8,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#176b87"),
        spaceBefore=5,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12.5,
        textColor=colors.HexColor("#243b53"),
        spaceAfter=6,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=7.5,
        leading=9.2,
        spaceAfter=0,
    )
    tiny = ParagraphStyle(
        "Tiny",
        parent=body,
        fontSize=6.7,
        leading=8.1,
        spaceAfter=0,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=small,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        fontSize=7.4,
        leading=8.8,
    )
    callout = ParagraphStyle(
        "Callout",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#7b341e"),
        backColor=colors.HexColor("#fffaf0"),
        borderColor=colors.HexColor("#f6ad55"),
        borderWidth=0.7,
        borderPadding=8,
        spaceBefore=4,
        spaceAfter=8,
    )
    mono = ParagraphStyle(
        "Mono",
        parent=small,
        fontName="Courier",
        fontSize=7.3,
        leading=9.4,
        textColor=colors.HexColor("#243b53"),
        backColor=colors.HexColor("#f4f7fa"),
        borderColor=colors.HexColor("#d9e2ec"),
        borderWidth=0.4,
        borderPadding=6,
        spaceAfter=7,
    )

    doc = BaseDocTemplate(
        str(OUT),
        pagesize=landscape(letter),
        leftMargin=0.52 * inch,
        rightMargin=0.52 * inch,
        topMargin=0.56 * inch,
        bottomMargin=0.50 * inch,
        title="Dream-RSI Methods and Results",
        author="Palindrome Sentence Generator",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])

    story = []
    story.append(Paragraph("Dream-RSI Methods and Results", title))
    story.append(
        Paragraph(
            "A construction log for long, genuinely readable English letter-level palindromes | Snapshot: 2026-09-18",
            subtitle,
        )
    )
    story.append(
        Paragraph(
            "Current status: no reader-worthy output has cleared the full gate yet. This report is intentionally not a negative-result thesis: it records the working search machinery, the exact diagnostic surfaces, and the next geometry-changing repair required to reach a long readable result.",
            callout,
        )
    )

    story.append(Paragraph("1. Acceptance gate", h1))
    story.append(
        Paragraph(
            "A candidate is admitted only when (1) normalized letters equal their reverse, verified by an independent two-pointer check and an independent forward/reverse SHA-256 check; (2) provenance shows fresh construction rather than catalogue text, finished-tape reversal, word-order symmetry, repeated units, or self-palindromic words/spans; (3) the surface is intact ordinary English prose; and (4) blinded human readers prefer the intact passage to a randomized word-shuffled control. Programmatic language measures may filter or diagnose, but never certify readability.",
            body,
        )
    )
    story.append(Paragraph("2. Dream-RSI controller", h1))
    story.append(
        Paragraph(
            "Each completed construction lane writes a replay node: rendered surface, normalized tape, exactness hashes, mismatch mask, grammar and semantic state, provenance flags, and a concrete next repair. Replay selects a new operator on a held-out split; a fresh branch then executes that operator. A failure is progress only when it changes the construction geometry and names the next repair. The controller is an orchestration layer, never a readability judge.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The current live operator freezes matched outer character assignments, reopens the seam-owning typed constituents, and lets lexical content, inflection, and word boundaries change while carrying the mirrored character obligation immediately. The next branch is a single discourse grammar: a complete question plus a complete answer clause generated jointly across the seam, with all proper palindromic subspans forbidden.",
            body,
        )
    )

    story.append(Paragraph("3. Actual Dream-RSI results", h1))
    round_data = [
        [para("Round / lane", table_header), para("Fresh work", table_header), para("Exact / admitted", table_header), para("Reader status", table_header)],
        [para("Model-guided span resynthesis", small), para("12 local GPT-2 seam proposals; exact rows were constructed by mirroring the residual; 86-120 letters.", small), para("12 exact by construction / 0 shortcut-free", small), para("Human-unreviewed; reflected residual had no lexical segmentation.", small)],
        [para("Replay round 119", small), para("21,990 independently audited nodes; 120 exact-but-fragment controls entered replay.", small), para("0 admissible exact", small), para("No reader gate; held-out mismatch-first stayed 0.780.", small)],
        [para("Dual-boundary model authoring", small), para("8 left x 8 right model proposals; 64 fresh paired worlds; no seed scaffold in output.", small), para("0 exact; longest 58 letters", small), para("Human-unreviewed; next repair reopens a complete residual word.", small)],
        [para("Discourse ellipsis + cross-word seam", small), para("148 exact diagnostic rows, 42-46 letters; 100 passed word-order check, but all retained rows have a forbidden proper palindromic island.", small), para("148 exact / 0 strict", small), para("Reader gate closed; next repair jointly authors a complete answer clause.", small)],
    ]
    t = Table(round_data, colWidths=[1.62 * inch, 4.25 * inch, 1.55 * inch, 2.75 * inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b87")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bcccdc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fbfd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fbfd"), colors.white]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 7))
    story.append(Paragraph("The strongest exact diagnostic surface", h2))
    story.append(
        Paragraph(
            "Independent audit: 44 normalized letters; two-pointer exact = true; forward SHA-256 = reverse SHA-256 = 0419d58b6773dcca6d527271d41fd0197da9dde876598c9dadfc87fa4f1a9f83. It is not admitted: the surface contains a proper palindromic island and therefore fails the shortcut-free gate. It has not been sent to readers.",
            body,
        )
    )
    story.append(Paragraph("Was Noel an era, a gas, an item? Met in a, saga, arena, Leon saw.", mono))
    story.append(
        Paragraph(
            "This row is useful because it demonstrates genuine cross-word boundary resegmentation (an era / arena, a gas / saga, an item / met in a) while also exposing the remaining failure mode: an exact tape can still be an engineered word-mirror or contain a proper palindrome island. The construction must now generate the whole discourse clause jointly and reject every such subspan before closure.",
            body,
        )
    )

    story.append(Paragraph("4. Fifty distinct construction methods and results", h1))
    story.append(
        Paragraph(
            "The matrix below is extracted from the novelty registry. Each row is a distinct construction signature, not a repeated parameter sweep. Results preserve the registry's wording; entries marked preflight have no run result and are not evidence of success. Exact diagnostic rows remain quarantined unless every shortcut-free gate passes.",
            body,
        )
    )

    methods = choose_methods(registry)
    rows = [[
        para("#", table_header),
        para("Method / operator", table_header),
        para("Construction signature", table_header),
        para("Observed result and next repair", table_header),
    ]]
    for idx, entry in enumerate(methods, 1):
        status = entry.get("status") or "No run status recorded; preflight idea only."
        rows.append([
            para(idx, tiny),
            para(short_id(str(entry.get("id", ""))), tiny),
            para(short_signature(str(entry.get("signature", ""))), tiny),
            para(short_result(status), tiny),
        ])
    matrix = Table(rows, colWidths=[0.28 * inch, 2.25 * inch, 3.15 * inch, 4.35 * inch], repeatRows=1)
    matrix.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102a43")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(matrix)
    story.append(Spacer(1, 8))
    story.append(PageBreak())
    story.append(Paragraph("5. Reader-facing protocol and next constructive repair", h1))
    story.append(
        Paragraph(
            "No row has reached the reader study in this snapshot. When one does, the package will contain the intact passage, a word-shuffled control, randomized blinded order, the exact normalization and hash audit, provenance and generator commit, and a reproducible rater form. The immediate next experiment is not another sweep: jointly generate a complete question-answer discourse across the seam, carry agreement and valency state, and reject any candidate containing a proper palindromic subspan before it can be considered for readers.",
            body,
        )
    )
    story.append(Paragraph("Reproducibility anchors", h2))
    story.append(
        Paragraph(
            "Source artifacts: docs/experiment-novelty-registry.json; runs/dream-rsi-dual-boundary-model-authoring-20260918.json; runs/dream-rsi-discourse-ellipsis-20260918.json. The report is generated from these files so the table and headline counts can be regenerated after the next Dream-RSI branch.",
            body,
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build_pdf()
