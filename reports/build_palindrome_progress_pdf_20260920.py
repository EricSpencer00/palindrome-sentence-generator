#!/usr/bin/env python3
"""Build the concise progress snapshot without collapsing exactness and readability."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/palindrome-progress-2026-09-20.pdf"


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def p(value: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(esc(value), style)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#617083"))
    canvas.drawString(0.55 * inch, 0.35 * inch, "Palindrome Sentence Generator | evidence snapshot | 2026-09-20")
    canvas.drawRightString(7.95 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=23, leading=27, textColor=colors.HexColor("#102a43"), spaceAfter=8)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#486581"), spaceAfter=10)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, leading=18, textColor=colors.HexColor("#102a43"), spaceBefore=8, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, leading=14, textColor=colors.HexColor("#176b87"), spaceBefore=6, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.4, leading=13, textColor=colors.HexColor("#243b53"), spaceAfter=6)
    small = ParagraphStyle("small", parent=body, fontSize=7.7, leading=9.8, spaceAfter=0)
    mono = ParagraphStyle("mono", parent=small, fontName="Courier", backColor=colors.HexColor("#f4f7fa"), borderColor=colors.HexColor("#d9e2ec"), borderWidth=0.4, borderPadding=6, spaceAfter=7)
    callout = ParagraphStyle("callout", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=colors.HexColor("#7b341e"), backColor=colors.HexColor("#fffaf0"), borderColor=colors.HexColor("#f6ad55"), borderWidth=0.7, borderPadding=8, spaceAfter=9)
    th = ParagraphStyle("th", parent=small, fontName="Helvetica-Bold", textColor=colors.white, fontSize=7.5, leading=9)

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.55 * inch, rightMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.58 * inch, title="Readable English palindrome progress", author="Palindrome Sentence Generator")
    story = [
        Paragraph("Readable English palindrome progress", title),
        Paragraph("Evidence update - 20 September 2026", sub),
        Paragraph("The project is still pursuing one thing: a long, genuinely readable English letter-level palindrome. Exactness is mechanically verified; readability is a blinded human judgment.", callout),
        Paragraph("Acceptance target", h1),
        Paragraph("An original, coherent, intact English palindrome longer than 38 letters, followed by randomized blinded intact-versus-shuffled reader evidence. No word-order symmetry, repeated/self-palindromic units, catalogue text, punctuation trick, fragment, or gibberish qualifies.", body),
        Paragraph("The claim that is safe to make", h1),
        Paragraph("<b>Best admitted exact candidate:</b> 38 letters. <b>Named v4 diagnostic:</b> 132 letters. Historical rendered artifacts reach 142 letters of exact gibberish, and synthetic exact fallback controls reach 100,001 letters. These are different scopes; none is currently original, coherent, shortcut-clean, and ready for readers.", body),
        Paragraph("Current benchmark", h2),
        Paragraph("An aide rips nine memos; some men inspire Diana.", mono),
        Paragraph("Normalized tape: anaideripsninememossomemeninspirediana | 38 letters | two-pointer exact = true | forward/reverse SHA-256 = ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6", small),
        Paragraph("What the public API inspired", h1),
        Paragraph("The live site demonstrates a valuable fast lexical mirror-state search and an inspectable word-to-mirror interaction. Its 1,198-letter demo is visibly word salad, not intact English prose. The active v4 design borrows the scalable character-orbit/index idea and the provenance-friendly inspection UX, then intersects it with typed grammar, agreement, valency, and semantic scene states before rendering a candidate.", body),
        PageBreak(),
        Paragraph("Exact diagnostics: retained, not promoted", title),
        Paragraph("Every row below is mechanically exact and independently audited, but each has a concrete rejection reason. None has been sent to readers as a success.", sub),
    ]
    diag = [
        [p("Length", th), p("Rendered diagnostic", th), p("Disposition", th)],
        [p("44", small), p("Was Noel an era, a gas, an item? Met in a, saga, arena, Leon saw.", small), p("Best human-looking diagnostic, but contains a proper self-palindromic multiword span and a fragmentary answer.", small)],
        [p("47", small), p("Some mad loss went save no level; one vast news sold a memos.", small), p("Malformed clause boundaries and determiner/noun agreement.", small)],
        [p("50", small), p("To new one post is an evening. Is sign in even as its open owe. Not.", small), p("Mechanically admitted, but incoherent and not reader-certified.", small)],
        [p("56", small), p("No evil Noel deliver desserts raw; war stressed reviled Leon live on.", small), p("Aligned whole-token semordnilap chain; withdrawn shortcut.", small)],
        [p("66", small), p("Erased on forever event is an evening. Is sign in even as it never ever. Of nodes are.", small), p("Hidden proper self-palindromic span; not admissible.", small)],
        [p("132", small), p("Name not left onto her a. Set add new one last one. Can all its an aide rips nine memos some men inspire Diana still an. Ace not sale now end dates. Are hot not felt one man.", small), p("Embeds the complete 38-letter anchor; explicit shortcut.", small)],
        [p("142", small), p("wanders remembers follows patient near remembers follows remembers patient still ll it st ne it ap sr eb me me rs wo ll of sr eb me me rr ae nt ne it aps wo ll of sr eb me me rs red naw", small), p("Historical open-vocabulary artifact; visibly fragmented/gibberish and never reader-eligible.", small)],
    ]
    table = Table(diag, colWidths=[0.55 * inch, 4.25 * inch, 2.0 * inch], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b87")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bcccdc")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fbfd"), colors.white]), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [table, Spacer(1, 8), Paragraph("Fresh constructions run in this update", h1)]
    fresh = [
        [p("Lane", th), p("Actual result", th), p("Next construction", th)],
        [p("Packed sentence-plan boundary DP", small), p("1,200 recursive plans packed to 195; four length-band buckets; 0 reverse-compatible states; 0 exact; three intact prose controls.", small), p("Add typed agreement features to the packed boundary state.", small)],
        [p("Corpus phrase WFSA", small), p("5,791 templates and 3,501 role/boundary index keys; 50,000 live states; 44,935 character prunes; 0 exact.", small), p("Add a typed finite-clause transition while keeping source-separated controls.", small)],
        [p("Semantic-frame character WFST", small), p("4 event frames x 4 settings; 80 best-first states and rendered controls to 83 letters; 0 exact.", small), p("Intersect variable-depth scene trees while carrying residual obligations.", small)],
        [p("Constructive scene lattice", small), p("81 indexed complete-prose states; 8 diagnostic controls; longest 103 letters; 0 exact.", small), p("Carry residual boundary classes across a second discourse relation edge.", small)],
        [p("Typed center-state WFST", small), p("36 typed scenes; 7 WFST states; explicit odd/even center parity and clause depth 0-2; 120 controls to 101 letters; 0 exact.", small), p("Increase typed clause-depth branching without dropping parity state.", small)],
        [p("Character-orbit scene residual", small), p("4 independent frame pairs; 160 controls; 40 mechanical exact rows, 0 exact-clean; 1,200 memoized states.", small), p("Add relative-clause and adjunct frames to the frame/word/residual product.", small)],
        [p("Lexical reverse automaton + latest Luna trio", small), p("Brown trie: 2,058 typed-clause renderings, 2,401 online states, 74 letters, 0 exact-clean. New role CSP / discourse graph / unequal-center lanes: 39 / 73 / 94-letter controls, all 0 exact-clean.", small), p("Expand held-out role classes; add temporal/instrumental roles, a relative edge, and ditransitive/relative-complement productions.", small)],
    ]
    t2 = Table(fresh, colWidths=[1.75 * inch, 3.25 * inch, 1.8 * inch], repeatRows=1)
    t2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b87")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bcccdc")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fbfd"), colors.white]), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [t2, PageBreak(), Paragraph("The actual reader-facing test", h1), Paragraph("The reproducible package contains intact prose and word-shuffled controls, randomized blinded order, separated answer keys, and a rater protocol. It remains closed to rejected exact diagnostics. The next candidate that passes exactness, provenance, and intact-prose review will enter that package; no programmatic score will substitute for readers.", body), Paragraph("Method and handoff", title), Paragraph("The API and paper now report the ledger without ambiguity:", sub)]
    method_rows = [
        [p("Field", th), p("Current value", th)],
        [p("best_admitted_exact_letters", small), p("38", small)],
        [p("longest_named_v4_diagnostic_letters", small), p("132", small)],
        [p("longest_rendered_historical_exact_artifact_letters", small), p("142 (gibberish)", small)],
        [p("longest_synthetic_exact_fallback_letters", small), p("100,001 (non-English control)", small)],
        [p("reader_evidence", small), p("Pending; no rejected diagnostic is promoted", small)],
        [p("generation policy", small), p("Constructive only: grammar, semantic roles, and mirrored character obligations are selected together; no post-hoc repair.", small)],
        [p("next reader-facing test", small), p("Randomized blinded intact-prose versus shuffled-control rating with reproducible provenance.", small)],
    ]
    t3 = Table(method_rows, colWidths=[2.45 * inch, 4.35 * inch], repeatRows=1)
    t3.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b87")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bcccdc")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fbfd"), colors.white]), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [t3, Spacer(1, 10), Paragraph("Recent committed method work", h1), Paragraph("0fc9f8dc packed sentence-plan boundary DP; 192edf45 character-orbit residual search; f6a4e140 lexical reverse-trie grammar; a46b1906 bilateral discourse graph; 600f7e2d unequal-center grammar; 4008fcd8 semantic-role live CSP.", body), Paragraph("Bottom line", h1), Paragraph("The public API is inspiring as a search/indexing and inspection reference. It does not change the success criterion: the next claimed win must be an actual long English palindrome that survives independent exact validation, novelty/shortcut checks, and blinded readers.", callout)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
