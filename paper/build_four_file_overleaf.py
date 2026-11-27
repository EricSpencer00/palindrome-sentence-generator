"""Build the minimal four-file anonymous Overleaf source package.

The canonical manuscript keeps its external figures. This export-only variant
replaces those two includegraphics commands with TikZ/PGFPlots figures built
from the archived analysis.json, then packages exactly four files.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ANALYSIS = ROOT / "runs/fragment-penalty-20261002/analysis.json"
OUT = ROOT / "output/naacl-submission/overleaf-v1-four-files"
ZIP = ROOT / "output/naacl-submission/overleaf-v1-four-files.zip"
ARMS = ("0", "4", "16")


def fmt(value: float) -> str:
    return f"{value:.5f}".rstrip("0").rstrip(".")


def tradeoff_figure(data: dict) -> str:
    arms = [data["arms"][p] for p in ARMS]
    exact = []
    exact_errors = []
    strict = []
    strict_errors = []
    distinct = []
    distance = []
    distance_errors = []
    distance_labels = []
    for i, (p, arm) in enumerate(zip(ARMS, arms)):
        exact.append(f"({i-0.17:.2f},{arm['exact_yield']*100:.2f})")
        strict.append(f"({i+0.17:.2f},{arm['strict_yield']*100:.2f})")
        eci = arm["intervals_95"]["exact_yield"]
        sci = arm["intervals_95"]["strict_yield"]
        exact_x = i - 0.17
        strict_x = i + 0.17
        exact_errors.extend([
            rf"\draw[black,line width=.45pt] (axis cs:{exact_x:.2f},{eci[0]*100:.2f}) -- (axis cs:{exact_x:.2f},{eci[1]*100:.2f});",
            rf"\draw[black,line width=.45pt] (axis cs:{exact_x-.035:.2f},{eci[0]*100:.2f}) -- (axis cs:{exact_x+.035:.2f},{eci[0]*100:.2f});",
            rf"\draw[black,line width=.45pt] (axis cs:{exact_x-.035:.2f},{eci[1]*100:.2f}) -- (axis cs:{exact_x+.035:.2f},{eci[1]*100:.2f});",
        ])
        strict_errors.extend([
            rf"\draw[black,line width=.45pt] (axis cs:{strict_x:.2f},{sci[0]*100:.2f}) -- (axis cs:{strict_x:.2f},{sci[1]*100:.2f});",
            rf"\draw[black,line width=.45pt] (axis cs:{strict_x-.035:.2f},{sci[0]*100:.2f}) -- (axis cs:{strict_x+.035:.2f},{sci[0]*100:.2f});",
            rf"\draw[black,line width=.45pt] (axis cs:{strict_x-.035:.2f},{sci[1]*100:.2f}) -- (axis cs:{strict_x+.035:.2f},{sci[1]*100:.2f});",
        ])
        distinct.append(f"({i},{arm['distinct_texts']})")
        lo, hi = arm["intervals_95"]["mean_pairwise_trigram_jaccard_distance"]
        val = arm["mean_pairwise_trigram_jaccard_distance"]
        distance.append(f"({i},{fmt(val)})")
        arm_color = {"0": "armzero", "4": "armfour", "16": "armsixteen"}[p]
        distance_errors.extend([
            rf"\draw[{arm_color},line width=1pt] (axis cs:{i},{fmt(lo)}) -- (axis cs:{i},{fmt(hi)});",
            rf"\draw[{arm_color},line width=1pt] (axis cs:{i-0.035},{fmt(lo)}) -- (axis cs:{i+0.035},{fmt(lo)});",
            rf"\draw[{arm_color},line width=1pt] (axis cs:{i-0.035},{fmt(hi)}) -- (axis cs:{i+0.035},{fmt(hi)});",
        ])
        distance_labels.append(f"\\node[font=\\scriptsize,anchor=south] at (axis cs:{i},{fmt(val+0.014)}) {{{val:.3f}}};")

    exact_coords = " ".join(exact)
    strict_coords = " ".join(strict)
    distance_coords = " ".join(distance)
    exact_error_coords = "\n".join(exact_errors)
    strict_error_coords = "\n".join(strict_errors)
    distance_error_coords = "\n".join(distance_errors)
    distinct_labels = "\n".join(
        rf"\node[font=\scriptsize,anchor=south] at (axis cs:{i},{arm['distinct_texts']+2}) {{{arm['distinct_texts']}}};"
        for i, arm in enumerate(arms)
    )
    return rf"""\begin{{tikzpicture}}
\pgfplotsset{{every axis/.append style={{every axis plot/.append style={{mark=none}},height=2.15cm,scale only axis,tick label style={{font=\scriptsize}},label style={{font=\scriptsize}},title style={{font=\scriptsize}},ymajorgrids=true,grid style={{gray!18,line width=.3pt}}}}}}
\begin{{axis}}[
  at={{(0,0)}},anchor=south west,width=0.27\textwidth,height=2.15cm,
  scale only axis,
  tick label style={{font=\scriptsize}},label style={{font=\scriptsize}},
  title style={{font=\small,align=left}},
  xtick={{0,1,2}},xticklabels={{0,4,16}},
  axis line style={{gray!70,line width=.35pt}},tick style={{gray!70,line width=.35pt}},
  ymajorgrids=true,grid style={{gray!18,line width=.3pt}},
  clip=false,
  title={{a  Coverage and strict checks (\%)}},
  ymin=0,ymax=100,ytick={{0,25,50,75,100}},
  legend style={{at={{(0.5,-0.23)}},anchor=north,draw=none,font=\tiny,legend columns=2,column sep=3pt}}
]
\addplot+[ybar,bar width=4pt,draw=none,fill=armzero] coordinates {{{exact_coords}}};
\addplot+[ybar,bar width=4pt,draw=none,fill=armfour] coordinates {{{strict_coords}}};
{exact_error_coords}
{strict_error_coords}
\legend{{Basic exact,Strict}}
\end{{axis}}
\begin{{axis}}[at={{(0.34\textwidth,0)}},anchor=south west,width=0.27\textwidth,height=2.15cm,
  title={{b  Distinct outputs}},
  xtick={{0,1,2}},xticklabels={{0,4,16}},ymin=0,ymax=100,ytick={{0,50,100}}]
\addplot+[ybar,bar width=11pt,draw=none,fill=armzero] coordinates {{{distinct[0]}}};
\addplot+[ybar,bar width=11pt,draw=none,fill=armfour] coordinates {{{distinct[1]}}};
\addplot+[ybar,bar width=11pt,draw=none,fill=armsixteen] coordinates {{{distinct[2]}}};
{distinct_labels}
\end{{axis}}
\begin{{axis}}[at={{(0.68\textwidth,0)}},anchor=south west,width=0.27\textwidth,height=2.15cm,
  title={{c  Trigram distance}},
  xtick={{0,1,2}},xticklabels={{0,4,16}},ymin=0.80,ymax=1.00,yticklabel=\empty,ytick style={{draw=none}},
  xmin=-0.25,xmax=2.25]
\addplot+[only marks,mark=*,mark size=1.7pt,color=armzero] coordinates {{{distance_coords}}};
{distance_error_coords}
{''.join(distance_labels)}
\end{{axis}}
\end{{tikzpicture}}"""


def rgb_blend(v: float) -> str:
    low = (247, 251, 255)
    high = (8, 48, 107)
    rgb = tuple(round(a + (b - a) * max(0.0, min(1.0, v))) for a, b in zip(low, high))
    return "".join(f"{channel:02X}" for channel in rgb)


def motif_figure(data: dict) -> str:
    counts = {p: dict(data["curves"][p]["all_trigram_counts"]) for p in ARMS}
    names = list(dict.fromkeys(
        [term for p in ("0", "4") for term, _ in data["curves"][p]["all_trigram_counts"][:4]]
    ))
    distinct = {p: data["arms"][p]["distinct_texts"] for p in ARMS}
    rows = []
    labels = []
    for i, name in enumerate(names):
        labels.append(rf"\node[anchor=east,font=\scriptsize] at (0,{7-i}) {{{name}}};")
        for j, p in enumerate(ARMS):
            val = counts[p].get(name, 0) / distinct[p]
            color = rgb_blend(val)
            text_color = "white" if val > 0.55 else "black"
            rows.append(
                rf"\fill[HTML]"  # replaced below with a named color for TikZ portability
            )
            color_name = f"motif{len(rows)}"
            rows[-1] = rf"\definecolor{{{color_name}}}{{HTML}}{{{color}}}\fill[{color_name}] ({j},{7-i}) rectangle ++(0.82,0.82);\node[font=\scriptsize,text={text_color}] at ({j+0.41},{7-i+0.41}) {{{val*100:.0f}\%}};"
    curve_lines = []
    for p in ARMS:
        entries = data["curves"][p]["all_trigram_counts"][:40]
        coords = " ".join(
            f"({i+1},{n/distinct[p]*100:.3f})" for i, (_term, n) in enumerate(entries)
        )
        color_name = {"0": "armzero", "4": "armfour", "16": "armsixteen"}[p]
        curve_lines.append(rf"\addplot+[no marks,line width=.8pt,color={color_name}] coordinates {{{coords}}};")
    category_labels = "".join(
        rf"\node[font=\scriptsize,anchor=north,align=center] at ({j+0.41},-0.08) {{{p}\\(n={distinct[p]})}};"
        for j, p in enumerate(ARMS)
    )
    return rf"""\begin{{tikzpicture}}[x=0.72cm,y=0.42cm]
\begin{{scope}}
  \node[anchor=west,font=\small] at (0,8.15) {{a  Trigram prevalence}};
  {''.join(labels)}
  {''.join(rows)}
  {category_labels}
  \node[font=\scriptsize,anchor=north] at (1.23,-1.2) {{Fragment penalty (distinct texts)}};
\end{{scope}}
\begin{{scope}}[xshift=5.1cm]
  \begin{{axis}}[
    at={{(0,0)}},anchor=south west,width=6.9cm,height=3.3cm,scale only axis,
    title={{b  Trigram concentration}},title style={{font=\small,align=left}},
    xlabel={{Trigram rank within each arm}},ylabel={{Distinct texts containing trigram (\%)}},
    xmin=1,xmax=40,ymin=0,ymax=100,xtick={{1,10,20,30,40}},ytick={{0,25,50,75,100}},
    tick label style={{font=\scriptsize}},label style={{font=\scriptsize}},
    axis line style={{gray!70,line width=.35pt}},tick style={{gray!70,line width=.35pt}},
    ymajorgrids=true,grid style={{gray!18,line width=.3pt}},
    legend style={{draw=none,font=\scriptsize,at={{(0.98,0.98)}},anchor=north east}}
  ]
\addplot[gray!70,dashed,line width=.4pt,forget plot] coordinates {{(1,50)(40,50)}};
  {''.join(curve_lines)}
  \legend{{Penalty 0,Penalty 4,Penalty 16}}
  \end{{axis}}
\end{{scope}}
\end{{tikzpicture}}"""


def build() -> None:
    data = json.loads(ANALYSIS.read_text())
    source = (PAPER / "naacl2027.tex").read_text()
    source = source.replace("\\usepackage{graphicx}", "\\usepackage{graphicx}\n\\usepackage{tikz}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}")
    source = source.replace("\\pgfplotsset{compat=1.18}", "\\pgfplotsset{compat=1.18}\n\\definecolor{armzero}{HTML}{246580}\n\\definecolor{armfour}{HTML}{BD6338}\n\\definecolor{armsixteen}{HTML}{707B65}")
    replacements = {
        r"\includegraphics[width=\textwidth]{fig/fragment-penalty-tradeoff.pdf}": tradeoff_figure(data),
        r"\includegraphics[width=\textwidth]{fig/fragment-penalty-motifs.pdf}": motif_figure(data),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"Expected one figure reference: {old}")
        source = source.replace(old, new)
    if r"\includegraphics" in source or re.search(r"fig/[^}\s]+\.(?:pdf|png)", source):
        raise ValueError("Generated export still depends on an external figure")
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "naacl2027.tex": source.encode(),
        "refs.bib": (PAPER / "refs.bib").read_bytes(),
        "acl.sty": (PAPER / "acl.sty").read_bytes(),
        "acl_natbib.bst": ("\n".join(line.rstrip() for line in
                          (PAPER / "acl_natbib.bst").read_text().splitlines()) + "\n").encode(),
    }
    for child in OUT.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            raise ValueError(f"Unexpected directory in exact-four-file export: {child}")
    for name, payload in files.items():
        (OUT / name).write_bytes(payload)
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    with zipfile.ZipFile(ZIP) as archive:
        if len(archive.namelist()) != 4:
            raise AssertionError("Overleaf archive must contain exactly four files")
        if set(archive.namelist()) != set(files):
            raise AssertionError("Unexpected Overleaf archive entries")
    if set(path.name for path in OUT.iterdir()) != set(files):
        raise AssertionError("Export directory must contain exactly four files")
    print(f"Built {OUT} and {ZIP}; four files in each export")


if __name__ == "__main__":
    build()
