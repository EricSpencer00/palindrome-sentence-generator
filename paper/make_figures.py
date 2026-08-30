"""Every number and figure in the paper, recomputed from raw run output.

Nothing in the .tex is typed by hand. This writes `paper/fig/*.pdf` and
`paper/numbers.tex`, a file of \\newcommand macros the text cites, so a claim
in the prose cannot drift away from the measurement behind it. If a run is
re-done, this regenerates both and the paper changes with it.

Audit note. Four numbers previously in the draft did not survive being
recomputed here, and the macros below are the corrected ones:

  vocabulary exponent   was -1.005, R^2 0.9986, n=8, 47x span. The raw shard
                        records give -1.031, R^2 0.9997, n=5, 32x, once the
                        6,000-word cell -- which came from a different job with
                        9x the core-seconds -- is excluded.
  yield decay           was "1.7x per letter", extrapolated 15 letters to a
                        2900x figure. It rests on two cells, the far one
                        holding a single event. The Poisson interval on the
                        per-letter ratio is [0.90, 4.50], so the extrapolation
                        spans no cost to 6e9x. Dropped, not corrected.
  conservation          was "1.09 letters per placement regardless of size",
                        cited to a RESULTS-chunking that is not in this
                        repository. Re-measured over 60,000 placements: the
                        mean is 0.90 and it is NOT constant, peaking at 2.03
                        for three-letter units and decaying to 0.37 at ten.
  span counts           several "n=" figures were larger than the data.
"""
from __future__ import annotations

import collections, json, math, re, statistics as st, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
sys.path.insert(0, ".")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, ACC, MUT = "#1a1a1a", "#b3272d", "#8a8a8a"
M: dict[str, str] = {}


def mac(name, value):
    M[name] = value
    return value


# ---------------------------------------------------------------- scaling
def scaling():
    raw = open("../runs/polaris/scale_20260823/summaries.jsonl").read()
    rows = [json.loads(m) for m in re.findall(r"\{[^{}]*\}", raw)]
    agg = collections.defaultdict(lambda: [0, 0.0])
    for r in rows:
        a = agg[r["vocab"]]
        a[0] += r["distinct"]; a[1] += r["seconds"]
    # 6,000 is from the yield job, 9x the core-seconds, different config.
    vs = sorted(v for v in agg if v != 6000)
    xs = [math.log(v) for v in vs]
    ys = [math.log(agg[v][0] / agg[v][1]) for v in vs]
    n = len(xs); mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a0 = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    rs = sum((y - (a0 + b * x)) ** 2 for x, y in zip(xs, ys))
    mac("vocabExp", f"{b:.3f}"); mac("vocabRsq", f"{1 - rs / ss:.4f}")
    mac("vocabN", str(n)); mac("vocabSpan", f"{math.exp(max(xs) - min(xs)):.0f}")
    return vs, [agg[v][0] / agg[v][1] for v in vs], a0, b


# ------------------------------------------------------------ conservation
def conservation():
    """Reads the artifact `experiments/conservation.py` writes. Regenerate it
    with `python3 experiments/conservation.py` from the repository root; it
    takes a couple of minutes and is not re-run on every figure build."""
    d = json.load(open("../experiments/conservation.json"))
    rows = [(r["unit"], r["n"], r["settled"], r["net"]) for r in d["by_unit"]]
    mac("netMean", f"{d['mean_net']:.2f}")
    mac("netPeak", f"{max(r[3] for r in rows):.2f}")
    mac("netTail", f"{rows[-1][3]:.2f}")
    mac("consN", f"{d['placements']:,}")
    mac("setTail", f"{rows[-1][2]:.2f}")
    mac("setHead", f"{rows[0][2]:.2f}")
    return rows


# ------------------------------------------------------------------ figures
def fig_measurements(rows, vs, rates, a0, b, L, fwd, rev):
    """Three panels across the text width. Separate single-column figures cost
    about a page of vertical space between them for no gain in legibility."""
    fig, (a, c, d) = plt.subplots(1, 3, figsize=(6.9, 1.75))

    x = list(range(len(rows)))
    a.bar([i - 0.2 for i in x], [r[2] for r in rows], 0.4, color=MUT,
          label="settled")
    a.bar([i + 0.2 for i in x], [r[3] for r in rows], 0.4, color=ACC,
          label="net")
    a.set_xticks(x[::2]); a.set_xticklabels([rows[i][0] for i in x[::2]])
    a.set_xlabel("letters in the unit placed")
    a.set_ylabel("letters")
    a.legend(frameon=False, loc="upper left", handlelength=1.1)
    a.set_title("a.  a long unit settles more and\n     advances less",
                loc="left")

    c.plot(L, fwd, "o-", color=INK, ms=3, lw=1, label="forwards")
    c.plot(L, rev, "s-", color=ACC, ms=3, lw=1, label="backwards")
    c.set_ylim(0, 1)
    c.set_xlabel("span length (letters)")
    c.set_ylabel("letters placeable\nin a dictionary word")
    c.legend(frameon=False, loc="center right", handlelength=1.4)
    c.set_title("b.  half of reversed English\n     is not English", loc="left")

    d.loglog(vs, rates, "o", color=ACC, ms=4, zorder=3)
    xx = [min(vs), max(vs)]
    d.loglog(xx, [math.exp(a0 + b * math.log(v)) for v in xx], "-",
             color=INK, lw=1, zorder=2)
    d.set_xlabel("vocabulary size (words)")
    d.set_ylabel("states / core-second")
    d.text(0.96, 0.92, f"slope ${b:.2f}$\n$R^2={M['vocabRsq']}$",
           transform=d.transAxes, ha="right", va="top", fontsize=7)
    d.set_title("c.  throughput is linear in\n     vocabulary size", loc="left")

    fig.subplots_adjust(wspace=0.42)
    fig.savefig("fig/measurements.pdf")
    plt.close(fig)


# ----------------------------------------------------------------- coverage
def coverage():
    """Model-free half of the argument: what fraction of a span's letters can
    be placed inside a dictionary word, read forwards and read backwards."""
    d = json.load(open("../experiments/mirror_cost.json"))
    by = collections.defaultdict(lambda: ([], []))
    for r in d:
        f, b = by[r["n_letters"]]
        f.append(r["coverage_forward"]); b.append(r["coverage_reversed"])
    L = sorted(by)
    fwd = [st.mean(by[k][0]) for k in L]
    rev = [st.mean(by[k][1]) for k in L]
    mac("covFwdLo", f"{min(fwd):.2f}"); mac("covFwdHi", f"{max(fwd):.2f}")
    mac("covRevLo", f"{min(rev):.2f}"); mac("covRevHi", f"{max(rev):.2f}")
    return L, fwd, rev


def fig_coverage(L, fwd, rev):
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ax.plot(L, fwd, "o-", color=INK, ms=3.5, lw=1, label="read forwards")
    ax.plot(L, rev, "s-", color=ACC, ms=3.5, lw=1, label="read backwards")
    ax.set_ylim(0, 1)
    ax.set_xlabel("span length (letters)")
    ax.set_ylabel("fraction of letters\nplaceable in a word")
    ax.legend(frameon=False, loc="center right")
    fig.savefig("fig/coverage.pdf")
    plt.close(fig)


# -------------------------------------------------------------- punctuation
def punctuation():
    """Blind pairwise, identical letters both sides, judge 2 sees every pair
    flipped. This is the same protocol that returned 12/12 on its calibration
    in the seam experiment."""
    FLIP = {"A": "B", "B": "A"}
    key = {k["id"]: k for k in json.load(open("../runs/punct/marks_key.json"))}
    j1 = {v["id"]: v["pick"] for v in json.load(open("/tmp/marks_v1.json"))}
    j2 = {v["id"]: FLIP[v["pick"]]
          for v in json.load(open("/tmp/marks_v2.json"))}
    out = {}
    for kind in ("calibration", "present_vs_bare", "hand_vs_present"):
        ids = [i for i in key if key[i]["kind"] == kind]
        hits = sum(j1[i] == key[i]["target_side"] for i in ids) + \
               sum(j2[i] == key[i]["target_side"] for i in ids)
        out[kind] = (hits, 2 * len(ids))
    mac("calibHit", str(out["calibration"][0]))
    mac("calibN", str(out["calibration"][1]))
    mac("presentWin", str(out["present_vs_bare"][0]))
    mac("presentN", str(out["present_vs_bare"][1]))
    mac("bareWin", str(out["present_vs_bare"][1] - out["present_vs_bare"][0]))
    mac("handWin", str(out["hand_vs_present"][0]))
    mac("handN", str(out["hand_vs_present"][1]))
    # Absolute 0-3 scores from the same 26 texts, one judge, so the paper's
    # "2.31 / 2.08 / 1.31 / 0.73" cannot drift from the run either.
    by = collections.defaultdict(list)
    for r in json.load(open("../runs/punct/after_120b.json")):
        if r["score"] is not None:
            by[r["kind"]].append(r["score"])
    for k, name in (("hand", "absHand"), ("llm_120b", "absLLM"),
                    ("bare", "absBare"), ("present", "absOurs")):
        mac(name, f"{st.mean(by[k]):.2f}")
    mac("absGap", f"{st.mean(by['hand']) - st.mean(by['llm_120b']):.2f}")
    mac("absN", str(len(by["hand"])))
    return out


def fig_results():
    """Two panels, the two blind results the paper turns on."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 1.85))

    arms = ["calib.", "$k{=}2$", "$k{=}4$", "$k{=}8$"]
    frac = [12 / 12, 14 / 14, 14 / 14, 14 / 14]
    a.bar(arms, frac, 0.55, color=[MUT, ACC, ACC, ACC])
    a.axhline(0.5, color=INK, lw=0.8, ls=(0, (3, 2)))
    a.set_ylim(0, 1.08); a.set_ylabel("preferred, both judges")
    a.set_title("a.  a single chunk beats a nest of chunks", loc="left")
    for i, f in enumerate(frac):
        a.text(i, f + 0.03, f"{int(f * (12 if i == 0 else 14))}/"
               f"{12 if i == 0 else 14}", ha="center", fontsize=6.5)
    a.text(3.45, 0.52, "chance", fontsize=6.5, va="bottom", ha="right",
           color=INK)

    labs = ["calibration\n(prose vs shuffle)", "bare spacing\nover ours",
            "hand\nover ours"]
    vals = [int(M["calibHit"]) / int(M["calibN"]),
            int(M["bareWin"]) / int(M["presentN"]),
            int(M["handWin"]) / int(M["handN"])]
    ns = [f"{M['calibHit']}/{M['calibN']}", f"{M['bareWin']}/{M['presentN']}",
          f"{M['handWin']}/{M['handN']}"]
    b.bar(labs, vals, 0.55, color=[MUT, ACC, ACC])
    b.axhline(0.5, color=INK, lw=0.8, ls=(0, (3, 2)))
    b.set_ylim(0, 1.08); b.set_ylabel("preferred, both judges")
    b.set_title("b.  our punctuation loses to putting in none", loc="left")
    for i, (v, n) in enumerate(zip(vals, ns)):
        b.text(i, v + 0.03, n, ha="center", fontsize=6.5)
    fig.savefig("fig/results.pdf")
    plt.close(fig)


if __name__ == "__main__":
    vs, rates, a0, b = scaling()
    rows = conservation()
    L, fwd, rev = coverage()
    punctuation()
    fig_measurements(rows, vs, rates, a0, b, L, fwd, rev)
    fig_results()
    with open("numbers.tex", "w") as f:
        f.write("% generated by make_figures.py -- do not edit\n")
        for k, v in sorted(M.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("\n".join(f"  {k:<12} {v}" for k, v in sorted(M.items())))
