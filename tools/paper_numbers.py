"""Numbers and figure data for paper/v5 from the paper metrics JSON.

usage: python tools/paper_numbers.py [runs/paper-metrics-20261125.json]

Writes paper/v5/numbers.tex (one \\newcommand per number the text uses) and
paper/v5/figdata/*.tex (pgfplots snippets). Rerun after the metrics change;
the paper picks everything up on the next build. Reads JSON only.
"""
import glob
import json
import os
import random
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "paper-metrics-20261125.json")
OUT = os.path.join(ROOT, "paper", "v5")
FIG = os.path.join(OUT, "figdata")

CORPORA = {"v1": "vOne", "v5_live": "vFiveLive", "v5_sophia": "vFiveSophia",
           "brown": "Brown", "classics": "Classics"}
LABEL = {"v1": "v1", "v5_live": "v5 live", "v5_sophia": "v5 long",
         "brown": "Brown", "classics": "Classics"}
# Held-out AUC printed by training/fit_spectrum.py for the READABLE weights in
# llm_palindrome/spectrum.py (recorded with them in commit 93560029).
VERIFIER_AUC = 0.874
# Warm single-thread time of Spectrum.readable(features(s)) over the 658
# sentences of the v5 live corpus on a laptop CPU, measured 2026-11-25.
VERIFIER_MS = 0.1

macros: dict[str, str] = {}


def put(name: str, value) -> None:
    macros[name] = str(value)


def num(x: float, dp: int) -> str:
    s = f"{x:.{dp}f}"
    return f"\\ensuremath{{-{s[1:]}}}" if s.startswith("-") else s


def quantile(v: list[float], q: float) -> float:
    v = sorted(v)
    i = q * (len(v) - 1)
    lo = int(i)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (i - lo)


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def exact(text: str) -> bool:
    t = re.sub("[^a-z]", "", text.lower())
    return bool(t) and t == t[::-1]


def boxplot(name: str, rows: list[float], pos: int, colour: str, log: bool) -> str:
    q = [quantile(rows, p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    fmt = (lambda v: f"{v:.4g}") if log else (lambda v: f"{v:.4f}")
    rng = random.Random(pos)
    pts = " ".join(f"({pos + rng.uniform(-0.18, 0.18):.3f},{fmt(v)})" for v in rows)
    return (f"% {name}: n={len(rows)}\n"
            f"\\addplot+[boxplot prepared={{draw position={pos}, lower whisker={fmt(q[0])}, "
            f"lower quartile={fmt(q[1])}, median={fmt(q[2])}, upper quartile={fmt(q[3])}, "
            f"upper whisker={fmt(q[4])}, box extend=0.6}}, draw={colour}, solid, fill={colour}!12, mark=none] "
            f"coordinates {{}};\n"
            f"\\addplot[only marks, mark=*, mark size=0.7pt, {colour}, opacity=0.45] coordinates {{{pts}}};\n")


def themes() -> None:
    """Theme-weight sweep: no theme vs. the deployed themes at weights 5, 8, 15.
    Every themed arm draws from the same kept themes (server/v5.py THEMES)."""
    tdir = os.path.join(ROOT, "runs", "themes")
    m = json.load(open(os.path.join(tdir, "metrics-20261125.json")))
    w = json.load(open(os.path.join(tdir, "weights-20261125.json")))
    src = open(os.path.join(ROOT, "server", "v5.py")).read()
    kept = {t.replace(" ", ",") for t in re.findall(r'"([a-z ]+)"', src.split("THEMES = [")[1].split("]")[0])}
    trial = [json.loads(line) for line in open(os.path.join(tdir, "themed.jsonl"))]
    w15 = [r for r, o in zip(m["themed"]["rows"], trial) if o["theme"] in kept]
    put("themeKept", len(kept))
    put("themeTried", len({o["theme"] for o in trial}))

    def tri(t):
        x = re.findall("[a-z]+", t.lower())
        return set(zip(x, x[1:], x[2:]))

    for p, rows in (("Zero", m["unthemed"]["rows"]), ("Five", w["w5"]["rows"]),
                    ("Eight", w["w8"]["rows"]), ("Fifteen", w15)):
        T = [tri(r["text"]) for r in rows]
        pairs = [(a, b) for i, a in enumerate(T) for b in T[i + 1:]]
        seen: dict = {}
        for s in T:
            for g in s:
                seen[g] = seen.get(g, 0) + 1
        g, n = max(seen.items(), key=lambda kv: (kv[1], kv[0]))
        put(f"themeN{p}", len(rows))
        put(f"themeExact{p}", sum(exact(r["text"]) for r in rows))
        put(f"themeOverlap{p}", num(st.mean(len(a & b) / len(a | b) for a, b in pairs), 2))
        put(f"themeTop{p}", num(100 * n / len(rows), 0))
        put(f"themeTopPhrase{p}", " ".join(g))
        put(f"themePPL{p}", num(st.median(r["gpt2_ppl"] for r in rows), 0))
        put(f"themeRead{p}", num(100 * st.mean(r["readable_share"] for r in rows), 0))
        put(f"themeShort{p}", num(100 * st.mean(r["short_share"] for r in rows), 0))


def main() -> None:
    d = json.load(open(SRC))
    os.makedirs(FIG, exist_ok=True)

    for key, p in CORPORA.items():
        if key not in d:
            continue
        rows, s = d[key]["rows"], d[key]["summary"]
        L = [r["letters"] for r in rows]
        put(p + "N", len(rows))
        put(p + "Distinct", len({r["text"] for r in rows}))
        put(p + "Letters", num(st.mean(L), 0))
        put(p + "LettersMin", min(L))
        put(p + "LettersMed", num(st.median(L), 0))
        put(p + "LettersMax", max(L))
        put(p + "Words", num(st.mean(r["words"] for r in rows), 0))
        put(p + "Exact", num(100 * s["exact"], 0))
        put(p + "ExactCheck", sum(exact(r["text"]) for r in rows))
        put(p + "Short", num(100 * s["short_share"], 1))
        put(p + "WordLen", num(s["mean_word_len"], 2))
        put(p + "TTR", num(s["type_token"], 2))
        put(p + "Bigrams", num(100 * s["distinct_bigrams"], 1))
        put(p + "SentWords", num(s["sentence_words"], 1))
        put(p + "LM", num(s["lm_per_token"], 2))
        put(p + "Readable", num(s["readable"], 2))
        put(p + "ReadableShare", num(100 * s["readable_share"], 0))
        put(p + "GarbageShare", num(100 * s["garbage_share"], 1))
        put(p + "Ism", num(s["ism"], 2))
        ppl = [r["gpt2_ppl"] for r in rows if "gpt2_ppl" in r]
        if ppl:
            put(p + "PPL", num(st.median(ppl), 0))
            put(p + "PPLQone", num(quantile(ppl, 0.25), 0))
            put(p + "PPLQthree", num(quantile(ppl, 0.75), 0))
            put(p + "PPLMin", num(min(ppl), 0))
            put(p + "PPLMax", num(max(ppl), 0))

    med = {k: st.median(r["gpt2_ppl"] for r in d[k]["rows"]) for k in ("v1", "v5_live", "v5_sophia", "brown")}
    put("pplRatioVOne", num(med["v1"] / med["v5_live"], 1))
    put("pplRatioVOneLong", num(med["v1"] / med["v5_sophia"], 1))
    put("pplRatioBrownLive", num(med["v5_live"] / med["brown"], 1))
    put("pplRatioBrownLong", num(med["v5_sophia"] / med["brown"], 1))
    brown_max = max(r["gpt2_ppl"] for r in d["brown"]["rows"])
    v1_min = min(r["gpt2_ppl"] for r in d["v1"]["rows"])
    for k, p in (("v5_live", "Live"), ("v5_sophia", "Long")):
        rows = d[k]["rows"]
        put(f"inBrownRange{p}", sum(r["gpt2_ppl"] <= brown_max for r in rows))
        put(f"belowVOneMin{p}", sum(r["gpt2_ppl"] < v1_min for r in rows))
        put(f"spearman{p}", num(spearman([r["letters"] for r in rows], [r["gpt2_ppl"] for r in rows]), 2))
        put(f"spearmanRead{p}", num(spearman([r["letters"] for r in rows], [r["readable_share"] for r in rows]), 2))
    delta = 100 * (d["v5_live"]["summary"]["short_share"] - d["v1"]["summary"]["short_share"])
    put("shortDeltaAbs", num(abs(delta), 1))
    put("shortDir", "above" if delta >= 0 else "below")
    deltaL = 100 * (d["v5_sophia"]["summary"]["short_share"] - d["v1"]["summary"]["short_share"])
    put("shortDeltaLongAbs", num(abs(deltaL), 1))
    put("shortDirLong", "above" if deltaL >= 0 else "below")

    # Long-run batches: settings and wall time live in the run records.
    batches = sorted(glob.glob(os.path.join(ROOT, "runs", "paragraph-sophia-20261125", "*.jsonl")))
    recs = [json.loads(line) for f in batches for line in open(f) if line.strip()]
    secs = [r["secs"] for r in recs if "secs" in r]
    put("longRecords", len(recs))
    put("longBatches", len(batches))
    put("longThemes", len({tuple(r["theme"]) for r in recs}))
    put("longSecsMin", min(secs))
    put("longSecsMed", num(st.median(secs), 0))
    put("longSecsMax", max(secs))
    put("longBeamMin", min(r["beam"] for r in recs))
    put("longBeamMax", max(r["beam"] for r in recs))
    put("longTargetMin", min(r["target"] for r in recs))
    put("longTargetMax", max(r["target"] for r in recs))

    # Verifier anchors, counted the way training/fit_spectrum.py builds them.
    posts = json.load(open(os.path.join(ROOT, "data", "reddit-palindromes", "2026-10-09", "posts.json")))["rows"]
    reddit = [r["text"] for r in posts if r["text"].isascii()
              and len(t := re.sub("[^a-z]", "", r["text"].lower())) >= 6 and t == t[::-1]]
    v1w = v1n = 0
    for path in glob.glob(os.path.join(ROOT, "data", "spectrum-anchors", "v1-live-*.jsonl")):
        for line in open(path):
            o = json.loads(line)
            w = o["left"] + ([o["center"]] if o["center"] else []) + o["right"]
            v1w += len(range(0, len(w) - 10, 7))
            v1n += 1
    classics = d["classics"]["summary"]["n"]
    put("anchorReddit", len(reddit))
    put("anchorBrown", 300)
    put("anchorClassics", classics)
    put("anchorVOneWindows", v1w)
    put("anchorVOneOutputs", v1n)
    put("anchorShuffled", len(reddit))
    put("anchorPos", len(reddit) + 300 + classics)
    put("anchorNeg", v1w + len(reddit))
    put("verifierAUC", f"{VERIFIER_AUC:.3f}")
    put("verifierMs", VERIFIER_MS)

    # Ruts: the word 4-gram found in the most paragraphs of a corpus.
    for k, p in (("v5_live", "Live"), ("v5_sophia", "Long"), ("v1", "VOne")):
        seen: dict = {}
        for r in d[k]["rows"]:
            w = re.findall("[a-z]+", r["text"].lower())
            for g in {tuple(w[i:i + 4]) for i in range(len(w) - 3)}:
                seen[g] = seen.get(g, 0) + 1
        g, n = max(seen.items(), key=lambda kv: (kv[1], kv[0]))
        put(f"rut{p}", " ".join(g))
        put(f"rut{p}N", n)

    themes()

    with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
        fh.write(f"% Generated by tools/paper_numbers.py from {os.path.relpath(SRC, ROOT)}. Do not edit.\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # Colour names are defined in main.tex.
    colours = {"v1": "cvOne", "v5_live": "cLive", "v5_sophia": "cLong", "brown": "cBrown", "classics": "cClassic"}
    order = ["v1", "v5_live", "v5_sophia", "brown"]
    with open(os.path.join(FIG, "ppl.tex"), "w") as fh:
        for i, k in enumerate(order, 1):
            fh.write(boxplot(k, [r["gpt2_ppl"] for r in d[k]["rows"]], i, colours[k], True))
    order5 = order + ["classics"]
    for metric, fname in (("readable_share", "readshare"), ("short_share", "short"), ("letters", "letters")):
        with open(os.path.join(FIG, fname + ".tex"), "w") as fh:
            for i, k in enumerate(order5, 1):
                fh.write(boxplot(k, [r[metric] for r in d[k]["rows"]], i, colours[k], False))
    with open(os.path.join(FIG, "length.tex"), "w") as fh:
        for k in ("v5_live", "v5_sophia", "v1", "brown"):
            pts = " ".join(f"({r['letters']},{r['gpt2_ppl']:.4g})" for r in d[k]["rows"])
            mark = {"v1": "triangle*", "brown": "square*"}.get(k, "*")
            fh.write(f"\\addplot[only marks, mark={mark}, mark size=1.1pt, {colours[k]}, opacity=0.6] "
                     f"coordinates {{{pts}}};\n\\addlegendentry{{{LABEL[k]}}}\n")
    print(f"{len(macros)} macros -> {os.path.relpath(OUT, ROOT)}/numbers.tex")


if __name__ == "__main__":
    main()
