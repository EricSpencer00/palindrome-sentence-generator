"""Plot the archived continuous diagnostics and fragment concentration, offline.

Requires matplotlib and numpy only for rendering. No inference or search runs.
All quantities are descriptive; shuffle ranges are not confidence intervals.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.audit_diagnostic_overlap import RUN, audit, require

PHRASE = "one poem association"
COLORS = {True: "#BD6338", False: "#246580"}


def collect(root: Path = ROOT) -> dict:
    audit(root)  # Exactness, inventories, and score/text provenance joins.
    source = json.loads((root / RUN / "static-verifier-input-aws.json").read_bytes())
    brown = {x["id"]: x for x in json.loads((root / RUN / "brown-v4-aws.json").read_bytes())["scores"]}
    gpt = {x["id"]: x for x in json.loads((root / RUN / "gpt2-v4-aws.json").read_bytes())["scores"]}
    controls = defaultdict(list)
    for x in source["items"]:
        if x["kind"] != "candidate":
            controls[x["pair_id"]].append(gpt[x["id"]]["mean_token_logprob"])
    records = []
    for x in source["items"]:
        if x["kind"] != "candidate":
            continue
        delta = [gpt[x["id"]]["mean_token_logprob"] - s for s in controls[x["pair_id"]]]
        words = re.findall(r"[a-z]+", x["text"].lower())
        records.append({
            "id": x["id"], "text": x["text"], "text_sha256": x["text_sha256"],
            "method": x["source"]["method"],
            "has_phrase": any(words[i:i + 3] == PHRASE.split() for i in range(len(words) - 2)),
            "brown_gain": brown[x["id"]]["brown_order_gain_vs_own_shuffle"],
            "gpt2_gain": mean(delta), "gpt2_candidate_minus_each_shuffle": delta,
        })
    by_text = defaultdict(list)
    for row in records:
        by_text[row["text_sha256"]].append(row)
    distinct = []
    for sha, copies in sorted(by_text.items()):
        distinct.append({
            "text_sha256": sha, "text": copies[0]["text"],
            "has_phrase": copies[0]["has_phrase"], "record_count": len(copies),
            "brown_gain": mean(x["brown_gain"] for x in copies),
            "gpt2_gain": mean(x["gpt2_gain"] for x in copies),
        })
    trigrams = Counter()
    for row in distinct:
        words = re.findall(r"[a-z]+", row["text"].lower())
        trigrams.update(set(" ".join(words[i:i + 3]) for i in range(len(words) - 2)))
    frequent = [list(pair) for pair in sorted(trigrams.items(), key=lambda x: (-x[1], x[0]))[:6]]
    extremes = {
        "A": max(distinct, key=lambda x: x["brown_gain"]),
        "B": max(distinct, key=lambda x: x["gpt2_gain"]),
        "C": min(distinct, key=lambda x: x["gpt2_gain"]),
    }
    groups = {}
    for present in (True, False):
        selected = [x for x in distinct if x["has_phrase"] == present]
        groups["phrase_present" if present else "phrase_absent"] = {
            "distinct_texts": len(selected),
            "mean_brown_gain": mean(x["brown_gain"] for x in selected),
            "mean_gpt2_gain": mean(x["gpt2_gain"] for x in selected),
        }
    stats = {
        "record_count": len(records), "distinct_text_count": len(distinct),
        "opposite_sign_distinct_texts": sum((x["brown_gain"] > 0) != (x["gpt2_gain"] > 0) for x in distinct),
        "shuffle_ranges_crossing_zero": sum(min(x["gpt2_candidate_minus_each_shuffle"]) < 0 < max(x["gpt2_candidate_minus_each_shuffle"]) for x in records),
        "candidate_above_all_32_shuffles": sum(min(x["gpt2_candidate_minus_each_shuffle"]) > 0 for x in records),
        "candidate_below_all_32_shuffles": sum(max(x["gpt2_candidate_minus_each_shuffle"]) < 0 for x in records),
        "frequent_trigrams": frequent, "phrase_groups": groups,
    }
    payload = {"records": records, "distinct_texts": distinct, "statistics": stats, "extremes": extremes}
    payload["numerical_payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload["definitions"] = {
        "scatter": "One point per exact surface text; each contrast is averaged across its saved records, with equal weight per record.",
        "ranges": "One row per selected record, showing min/max and interquartile range of candidate-minus-each-shuffle GPT-2 scores, plus their mean. There are 32 shuffles per row. These are permutation spreads, not confidence intervals.",
        "fragments": "Number of distinct texts containing each lowercase ASCII-word trigram; one vote per text. Six most frequent, ties alphabetically. Trigrams may overlap.",
        "groups": "Post-hoc grouping by the previously noted phrase one poem association. Equal weight per distinct text after averaging duplicate records; descriptive, without causal attribution.",
        "extremes": "A: maximum mean Brown contrast; B/C: maximum/minimum mean GPT-2 contrast. Selected to illustrate diagnostic extremes, not typical outputs.",
    }
    return payload


def render(data: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#777777", "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    out.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.canvas.draw()
        # Check all visible labels against the canvas before exporting.
        renderer = fig.canvas.get_renderer()
        for artist in fig.findobj(matplotlib.text.Text):
            if artist.get_visible() and artist.get_text():
                box = artist.get_window_extent(renderer)
                require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1,
                        f"clipped figure text: {artist.get_text()}")
        fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(out / f"{name}.png", dpi=230)
        plt.close(fig)

    distinct = data["distinct_texts"]
    records = sorted(data["records"], key=lambda x: (x["gpt2_gain"], x["id"]))
    fig, (ax, rng) = plt.subplots(1, 2, figsize=(7.05, 3.05), gridspec_kw={"width_ratios": [1, 1.04]})
    fig.subplots_adjust(left=.09, right=.985, bottom=.18, top=.83, wspace=.34)
    ax.set_title("a  Two scorers, different preferences", loc="left", pad=11)
    ax.axhline(0, color="#999999", linewidth=.7, linestyle="--", zorder=0)
    ax.axvline(0, color="#999999", linewidth=.7, linestyle="--", zorder=0)
    for present in (False, True):
        rows = [x for x in distinct if x["has_phrase"] == present]
        ax.scatter([x["brown_gain"] for x in rows], [x["gpt2_gain"] for x in rows],
                   c=COLORS[present], s=23, edgecolors="white", linewidths=.4, alpha=.92, zorder=3)
    for label, row in data["extremes"].items():
        xy = (row["brown_gain"], row["gpt2_gain"])
        ax.annotate(label, xy, xytext=(5, 6), textcoords="offset points", fontsize=8, weight="bold")
        ax.scatter(*xy, facecolor="none", edgecolor="#222222", s=43, linewidth=.7, zorder=4)
    ax.set(xlim=(-.58, .96), ylim=(-.69, .75), xticks=[-.5, 0, .5], yticks=[-.5, 0, .5], xlabel="Brown gain (nats / word transition)",
           ylabel="GPT-2 gain (log prob. / predicted token)")
    ax.text(.97, .025, "28 / 52 have opposite signs", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5)
    rng.set_title("b  The mean hides the shuffle spread", loc="left", pad=11)
    rng.axvline(0, color="#666666", linewidth=.8, linestyle="--", zorder=0)
    for i, row in enumerate(records, 1):
        vals = row["gpt2_candidate_minus_each_shuffle"]
        q1, q3 = np.quantile(vals, [.25, .75])
        color = COLORS[row["has_phrase"]]
        rng.plot([min(vals), max(vals)], [i, i], color=color, alpha=.24, linewidth=.65)
        rng.plot([q1, q3], [i, i], color=color, alpha=.65, linewidth=1.45)
        rng.plot(row["gpt2_gain"], i, "o", color=color, markersize=2.1)
    rng.set(xlim=(-1.15, 1.3), ylim=(-3, 79), xticks=[-1, -.5, 0, .5, 1], yticks=[1, 22, 44, 66],
            xlabel="GPT-2: candidate minus one shuffle", ylabel="Record rank by mean contrast")
    rng.text(.03, .96, "63 / 66 ranges cross zero", transform=rng.transAxes, va="top", fontsize=7.5)
    rng.text(.02, -.26, "thin: min-max   thick: middle 50%   dot: mean", transform=rng.transAxes, fontsize=7)
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[k], markersize=4,
                       label=label) for k, label in ((True, 'Contains “one poem association”'), (False, "Other texts"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.015), ncol=2, frameon=False, fontsize=7.5)
    save(fig, "diagnostic-landscape")

    fig = plt.figure(figsize=(7.05, 2.22))
    grid = fig.add_gridspec(2, 2, left=.20, right=.985, bottom=.22, top=.86, width_ratios=[.88, 1.12], wspace=.64, hspace=.7)
    bars = fig.add_subplot(grid[:, 0])
    top = data["statistics"]["frequent_trigrams"]
    labels, counts = zip(*top)
    bars.barh(range(6), counts, color=[COLORS[True] if t == PHRASE else "#7C8991" for t in labels], height=.55)
    bars.set_yticks(range(6), labels)
    bars.invert_yaxis()
    bars.set(xlim=(0, 55), xticks=[0, 13, 26, 39, 52], xlabel="Distinct texts containing the trigram")
    bars.tick_params(axis="y", length=0)
    bars.set_title("a  Shared fragments across distinct texts", loc="left", pad=12)
    for i, n in enumerate(counts):
        bars.text(n + .7, i, str(n), va="center", fontsize=7)
    for i, (key, label, unit, limits) in enumerate([
        ("brown_gain", "b  Brown", "nats / word transition", (-.58, .96)),
        ("gpt2_gain", "c  GPT-2", "log prob. / predicted token", (-.69, .75)),
    ]):
        axis = fig.add_subplot(grid[i, 1])
        axis.axvline(0, color="#BBBBBB", linewidth=.65, linestyle="--")
        for present, y in ((True, 1), (False, 0)):
            rows = [r for r in distinct if r["has_phrase"] == present]
            for row in rows:
                jitter = (int(row["text_sha256"][:8], 16) / 0xffffffff - .5) * .36
                axis.plot(row[key], y + jitter, "o", color=COLORS[present], alpha=.65, markersize=2.5)
            axis.plot(mean(r[key] for r in rows), y, marker="D", color="#111111", markersize=3.8, zorder=5)
        axis.set(yticks=[0, 1], yticklabels=["Absent (28)", "Present (24)"], ylim=(-.5, 1.5), xlim=limits, xticks=[-.5, 0, .5])
        axis.tick_params(axis="y", length=0, labelsize=6.8)
        axis.set_title(f"{label} gain ({unit})", loc="left", fontsize=8, pad=3)
    fig.text(.635, .035, "Phrase groups; black diamonds mark means", fontsize=7, ha="center")
    save(fig, "fragment-concentration")


def main():
    data = collect()
    out = ROOT / "paper/fig"
    render(data, out)
    (out / "diagnostic-data.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(data["statistics"], indent=2))


if __name__ == "__main__":
    main()
