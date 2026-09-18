"""Vector figures for the current paper, using only its retained evidence.

Requires matplotlib. This does not run the archived make_figures.py.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'fig'
QA = ROOT / 'paper' / 'out' / 'figures-qa'
WIDTH = 16 / 2.54
FULL_WIDTH = 18.2 / 2.54
TERMINAL, PLANNED = '#787878', '#185C80'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#555555', 'axes.linewidth': .6,
    'grid.color': '#dedede', 'grid.linewidth': .5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})


def finish(fig, name, labels):
    """Check text rectangles and export without cropping the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for ax in fig.axes:
        texts.extend(ax.get_xticklabels() + ax.get_yticklabels())
        texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
    texts.extend(labels)
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts
             if t.get_visible() and t.get_text()]
    collisions, clipping = [], []
    canvas = fig.bbox
    for label, a in boxes:
        if a.x0 < canvas.x0 or a.y0 < canvas.y0 or a.x1 > canvas.x1 or a.y1 > canvas.y1:
            clipping.append(label)
    for i, (label_a, a) in enumerate(boxes):
        for label_b, b in boxes[i + 1:]:
            if min(a.x1, b.x1) - max(a.x0, b.x0) > .5 and min(a.y1, b.y1) - max(a.y0, b.y0) > .5:
                collisions.append([label_a, label_b])
    report = {'text_elements': len(boxes), 'overlaps': collisions, 'clipped': clipping,
              'size_inches': list(fig.get_size_inches())}
    (QA / f'{name}-qa.json').write_text(json.dumps(report, indent=2) + '\n')
    assert not collisions and not clipping, report
    fig.savefig(OUT / f'{name}.pdf', metadata={'Title': name, 'CreationDate': None, 'ModDate': None})
    fig.savefig(QA / f'{name}.png', dpi=220)
    plt.close(fig)
    return report


def main():
    OUT.mkdir(exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'runs/polaris/sentence_plan_20260904_204815/aggregate.json'
    aggregate = json.loads(source.read_text())
    counts = {}
    for arm in ('terminal', 'planned'):
        keys = {(p['left'], p['right']) for p in aggregate[arm]['pairs']}
        assert len(keys) == aggregate[arm]['hits']
        counts[arm] = [len(keys),
            len({(l + r).replace(' ', '') for l, r in keys}),
            len({(l.split()[-1], r.split()[0]) for l, r in keys})]
    assert counts == {'terminal': [20989, 19479, 20], 'planned': [86511, 82056, 30]}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.20))
    fig.subplots_adjust(left=.074, right=.983, bottom=.23, top=.79, wspace=.58)
    labels = []
    titles = ['(a) Accepted token pairs', '(b) Distinct letter strings', '(c) Junction families']
    for i, ax in enumerate(axes):
        values = [counts[a][i] for a in ('terminal', 'planned')]
        ax.bar([0, 1.25], values, width=.52, color=[TERMINAL, PLANNED], zorder=3)
        ax.set_xticks([0, 1.25], ['Terminal', 'Incremental'])
        ax.set_xlim(-.62, 1.87)
        ax.set_title(titles[i], pad=12)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.tick_params(axis='both', length=3, width=.6)
        if i < 2:
            ax.set_ylim(0, 110000)
            ax.set_yticks([0, 25000, 50000, 75000, 100000])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if not x else f'{x/1000:g}k'))
        else:
            ax.set_ylim(0, 36)
            ax.set_yticks([0, 10, 20, 30])
        for x, y in enumerate(values):
            labels.append(ax.annotate(f'{y:,}', (x * 1.25, y), xytext=(0, 5),
                          textcoords='offset points', ha='center', va='bottom', fontsize=8))
    reports = {'candidate-yield': finish(fig, 'candidate-yield', labels)}

    mirror_source = ROOT / 'runs/mirror-cost-2026-09-11/results.json'
    mirror = json.loads(mirror_source.read_text())
    rows = mirror['rows']
    model_names = [row['name'] for row in mirror['models']]
    display_names = {'gpt2': 'GPT-2',
                     'HuggingFaceTB/SmolLM2-135M': 'SmolLM2-135M'}
    strategies = mirror['design']['segmentation_strategies']
    colours = {'unigram': '#176B87', 'fewest': '#A65E2E', 'greedy': '#555555'}
    markers = {'unigram': 'o', 'fewest': 's', 'greedy': '^'}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.34), sharey=True)
    fig.subplots_adjust(left=.075, right=.982, bottom=.22, top=.73, wspace=.13)
    for ax, model in zip(axes, model_names):
        for strategy in strategies:
            selected = sorted((row for row in rows
                               if row['model'] == model and row['strategy'] == strategy),
                              key=lambda row: row['n_letters'])
            x = [row['n_letters'] for row in selected]
            y = [row['mirror_cost'] for row in selected]
            ci = [1.96 * row['mirror_cost_se'] for row in selected]
            ax.errorbar(x, y, yerr=ci, color=colours[strategy],
                        marker=markers[strategy], markersize=3.5,
                        linewidth=1.15, elinewidth=.65, capsize=2,
                        label=strategy.capitalize())
        ax.set_title(display_names[model], pad=6)
        ax.set_xlabel('Target span length (letters)')
        ax.set_xticks(mirror['design']['target_lengths'])
        ax.set_xlim(15, 125)
        ax.set_ylim(2.0, 3.55)
        ax.set_yticks([2.0, 2.5, 3.0, 3.5])
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.tick_params(axis='both', length=3, width=.6)
    axes[0].set_ylabel('Additional bits per letter')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=3,
               frameon=False, bbox_to_anchor=(.5, .985), handlelength=2.0)
    reports['mirror-cost'] = finish(fig, 'mirror-cost', [])
    specs = [
        ('Static | 45 s | cap 3', 'norvig-letter-static'),
        ('Unused | 45 s | cap 3', 'norvig-letter-dynamic'),
        ('Unused | 120 s | cap 3', 'norvig-letter-dynamic-long'),
        ('Feasible | 120 s | cap 3', 'norvig-letter-feasible'),
        ('Unused | 120 s | no cap', 'norvig-letter-comparable'),
        ('Feasible | 300 s | cap 3', 'norvig-letter-feasible-300'),
    ]
    paths = [ROOT / 'runs' / folder / 'result.json' for _, folder in specs]
    values = [json.loads(p.read_text())['letters'] for p in paths]
    assert values == [68286, 76979, 88101, 88095, 88455, 90937]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.68))
    fig.subplots_adjust(left=.305, right=.965, top=.92, bottom=.22)
    bars = ax.barh(range(len(values)), values, height=.55,
                   color=[TERMINAL]*5 + [PLANNED], zorder=3)
    bars[-1].set_hatch('//')
    bars[-1].set_edgecolor('white')
    ax.set_yticks(range(len(values)), [s for s, _ in specs])
    ax.invert_yaxis()
    ax.set_xlim(0, 110000)
    ax.set_xticks([0, 25000, 50000, 75000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if not x else f'{x/1000:g}k'))
    ax.set_xlabel('Letters in the best saved output', labelpad=7)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=9)
    labels = [ax.annotate(f'{value:,}', (value, i), xytext=(5, 0),
                         textcoords='offset points', va='center', fontsize=8,
                         bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .5})
              for i, value in enumerate(values)]
    reports['inventory-runs'] = finish(fig, 'inventory-runs', labels)
    (QA / 'figure-manifest.json').write_text(json.dumps({
        'sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [source, mirror_source] + paths},
        'checks': reports}, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
