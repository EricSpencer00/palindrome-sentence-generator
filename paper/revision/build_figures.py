"""Regenerate every manuscript plot from frozen records; no search or inference.

Run python3 revision/build_figures.py from paper/, or from any working directory.
The paired bootstrap is inherited from the frozen October 2 analysis.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault('MPLCONFIGDIR', '/tmp/palindrome-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_records import analyze

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
EVIDENCE = HERE / 'evidence'
FIG = PAPER / 'fig'
PENALTIES = (0, 4, 16)
COLORS = ('#246580', '#BD6338', '#707B65')


def normalize(text):
    return ''.join(c.lower() for c in text if c.isascii() and c.isalpha())


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, artist.get_text()
            assert box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, artist.get_text()
    fig.savefig(FIG / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def verify_examples(rows, scores):
    illustrative = [
        ('Never odd or even.', 'existing manuscript illustration', 14),
        ('No rider sees red iron.', 'illustrative calibration; novelty unverified', 18),
        ('Liam sees mail.', 'established illustrative example', 12),
        ('No evil did live on.', 'established illustrative example', 15),
    ]
    examples = []
    for text, origin, letters in illustrative:
        tape = normalize(text)
        assert len(tape) == letters and tape == tape[::-1]
        examples.append(dict(text=text, normalized=tape, letters=letters,
                             exact=True, provenance=origin, experimental=False))
    assert normalize('Liam') == normalize('mail')[::-1]
    assert normalize('sees') == normalize('sees')[::-1]
    for job in ('s7100-b1-center-p0', 's7100-b1-center-p4'):
        row = next(r for r in rows if r['id'] == job)
        tape = normalize(row['text'])
        assert tape == tape[::-1] and len(tape) == row['letters'] == 48
        assert row['strict_admitted_selected_output'] is True
        examples.append(dict(text=row['text'], normalized=tape, letters=len(tape),
                             exact=True, experimental=True, job_id=job,
                             text_sha256=row['text_sha256'], strict=True,
                             model_scores=scores[row['text_sha256']]))
    return examples


def build():
    FIG.mkdir(exist_ok=True)
    for name, source in read('SOURCE-HASHES.json').items():
        assert hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest() == source['sha256'], name
    data = analyze()
    rows = read('results.json')['results']
    qm, qr = read('quality-manifest.json'), read('quality-results.json')
    answers = {r['id']: r for r in qr['responses']}
    scores = {}
    controls = Counter()
    for p in qm['presentations']:
        answer = answers[p['id']]
        assert answer['text_sha256'] == p['text_sha256']
        raw = ''.join(c.get('text', '') for c in answer['response']['output']['message']['content'])
        # Independently parse each saved judge reply rather than trusting summary counts.
        parsed = json.loads(raw[raw.index('{'):raw.rindex('}')+1])
        assert parsed == answer['scores']
        if p['condition'] == 'candidate':
            scores[p['text_sha256']] = parsed
        elif p['condition'] != 'hidden_repeat':
            controls[p['condition']] += int(min(parsed.values()) >= 2)
    assert len(scores) == 148
    joint = np.zeros((4,4), dtype=int)
    for s in scores.values():
        joint[s['grammaticality'],s['coherent_meaning']] += 1
    assert joint.sum() == 148 and joint[2:,2:].sum() == 0
    examples = verify_examples(rows, scores)
    manuscript = (PAPER/'naacl2027.tex').read_text().replace('\\newline ', ' ')
    for e in examples:
        assert e['text'] in manuscript, e['text']
        if not e['experimental']:
            assert e['normalized'] in manuscript, e['normalized']
    displays = re.findall(r'\\begin\{quote\}\\small\\ttfamily\s*(.*?)\\end\{quote\}', manuscript, re.S)
    assert len(displays) == 2
    for display, example in zip(displays, examples[-2:]):
        assert normalize(display) == example['normalized']
    old_texts = read('diagnostic-data.json')['distinct_texts']
    old_counts = Counter()
    for old in old_texts:
        assert hashlib.sha256(old['text'].encode()).hexdigest() == old['text_sha256']
        words = re.findall(r'[a-z]+', old['text'].lower())
        old_counts.update(set(tuple(words[i:i+3]) for i in range(len(words)-2)))
    assert len(old_texts) == 52 and old_counts[('no','it','a')] == 46
    targets = [list(t) for t,n in sorted(old_counts.items(), key=lambda x:(-x[1],x[0]))[:6]]
    assert targets == read('manifest.json')['fragments']
    paired = {(r['seed'],r['method'],r['band_id'],r['penalty']):r for r in rows}
    differences = sum(r['text'] != paired[(r['seed'],r['method'],r['band_id'],16)]['text']
                      for r in rows if r['penalty']==4)
    assert differences == 8
    probe = read('probe-results-v3.json')
    pm = read('probe-manifest-v4.json')
    assert probe['manifest_sha256'] == hashlib.sha256((EVIDENCE/'probe-manifest-v4.json').read_bytes()).hexdigest()
    assert len(probe['results']) == len(pm['jobs']) == 32
    selected = [r for r in probe['results'] if r['text']]
    for r in selected:
        tape = normalize(r['text'])
        assert tape == tape[::-1] and len(tape) == r['letters']
        assert hashlib.sha256(r['text'].encode()).hexdigest() == r['text_sha256']
    probe_summary = dict(requested=32, exact=len(selected),
                         strict=sum(r['strict_admitted_selected_output'] for r in selected),
                         max_letters=max(r['letters'] for r in selected))
    assert probe_summary == dict(requested=32, exact=16, strict=0, max_letters=824)
    arms = [data['arms'][p] for p in PENALTIES]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'axes.titlesize':9,'axes.labelsize':8,
                         'xtick.labelsize':7,'ytick.labelsize':7,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'axes.linewidth':.6})
    fig, axes = plt.subplots(1,3,figsize=(7.05,2.38))
    fig.subplots_adjust(left=.075,right=.985,bottom=.23,top=.78,wspace=.47)
    x = np.arange(3)
    ax = axes[0]
    ax.set_title('a  Exactness and strict screening',loc='left',pad=19)
    for metric,color,offset,label in [('exact_yield',COLORS[0],-.17,'Basic exact'),
                                      ('strict_yield',COLORS[1],.17,'Strict screen')]:
        heights = np.array([a[metric]*100 for a in arms])
        ci = np.array([a['intervals_95'][metric] for a in arms])*100
        ax.bar(x+offset,heights,width=.3,color=color,label=label)
        ax.errorbar(x+offset,heights,yerr=[heights-ci[:,0],ci[:,1]-heights],
                    fmt='none',ecolor='#333333',linewidth=.7,capsize=2)
    ax.set(ylim=(0,100),yticks=[0,25,50,75,100],ylabel='Outputs / 120 jobs (%)',
           xticks=x,xticklabels=['0','4','16'],xlabel='Fragment penalty')
    ax.legend(loc='upper left',bbox_to_anchor=(-.08,1.2),frameon=False,
              ncol=2,fontsize=6.5,columnspacing=.8)
    ax = axes[1]; ax.set_title('b  Surface variety',loc='left',pad=19)
    ax.bar(x,[a['distinct_texts'] for a in arms],color=COLORS,width=.55)
    for i,a in enumerate(arms): ax.text(i,a['distinct_texts']+2,f"{a['distinct_texts']}/90",ha='center')
    ax.set(ylim=(0,100),yticks=[0,30,60,90],ylabel='Distinct texts / 90 outputs',
           xticks=x,xticklabels=['0','4','16'],xlabel='Fragment penalty')
    ax = axes[2]; ax.set_title('c  Trigram diversity',loc='left',pad=19)
    for i,a in enumerate(arms):
        val = a['mean_pairwise_trigram_jaccard_distance']
        lo,hi = a['intervals_95']['mean_pairwise_trigram_jaccard_distance']
        ax.plot([i,i],[lo,hi],color=COLORS[i],linewidth=1.3)
        ax.scatter(i,val,color=COLORS[i],s=27,zorder=3)
        ax.text(i,val+.019,f'{val:.3f}',ha='center',fontsize=7)
    ax.set(ylim=(.80,1),yticks=[.80,.85,.90,.95,1],ylabel='Mean Jaccard distance',
           xticks=x,xticklabels=['0','4','16'],xlabel='Fragment penalty',xlim=(-.5,2.5))
    save(fig,'fragment-penalty-tradeoff')

    ranked = {p:dict(data['curves'][p]['all_trigram_counts']) for p in PENALTIES}
    names = list(dict.fromkeys(t for p in (0,4) for t,n in data['curves'][p]['all_trigram_counts'][:4]))
    fig,(hm,curve) = plt.subplots(1,2,figsize=(7.05,2.55),gridspec_kw={'width_ratios':[1,.95]})
    fig.subplots_adjust(left=.16,right=.985,bottom=.24,top=.83,wspace=.52)
    values = np.array([[ranked[p].get(t,0)/data['arms'][p]['distinct_texts'] for p in PENALTIES] for t in names])
    hm.imshow(values,aspect='auto',cmap='Blues',vmin=0,vmax=1)
    hm.set(yticks=range(len(names)),yticklabels=names,xticks=range(3),
           xticklabels=[f"{p} (n={data['arms'][p]['distinct_texts']})" for p in PENALTIES],
           xlabel='Penalty (distinct texts)')
    hm.tick_params(axis='both',length=0)
    hm.set_title('a  Old and replacement fragments',loc='left',pad=11)
    for i,t in enumerate(names):
        for j,p in enumerate(PENALTIES):
            hm.text(j,i,str(ranked[p].get(t,0)),ha='center',va='center',
                    color='white' if values[i,j]>.55 else '#222222',fontsize=7)
    for p,color in zip(PENALTIES,COLORS):
        counts = [n/data['arms'][p]['distinct_texts']*100 for t,n in data['curves'][p]['all_trigram_counts'][:40]]
        curve.plot(range(1,len(counts)+1),counts,color=color,linewidth=1.25,label=f'Penalty {p}')
    curve.axhline(50,linestyle='--',color='#999999',linewidth=.6)
    curve.set(xlim=(1,40),ylim=(0,100),xticks=[1,10,20,30,40],yticks=[0,25,50,75,100],
              xlabel='Trigram rank within each arm',ylabel='Distinct texts with trigram (%)')
    curve.set_title('b  Concentration after intervention',loc='left',pad=11)
    curve.legend(frameon=False,fontsize=7,loc='upper right')
    save(fig,'fragment-penalty-motifs')

    methods = ('center_out_project','outside_in_norvig_hoey_adaptation')
    cells = []
    for method in methods:
        for band in (1,2,3):
            counts = []
            for p in PENALTIES:
                block = [r for r in rows if r['method']==method and r['band_id']==band and r['penalty']==p]
                assert len(block)==20 and sum(bool(r['text']) for r in block)==15
                counts.append(sum(r['strict_admitted_selected_output'] is True for r in block))
            cells.append(counts)
    fig,ax = plt.subplots(figsize=(3.36,2.12))
    fig.subplots_adjust(left=.39,right=.98,bottom=.23,top=.93)
    ax.imshow(np.array(cells)/20,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    labels = [f'{m}  {b}' for m in ('Center-out','Outside-in') for b in ('30-49','50-79','80-119')]
    ax.set(yticks=range(6),yticklabels=labels,xticks=range(3),xticklabels=['0','4','16'],xlabel='Fragment penalty')
    ax.tick_params(length=0)
    for i,cs in enumerate(cells):
        for j,n in enumerate(cs): ax.text(j,i,f'{n}/20',ha='center',va='center',color='white' if n>=11 else '#222222')
    save(fig,'strict-screen-strata')

    fig,ax = plt.subplots(figsize=(3.36,2.12))
    fig.subplots_adjust(left=.20,right=.93,bottom=.24,top=.94)
    colors = np.ones((4,4,4))
    colors[:,:,:3] = .97
    colors[2:,2:,:3] = (.86,.93,.86)
    ax.imshow(colors,origin='lower',aspect='auto')
    for g in range(4):
        for m in range(4):
            n=joint[g,m]
            ax.text(m,g,str(n),ha='center',va='center',fontweight='bold' if n else 'normal',
                    color='#246580' if n else '#777777')
    ax.set(xticks=range(4),yticks=range(4),xlabel='Coherent meaning score',ylabel='Grammar score')
    ax.set_xticks(np.arange(-.5,4,1),minor=True); ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.5)
    ax.tick_params(which='both',length=0)
    save(fig,'model-score-distribution')

    output = dict(data)
    output['strict_strata_counts'] = cells
    output['joint_model_score_counts'] = joint.tolist()
    output['rederived_control_pass_counts'] = dict(controls)
    output['examples'] = examples
    output['probe'] = probe_summary
    output['historical_target_counts'] = {'distinct_texts':len(old_texts), 'no it a':46}
    output['penalty_4_vs_16_different_tasks'] = differences
    output['strict_failure_counts'] = {
        p:dict(Counter(k for r in rows if r['penalty']==p and r['text']
                       for k,v in r['strict_admission_checks'].items() if not v))
        for p in PENALTIES}
    for p, spans, words in ((0,21,17),(4,58,16),(16,61,16)):
        assert output['strict_failure_counts'][p] == {
            'no_self_palindromic_proper_multiword_span':spans,
            'no_self_palindromic_word':words}
    (HERE/'derived-data.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'arms':{p:{k:data['arms'][p][k] for k in ('requested','exact_outputs','strict_outputs','distinct_texts')} for p in PENALTIES},'quality':data['quality'],'examples':len(examples),'probe':probe_summary},indent=2))


if __name__ == '__main__':
    build()
