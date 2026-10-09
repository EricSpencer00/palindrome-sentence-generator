"""Reproduce the frozen paired analysis and figures from saved AWS results."""
from __future__ import annotations

from collections import Counter
import itertools
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fragment_penalty_20261002 import OUT, SEEDS, STRENGTHS, sha, tokens, trigrams, write
from experiments.structural_checks import STRUCTURAL_CHECKS


def validate_saved_row(row, expected_job):
    """Check internal agreement in frozen output records without rerunning admission."""
    assert all(row[k] == v for k, v in expected_job.items())
    assert row['status'] in {'selected_basic_valid_output', 'no_basic_valid_output'}
    has_text = bool(row.get('text'))
    assert has_text == (row['status'] == 'selected_basic_valid_output')

    basic = row['basic_structural_checks']
    assert set(basic) == set(STRUCTURAL_CHECKS)
    assert all(isinstance(value, bool) for value in basic.values())
    strict = row['strict_admission_checks']
    if has_text:
        assert row['letters'] == len(''.join(tokens(row['text'])))
        assert row['word_count'] == len(tokens(row['text']))
        assert strict and all(isinstance(value, bool) for value in strict.values())
        assert all(name in strict and basic[name] == strict[name] for name in STRUCTURAL_CHECKS)
        assert isinstance(row['strict_admitted_selected_output'], bool)
        assert row['strict_admitted_selected_output'] == all(strict.values())
    else:
        assert row['letters'] == 0 and row['word_count'] == 0
        assert strict == {}
        assert row['strict_admitted_selected_output'] is None
        assert not any(basic.values())


def analyze(replicates=20000):
    import numpy as np
    protocol=json.loads((OUT/'manifest.json').read_text())
    result=json.loads((OUT/'results.json').read_text())
    assert result['manifest_sha256']==sha(OUT/'manifest.json')
    assert result['status']=='complete'
    assert tuple(protocol['seeds'])==SEEDS
    assert tuple(protocol['strengths'])==STRENGTHS
    rows=result['results']
    assert len(rows)==len(protocol['jobs'])==360
    expected={r['id']:r for r in protocol['jobs']}
    assert len({r['id'] for r in rows})==360
    expected_blocks={}
    for job in protocol['jobs']:
        expected_blocks.setdefault(job['block'],set()).add(job['penalty'])
    assert len(expected_blocks)==len(SEEDS)*3*2
    assert all(penalties==set(STRENGTHS) for penalties in expected_blocks.values())
    for row in rows:
        validate_saved_row(row, expected[row['id']])
        if row.get('text'):
            import hashlib
            text=row['text']; tape=''.join(tokens(text))
            assert tape==tape[::-1] and row['min_letters']<=len(tape)<=row['max_letters']
            assert hashlib.sha256(text.encode()).hexdigest()==row['text_sha256']
    seed_index={s:i for i,s in enumerate(SEEDS)}
    draws=np.random.default_rng(20261002).integers(0,len(SEEDS),size=(replicates,len(SEEDS)))
    counts=np.stack([(draws==i).sum(axis=1) for i in range(len(SEEDS))],axis=1)
    seed_present=(counts>0).astype(float)
    summaries={}; boot={}; curves={}
    for strength in STRENGTHS:
        arm=[r for r in rows if r['penalty']==strength]
        successes=[r for r in arm if r.get('text')]
        texts=sorted({r['text'] for r in successes})
        text_index={t:i for i,t in enumerate(texts)}
        sets=[set(trigrams(tokens(t))) for t in texts]
        freq=Counter(t for ts in sets for t in ts)
        ranked=sorted(freq.items(),key=lambda x:(-x[1],x[0]))
        text_seeds=np.zeros((len(texts),len(SEEDS)))
        seed_metrics=np.zeros((len(SEEDS),2))
        for r in successes:
            text_seeds[text_index[r['text']],seed_index[r['seed']]]=1
            seed_metrics[seed_index[r['seed']],0]+=1
            seed_metrics[seed_index[r['seed']],1]+=bool(r['strict_admitted_selected_output'])
        included=(seed_present @ text_seeds.T > 0).astype(float)
        totals=included.sum(axis=1)
        # Singleton trigrams never exceed the count of any retained recurring
        # trigram; include one as a fallback without a large sparse matrix.
        recurring=[t for t,n in ranked if n>=2]
        matrix=np.array([[t in s for t in recurring] for s in sets],dtype=float)
        maxima=np.maximum(1,(included @ matrix).max(axis=1))/totals
        distances=np.zeros((len(texts),len(texts)))
        for i,j in itertools.combinations(range(len(texts)),2):
            distances[i,j]=distances[j,i]=1-len(sets[i]&sets[j])/len(sets[i]|sets[j])
        boot_dist=np.einsum('ij,ij->i',included @ distances,included)/(totals*(totals-1))
        target=set(tuple(t) for t in protocol['fragments'])
        targeted=np.array([bool(s&target) for s in sets],dtype=float)
        boot[strength]={
            'exact_yield':(counts @ seed_metrics)[:,0]/len(arm),
            'strict_yield':(counts @ seed_metrics)[:,1]/len(arm),
            'maximum_trigram_prevalence':maxima,
            'mean_pairwise_trigram_jaccard_distance':boot_dist,
            'targeted_fragment_prevalence':included @ targeted/totals,
        }
        summaries[strength]={
            'requested':len(arm),'exact_outputs':len(successes),'distinct_texts':len(texts),
            'exact_yield':len(successes)/len(arm),
            'strict_outputs':sum(bool(r['strict_admitted_selected_output']) for r in successes),
            'strict_yield':sum(bool(r['strict_admitted_selected_output']) for r in successes)/len(arm),
            'maximum_trigram_prevalence':ranked[0][1]/len(texts),
            'mean_pairwise_trigram_jaccard_distance':float(distances.sum()/(len(texts)*(len(texts)-1))),
            'targeted_fragment_prevalence':float(targeted.mean()),
            'distinct_texts_with_targeted_fragments':int(targeted.sum()),
            'successful_records_with_targeted_fragments':sum(r['target_fragment_occurrences']>0 for r in successes),
            'top_trigrams':[[' '.join(t),n] for t,n in ranked[:10]],
            'mean_elapsed_seconds':statistics.mean(r['elapsed_seconds'] for r in arm),
            'mean_scorer_calls':statistics.mean(r['scorer_calls'] for r in arm),
            'deadline_count':sum('deadline' in r['status'] for r in arm),
            'intervals_95':{k:[float(v) for v in np.quantile(values,[.025,.975])] for k,values in boot[strength].items()},
            'strata':[],
        }
        summaries[strength]['strata']=[]
        for method in sorted({r['method'] for r in arm}):
            for band in (1,2,3):
                stratum=[r for r in arm if r['method']==method and r['band_id']==band]
                summaries[strength]['strata'].append({'method':method,'band':band,'requested':len(stratum),'exact':sum(bool(r['text']) for r in stratum),'strict':sum(bool(r.get('strict_admitted_selected_output')) for r in stratum)})
        curves[strength]={'all_trigram_counts':[[' '.join(t),n] for t,n in ranked], 'distinct_texts':texts}
    differences={p:{k:{'difference':summaries[p][k]-summaries[0][k], 'interval_95':[float(v) for v in np.quantile(boot[p][k]-boot[0][k],[.025,.975])]} for k in boot[0]} for p in (4,16)}
    quality=None
    if (OUT/'quality-results.json').exists():
        q=json.loads((OUT/'quality-results.json').read_text())
        if q['status']=='complete':
            qm=json.loads((OUT/'quality-manifest.json').read_text())
            assert q['manifest_sha256']==sha(OUT/'quality-manifest.json')
            answers={r['id']:r for r in q['responses']}
            assert len(answers)==len(qm['presentations'])
            by_sha={r['text_sha256']:answers[r['id']]['scores'] for r in qm['presentations'] if r['condition']=='candidate'}
            assert set(by_sha)=={r['text_sha256'] for r in rows if r.get('text')}
            repeats=[r for r in qm['presentations'] if r['condition']=='hidden_repeat']
            originals={r['text_sha256']:answers[r['id']]['scores'] for r in qm['presentations'] if r['condition']!='hidden_repeat'}
            quality={'distinct_candidates':len(by_sha),'score_pairs':dict(Counter(f"{v['grammaticality']},{v['coherent_meaning']}" for v in by_sha.values())), 'control_pass_counts':q['control_pass_counts'], 'control_gate_passed':q['control_gate_passed'], 'hidden_repeat_agreement':sum(answers[r['id']]['scores']==originals[r['text_sha256']] for r in repeats),'hidden_repeat_count':len(repeats),'estimated_cost_usd':q['reserved_cost_usd'],'source_sha256':sha(OUT/'quality-results.json')}
            for p in STRENGTHS:
                arm=[r for r in rows if r['penalty']==p and r.get('text')]
                summaries[p]['quality_pass_outputs']=sum(all(by_sha[r['text_sha256']][k]>=2 for k in ('grammaticality','coherent_meaning')) for r in arm)
                unique_scores=[by_sha[s] for s in {r['text_sha256'] for r in arm}]
                summaries[p]['quality_score_pairs_distinct']=dict(sorted(Counter(f"{v['grammaticality']},{v['coherent_meaning']}" for v in unique_scores).items()))
    result={'search_results_sha256':sha(OUT/'results.json'),'manifest_sha256':sha(OUT/'manifest.json'),'analysis_source_sha256':sha(__file__),'bootstrap':{'replicates':replicates,'seed':20261002,'unit':'paired search seed across all bands, methods, arms','interval':'2.5th and 97.5th percentile; conditional diversity recomputed after distinct-text collapse; describes search-seed variability, not human quality','distinct_counts':'raw finite-collection counts; no bootstrap interval for number of distinct surfaces, which has bootstrap bias'},'arms':summaries,'paired_differences':differences,'curves':curves,'quality':quality}
    return result


def figures(data):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.linewidth':.6})
    dest=ROOT/'paper/fig'; dest.mkdir(parents=True,exist_ok=True)
    colors=['#246580','#BD6338','#707B65']
    arms=[data['arms'][p] for p in STRENGTHS]
    def save(fig,name):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for artist in fig.findobj(matplotlib.text.Text):
            if artist.get_visible() and artist.get_text():
                box=artist.get_window_extent(renderer)
                assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1,artist.get_text()
        fig.savefig(dest/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(dest/f'{name}.png',dpi=230);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.62))
    fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.79,wspace=.43)
    x=np.arange(3)
    ax=axes[0];ax.set_title('a  Coverage and strict checks',loc='left',pad=17)
    for metric,color,offset,label in [('exact_yield','#246580',-.17,'Basic exact'),('strict_yield','#BD6338',.17,'Strict')]:
        heights=np.array([a[metric]*100 for a in arms]);intervals=np.array([a['intervals_95'][metric] for a in arms])*100
        ax.bar(x+offset,heights,width=.30,color=color,label=label)
        ax.errorbar(x+offset,heights,yerr=[heights-intervals[:,0],intervals[:,1]-heights],fmt='none',ecolor='#333333',linewidth=.7,capsize=2)
    ax.set(ylim=(0,100),yticks=[0,25,50,75,100],ylabel='Requested jobs accepted (%)',xticks=x,xticklabels=['0','4','16'],xlabel='Fragment penalty')
    ax.legend(loc='upper center',bbox_to_anchor=(.54,1.18),frameon=False,ncol=2,fontsize=6.5,columnspacing=.8)
    ax=axes[1];ax.set_title('b  More distinct surfaces',loc='left',pad=17)
    ax.bar(x,[a['distinct_texts'] for a in arms],color=colors,width=.55)
    for i,a in enumerate(arms):ax.text(i,a['distinct_texts']+2,str(a['distinct_texts']),ha='center',fontsize=8)
    ax.set(ylim=(0,100),yticks=[0,30,60,90],ylabel='Distinct texts / 90 successes',xticks=x,xticklabels=['0','4','16'],xlabel='Fragment penalty')
    ax=axes[2];ax.set_title('c  Less trigram overlap',loc='left',pad=17)
    metric='mean_pairwise_trigram_jaccard_distance'
    for i,a in enumerate(arms):
        val=a[metric];lo,hi=a['intervals_95'][metric]
        ax.plot([i,i],[lo,hi],color=colors[i],linewidth=1.3)
        ax.scatter(i,val,color=colors[i],s=27,zorder=3)
        ax.text(i,val+.021,f'{val:.3f}',ha='center',fontsize=7)
    ax.set(ylim=(.80,1.0),yticks=[.80,.85,.90,.95,1.0],ylabel='Mean pairwise Jaccard distance',xticks=x,xticklabels=['0','4','16'],xlabel='Fragment penalty',xlim=(-.5,2.5))
    save(fig,'fragment-penalty-tradeoff')
    ranked={p:dict(data['curves'][p]['all_trigram_counts']) for p in STRENGTHS}
    names=list(dict.fromkeys([t for p in (0,4) for t,n in data['curves'][p]['all_trigram_counts'][:4]]))
    fig,(hm,curve)=plt.subplots(1,2,figsize=(7.05,2.68),gridspec_kw={'width_ratios':[1,.95]})
    fig.subplots_adjust(left=.16,right=.985,bottom=.23,top=.84,wspace=.50)
    values=np.array([[ranked[p].get(t,0)/data['arms'][p]['distinct_texts'] for p in STRENGTHS] for t in names])
    hm.imshow(values,aspect='auto',cmap='Blues',vmin=0,vmax=1)
    hm.set(yticks=range(len(names)),yticklabels=names,xticks=range(3),xticklabels=['0 (n=69)','4 (n=78)','16 (n=78)'],xlabel='Fragment penalty (distinct texts)')
    hm.tick_params(axis='both',length=0)
    hm.set_title('a  Targeted motifs give way to others',loc='left',pad=11)
    for i in range(len(names)):
        for j,p in enumerate(STRENGTHS):hm.text(j,i,f'{values[i,j]:.0%}',ha='center',va='center',color='white' if values[i,j]>.55 else '#222222',fontsize=7)
    for p,color in zip(STRENGTHS,colors):
        counts=[n/data['arms'][p]['distinct_texts']*100 for t,n in data['curves'][p]['all_trigram_counts']]
        curve.plot(range(1,min(40,len(counts))+1),counts[:40],color=color,linewidth=1.25,label=f'Penalty {p}')
    curve.axhline(50,linestyle='--',color='#999999',linewidth=.6)
    curve.set(xlim=(1,40),ylim=(0,100),xticks=[1,10,20,30,40],yticks=[0,25,50,75,100],xlabel='Trigram rank within each arm',ylabel='Distinct texts containing trigram (%)')
    curve.set_title('b  New concentration remains',loc='left',pad=11)
    curve.legend(frameon=False,fontsize=7,loc='upper right')
    save(fig,'fragment-penalty-motifs')


if __name__=='__main__':
    result=analyze();write(OUT/'analysis.json',result);figures(result)
    print(json.dumps({'arms':result['arms'],'paired_differences':result['paired_differences'],'quality':result['quality']},indent=2))
