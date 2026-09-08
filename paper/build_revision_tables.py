"""Derive the revision tables entirely from saved artifacts; no inference calls."""
import collections,json,pathlib,statistics as st,sys
P=pathlib.Path(__file__).resolve().parent;R=P.parent/'runs/revision-2026-09-07'
sys.path.insert(0,str(P.parent))
from experiments.revision_agreement import ordinal_alpha

def out(name,s): (P/name).write_text(s+'\n')
def table(spec,head,rows):
 return '\\begin{center}\\small\n\\begin{tabular}{'+spec+'}\\toprule\n'+head+' \\\\\\midrule\n'+'\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)+'\n\\bottomrule\\end{tabular}\n\\end{center}'

def main():
 c=json.loads((R/'conservation.json').read_text());rows=[]
 for a,b in zip(c[::2],c[1::2]):
  assert a['requested']==b['requested'] and a['mode']=='breadth_first' and b['mode']=='depth_first'
  rows.append([f"{a['requested']:,}",f"{a['actual']:,}",f"{a['n']:,}",f"{a['mean_net']:.2f}",f"{b['mean_net']:.2f}"])
 out('revision-conservation-table.tex',table('rrrrr','Requested $V$ & Actual $V$ & Edges per arm & BFS mean $\\Delta$ & DFS mean $\\Delta$',rows))
 by=[]
 for f in ['after_20b.json','after_120b.json']:
  d=collections.defaultdict(list)
  for r in json.loads((P.parent/'runs/punct'/f).read_text()):
   if r['score'] is not None:d[r['kind']].append(r['score'])
  by.append(d)
 rows=[]
 for arm,label in [('hand','Hand/catalogue marks'),('llm_120b','Post-hoc 120B marks'),('llm_20b','Post-hoc 20B marks'),('bare','Bare spacing'),('present','Search-time presenter')]:
  assert all(len(d[arm])==26 for d in by)
  rows.append([label]+[f'{st.mean(d[arm]):.2f}' for d in by])
 out('revision-punctuation-table.tex',table('lrr','Presentation (26 texts each) & 20B evaluator & 120B evaluator',rows))
 p=json.loads((R/'protocol.json').read_text());js=json.loads((R/'judges.json').read_text());index={r['model']:r for r in js}
 assert set(index)==set(p['models']), 'Final tables require all five attempted judges'
 rows=[]
 for m in p['models']:
  r=index[m];missing=sum(x.get('pick') is None for x in r['calibration'])
  rows.append([m,f"{r['hits']}/12",str(missing),'Pass' if r['pass'] else 'Fail',str(len(r['ratings'])) if r['pass'] else '--'])
 out('revision-judge-table.tex',table('lrrrr','Local model & Exact gate & Format/missing & Status & Ratings',rows)+'\nA failed exact gate need not mean an incorrect semantic preference. Dashes denote experimental items not evaluated under the frozen exclusion rule.')
 passing=[r for r in js if r['pass']];rows=[];summaries=[]
 for r in passing:
  b=collections.defaultdict(list)
  for x in r['ratings']:
   if x['score'] is not None:b[x['arm']].append(x['score'])
  rows.append([r['model']]+[f'{st.mean(b[k]):.2f} ({len(b[k])})' if b[k] else '--' for k in ['catalogue','mid_single','random','optimized']])
  v={x['id']:x['score'] for x in r['ratings']};diff=[];lo=hi=0
  for i in range(12):
   a=v.get(f'g{i:02}_optimized');z=v.get(f'g{i:02}_random')
   if a is not None and z is not None:diff.append(a-z)
   lo+=(a if a is not None else 0)-(z if z is not None else 3)
   hi+=(a if a is not None else 3)-(z if z is not None else 0)
  nests=b['random']+b['optimized'];summaries.append(dict(model=r['model'],means={k:st.mean(v) for k,v in b.items()},valid={k:len(v) for k,v in b.items()},nest_floor=nests.count(0),nests=len(nests),missing=sum(x['score'] is None for x in r['ratings']),complete_pairs=len(diff),paired_effect=st.mean(diff) if diff else None,bounds=[lo/12,hi/12]))
 text=table('lrrrr','Passing model & Catalogue & Single & Random nest & Ordered nest',rows)+'\nCells show mean (valid ratings). '
 for r in summaries:
  text+=f"{r['model']}: nest floor {r['nest_floor']}/{r['nests']}; missing answers {r['missing']}; {r['complete_pairs']} complete order pairs"
  if r['paired_effect'] is not None:text+=f", mean effect {r['paired_effect']:+.2f}. "
  else:text+='; effect unestimated. '
  if r['bounds'][0]!=r['bounds'][1]:text+=f"Allowing missing scores anywhere in 0--3 bounds the full order effect by [{r['bounds'][0]:+.2f}, {r['bounds'][1]:+.2f}]; this is not a confidence interval. "
 text+='No readability gain is established in this evaluation.'
 out('revision-seam-results.tex',text)
 maps=[{x['id']:x['score'] for x in r['ratings']} for r in passing]
 items=[[m.get(x['id']) for m in maps] for x in p['items']]
 pairable=sum(sum(v is not None for v in row)>=2 for row in items);alpha=ordinal_alpha(items)
 nests=[[m.get(x['id']) for m in maps] for x in p['items'] if x['arm'] in ['random','optimized']];nest_alpha=ordinal_alpha(nests)
 if alpha is not None:
  s=f'Ordinal model Krippendorff alpha is {alpha:.2f} across {len(passing)} passing models and {pairable} pairable items. It includes the readability gradient. '
 else:s='Model alpha is not estimable from this matrix. '
 s+=('Nest-only alpha is undefined because all observed nest scores are zero. ' if nest_alpha is None and all(v in [None,0] for row in nests for v in row) else f'Nest-only alpha: {nest_alpha}. ')
 out('revision-agreement.tex',s+'This is model agreement, not human validation.')
 (R/'summary.json').write_text(json.dumps(dict(judges=[dict(model=r['model'],hits=r['hits'],passed=r['pass'],attempts=len(r['ratings'])) for r in js],seams=summaries,ordinal_alpha=alpha,pairable_items=pairable,nest_alpha=nest_alpha),indent=2)+'\n')
if __name__=='__main__':main()
