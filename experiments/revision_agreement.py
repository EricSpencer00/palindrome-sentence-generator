"""Ordinal alpha using coincidence weights and pooled marginal ranks.
Input rows are items, columns are raters; None denotes missing ratings.
Krippendorff (2011), Computing Alpha-Reliability, section D.
"""
import collections

def ordinal_alpha(items):
 coincidence=collections.defaultdict(float)
 for row in items:
  xs=[x for x in row if x is not None]
  if len(xs)<2:continue
  for i,a in enumerate(xs):
   for j,b in enumerate(xs):
    if i!=j:coincidence[a,b]+=1/(len(xs)-1)
 marginal=collections.defaultdict(float)
 for (a,b),v in coincidence.items():marginal[a]+=v
 n=sum(marginal.values())
 if n<2:return None
 rank={};total=0
 for x in sorted(marginal):rank[x]=total+marginal[x]/2;total+=marginal[x]
 observed=sum(v*(rank[a]-rank[b])**2 for (a,b),v in coincidence.items())/n
 expected=sum(na*nb*(rank[a]-rank[b])**2 for a,na in marginal.items() for b,nb in marginal.items())/(n*(n-1))
 return 1-observed/expected if expected else None

if __name__=='__main__':
 assert ordinal_alpha([[0,0],[1,1],[2,2]])==1
 assert ordinal_alpha([[0,0],[0,0]]) is None
 # Published binary example: 10 items, marginal counts 14 and 6, 4 disagreeing items (8 off-diagonal coincidences).
 a=[1,0,0,1,1,0,0,0,0,0];b=[1,1,1,0,0,0,0,0,0,0]
 assert abs(ordinal_alpha(list(zip(a,b)))-(.09523809523809534))<1e-12
 assert ordinal_alpha([[0,None,0],[1,1,None],[2,2,2]])==1
 print('Ordinal alpha checks passed (perfect, degenerate, published binary example, missing values).')
