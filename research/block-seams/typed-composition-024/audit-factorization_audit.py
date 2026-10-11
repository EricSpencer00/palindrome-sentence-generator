import json, itertools, collections
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
r=json.loads((BASE/'evidence/research/block-seams/joint-semantic-plans-023/proposals.json').read_text())
norm=lambda s:''.join(c for c in s.lower() if 'a'<=c<='z')
def np(mod, care, ref):
 b=' '.join(w for w in (mod,care,'ward') if w)
 if ref=='pronoun':return 'it'
 if ref=='ward_only':return 'the ward'
 if ref=='omit_modifier':return 'the '+' '.join(w for w in (care,'ward') if w)
 if ref=='definite_full':return 'the '+b
 if ref=='demonstrative_full':return 'that '+b
 return ('an ' if b[0] in 'aeiou' else 'a ')+b
families=collections.defaultdict(lambda:[set() for _ in range(4)])
old=set()
for x in r:
 if x['arm']!='drawing_dialogue':continue
 old.add(x['text'])
 a='Draw '; b=np(x['modifier'],x['care_type'],'indefinite_full')+'. ';d=np(x['modifier'],x['care_type'],x['reference'])+'.'
 c=x['text'][len(a+b):-len(d)]
 assert a+b+c+d==x['text']
 # Share only local role interfaces under the same entity/performer bindings.
 group=families[(x['document'],x['response_actor'])]
 for slot,t in zip(group,(a,b,c,d)):slot.add(t)
new={''.join(parts) for group in families.values() for parts in itertools.product(*group)}
assert new==old
old_exact={t for t in old if norm(t)==norm(t)[::-1]}
# Independent phrase donor paths; neither donor is exact. Feature-state
# recombination generates a target not present as a complete proposal.
donors=[['Do geese ','on a cedar ','fishing canoe ','see God?'],
        ['Do geese ','on a wooden ','trade canoe ','see God?']]
private={''.join(x) for x in donors}
shared={''.join(x) for x in itertools.product(*[sorted({d[i] for d in donors}) for i in range(4)])}
private_exact={t for t in private if norm(t)==norm(t)[::-1]}
shared_exact={t for t in shared if norm(t)==norm(t)[::-1]}
assert not private_exact
assert shared_exact=={'Do geese on a cedar trade canoe see God?'}
result={
 '023_factoring_only':{'semantic_contexts':len(families),'original_unique_texts':len(old),'factored_unique_texts':len(new),'languages_equal':new==old,'exact_texts':sorted(old_exact),'new_exact_outputs':0},
 'heldout_composition_test':{'donors':donors,'private_paths':len(private),'shared_paths':len(shared),'private_exact':sorted(private_exact),'shared_exact':sorted(shared_exact),'adds_no_words_or_chunks':True,'caution':'Known 019 target, a compiler regression only. It does not demonstrate improvement on 023 or fresh semantic quality.'}}
(BASE/'proof/results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
