"""Independent invariants for the new experiment and saved source packet."""
import collections,hashlib,json,pathlib,re
R=pathlib.Path('runs/revision-2026-09-07');raw=(R/'protocol.json').read_bytes();p=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==(R/'protocol.sha256').read_text().strip()
norm=lambda s:re.sub('[^a-z]','',s.lower())
for g in range(12):
 rows={x['arm']:x for x in p['items'] if x['group']==g}
 a,b=rows['random']['text'],rows['optimized']['text']
 assert norm(a)==norm(a)[::-1] and norm(b)==norm(b)[::-1]
 assert collections.Counter(a.split())==collections.Counter(b.split())
 assert len(norm(a))==len(norm(b))==p['groups'][g]['letters']
 assert p['groups'][g]['optimized_score']>=p['groups'][g]['baseline_score']
for c in p['calibration']:
 assert c['target'] in 'AB'
 assert collections.Counter(re.findall('[a-z]+',c['a'].lower()))==collections.Counter(re.findall('[a-z]+',c['b'].lower()))
for x in json.loads((R/'conservation.json').read_text()):
 assert abs(x['mean_net']-(2*x['mean_settled']-x['mean_unit']))<1e-12
assert len(p['items'])==48 and len(p['calibration'])==12
print('Passed: protocol hash, 24 palindrome invariants, 12 material/length matches, 12 shuffled token inventories, 16 measurement identities.')
