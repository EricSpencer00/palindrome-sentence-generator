import hashlib,json,re,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]; P=R/'runs/char-lm-semantic-beam-20260916.json'
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/char_lm_semantic_beam_20260916.py')],check=True)
 d=json.loads(P.read_text()); assert d['novelty_preflight']['passed']; assert d['summary']['max_length']>100
 for x in d['rows']:
  t=re.sub('[^a-z]','',x['rendered'].lower()); assert x['audit']['length']==len(t)
  assert x['audit']['sha256_forward']==hashlib.sha256(t.encode()).hexdigest()
  assert x['audit']['sha256_reverse']==hashlib.sha256(t[::-1].encode()).hexdigest(); assert x['audit']['self_palindromic_words']==[]
  assert x['anti_shortcut']['self_palindromic_words'] is True; assert x['anti_shortcut']['repeated_unit'] is False; assert x['next_repair']
