from authored_wordpair_grammar_20260921 import run

def test_bounded_joint_grammar_and_provenance():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['joint_candidates']==12; assert r['stats']['exact_count']==0; assert r['stats']['longest_letters']>38
 for c in r['candidates']:
  assert c['complete_prose'] and c['provenance']['generated_jointly_before_render']
  assert not c['provenance']['finished_tape_reversal'] and not c['provenance']['catalogue_text']

def test_independent_pointer_and_sha_audits():
 for c in run()['candidates']:
  a=c['audit']; assert a['sha256_forward'] and a['sha256_reverse']; assert a['sha256_forward'] != a['sha256_reverse']; assert not c['reader_eligible']
