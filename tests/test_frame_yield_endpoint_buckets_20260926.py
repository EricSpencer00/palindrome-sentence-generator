from experiments.frame_yield_endpoint_buckets_20260926 import audit,run,tape
def test_audit_independent_hashes():
 s='An aide rips nine memos; some men inspire Diana.';a=audit(s);t=tape(s)
 assert a['two_pointer_exact'] and a['sha256_forward']==a['sha256_reverse']
def test_endpoint_bucket_run_is_provenanced():
 p=run(40,10); assert p['endpoint_buckets']>0 and p['states']>0 and p['next_construction']
 for x in p['candidates']: assert x['provenance']['live_character_invariant'] and x['provenance']['no_posthoc_repair']
