from experiments.role_attachment_endpoint_product_20260926 import audit,tape,run
def test_audit():
 s='An aide rips nine memos; some men inspire Diana.';a=audit(s);assert a['two_pointer_exact'];assert a['sha256_forward']==a['sha256_reverse']
def test_bounded_role_product():
 p=run(40,10);assert p['endpoint_buckets'] and p['probes'] and p['next_construction']
 for x in p['candidates']:assert x['provenance']['attachment_feature_keyed'] and x['provenance']['no_finished_tape_reversal']
