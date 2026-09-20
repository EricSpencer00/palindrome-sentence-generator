from experiments.seam_family_indexed_centerout_20260921 import audit, build_index, run
def test_np_np_vp_vp_index_populated():
 ix=build_index(); assert len(ix)>20 and any(k[0]=='NP_NP' for k in ix) and any(k[1]=='VP_VP' for k in ix)
def test_clean_result_and_independent_audit():
 p=run(5,500); assert p['index_keys']>0 and p['closures']==len(p['candidates'])
 a=audit('a quiet mason'); assert a['sha256_forward']!=a['sha256_reverse'] and a['pointer_mismatches']>0
