import importlib.util
from pathlib import Path
S=importlib.util.spec_from_file_location("lane",Path(__file__).parents[1]/"experiments/cross_word_center_repair_20260917.py")
M=importlib.util.module_from_spec(S); S.loader.exec_module(M)
def test_cross_word_seam_audit_is_exact_but_catalogue_rejected():
    r=M.run(); c=r["candidates"][0]
    assert c["letters"]==43 and c["exact"] and c["center_is_token"] is False
    assert len(r["candidates"]) == 3
    assert not any(x["exact"] and x["novel"] for x in r["candidates"])
    assert c["normalized_sha256"]==c["reverse_sha256"] and not r["admitted"]
