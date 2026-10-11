from experiments.multiword_seam_run_017 import tag

def test_brown_lowercase_tags_and_title_suffix_are_supported():
 assert tag('at')=='D';assert tag('nn-tl')=='N';assert tag('vbd')=='V';assert tag('in')=='P';assert tag('jj')=='J'
