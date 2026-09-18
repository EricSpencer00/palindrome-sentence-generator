import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.two_letter_frame_seam_csp_20260917 import run, letters
def test_two_letter_csp_is_pre_render_and_exactly_audited():
    result=run(); assert result['stats']['rejected_pre_render']>0; assert result['stats']['rendered']>0; assert result['stats']['exact']==0
    for row in result['rendered_candidates']:
        assert row['csp']['left_signature']==row['csp']['right_signature']
        tape=letters(row['rendered']); assert row['audit']['two_pointer_exact']==(tape==tape[::-1])
