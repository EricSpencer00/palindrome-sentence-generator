from experiments.semantic_event_frame_lane_20260918 import audit, run

def test_audit_is_independent_two_pointer():
    assert audit('A man, a plan, a canal.')['two_pointer_exact'] is False
    assert audit('Able was I ere I saw Elba')['two_pointer_exact'] is True

def test_event_frame_lane_keeps_provenance_and_no_promoted_unreviewed_text():
    result=run(max_rows=40)
    assert result['provenance']['finished_tape_reversed'] is False
    assert result['provenance']['human_readability_certified'] is False
    assert result['stats']['rows']==40
    assert all(row['reader_status'].startswith('not_run') for row in result['rendered_candidates_and_probes'])
