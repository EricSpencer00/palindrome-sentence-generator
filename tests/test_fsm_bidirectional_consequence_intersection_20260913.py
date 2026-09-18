from experiments.fsm_bidirectional_consequence_intersection_20260913 import replay,run
def test_fsm_state_and_event_graph():
 r=run();assert r['config']['source_trie_node_and_slot_state'] and r['config']['target_trie_node_and_slot_state'];assert r['config']['shared_event_graph']
def test_thresholds_and_replay_are_explicit():
 r=run();assert r['stats']['target_terminal_completions']>=2;assert all(x['replay_ok'] for x in r['dead_ledgers']+r['diagnostics']);assert r['config']['search_status']=='exhausted'
def test_exact_survivors_are_gate_checked():
 r=run();assert all(x['independent_exact_audit']['exact'] and x['mechanically_admitted'] for x in r['admitted_exact_survivors'])
