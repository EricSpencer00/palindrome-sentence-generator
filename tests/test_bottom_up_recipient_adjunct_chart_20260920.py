from experiments.bottom_up_recipient_adjunct_chart_20260920 import audit, chart, consume, run

def test_semantic_nonterminals_are_present():
    c=chart()
    assert c['RECIP'] and c['ADJ'] and c['DITRANS'] and c['VPADJ']
    assert any('recipient' in x.roles for x in c['DITRANS'])
    assert any('adjunct' in x.roles for x in c['VPADJ'])

def test_complete_controls_and_independent_audit():
    x=run(state_limit=12000)
    assert x['novelty_preflight']['status']=='passed'
    assert x['stats']['combines']>0
    for row in x['complete_prose_controls']:
        assert audit(row['rendered'])==row['audit']
    assert consume('an','diana')==('', 'dia')
