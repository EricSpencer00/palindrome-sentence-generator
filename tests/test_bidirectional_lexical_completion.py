import itertools
import pytest
from llm_palindrome.admission import normalize_letters as norm
from llm_palindrome.bidirectional_lexical import Slot,Frame,GrammarDAG,exact_grammar_palindromes,SearchBudgetExceeded
from experiments.bidirectional_lexical_completion_20261009 import frames,enumerate_sentences,vocabulary_receipt

def test_new_word_from_two_partial_offsets_is_role_licensed_not_bank_lookup():
    f=Frame('escort',(Slot('addressee',('Noel',)),Slot('imperative',('deliver',)),Slot('human_adjective',('reviled','tired')),Slot('human_theme',('Leon',))),'vocative')
    graph=GrammarDAG([f]);old_fragment_bank=['red','tired','selfless']
    got=graph.lexical_completions('re','iled','human_adjective')
    assert got==[dict(word='reviled',role='human_adjective',left_consumed=2,right_consumed=4,middle='v')]
    assert got[0]['word'] not in old_fragment_bank
    assert graph.lexical_completions('re','iled','animate_theme')==[]
    assert graph.lexical_completions('revile','iled','human_adjective')==[]
    paths,receipt=exact_grammar_palindromes(graph)
    assert receipt['complete'] and receipt['partial_word_states']>0
    assert [graph.materialize(p)['text'] for p in paths]==['Noel, deliver reviled Leon.']
    assert not graph.accepts_words(('Noel','delivers','reviled','Leon'))
    assert not graph.accepts_words(('Noel','deliver','tired','mail'))

def test_global_exactness_rejects_locally_licensed_but_wrong_suffix():
    spec=Frame('escort',(Slot('addressee',('Noel',)),Slot('imperative',('deliver',)),Slot('human_adjective',('reviled',)),Slot('human_theme',('Anna',))),'vocative')
    graph=GrammarDAG([spec]);assert graph.accepts_words(('Noel','deliver','reviled','Anna'))
    assert graph.lexical_completions('re','iled','human_adjective')
    paths,receipt=exact_grammar_palindromes(graph)
    assert paths==() and receipt['complete']

def test_whole_stream_not_independent_palindromic_sentences():
    specs=[Frame('delivery',(Slot('addressee',('Noel',)),Slot('imperative',('deliver',)),Slot('theme',('mail',))),'vocative'),Frame('revile',(Slot('agent',('Liam',)),Slot('verb',('reviled',)),Slot('human_theme',('Leon',))))]
    graph=GrammarDAG(specs,2);paths,_=exact_grammar_palindromes(graph)
    text='Noel, deliver mail. Liam reviled Leon.'
    assert sorted(graph.materialize(p)['text'] for p in paths)==sorted([text,'Liam reviled Leon. Noel, deliver mail.'])
    assert all(norm(t)!=norm(t)[::-1] for t in ['Noel, deliver mail.','Liam reviled Leon.'])
    assert norm(text)==norm(text)[::-1]

def test_product_matches_exhaustive_including_duplicate_derivations_and_odd_midword_centers():
    specs=[Frame('x',(Slot('subject',('Anna','Eve')),)),Frame('duplicate',(Slot('subject',('Anna',)),)),Frame('two',(Slot('subject',('Noel','Leon')),Slot('verb',('refer',)),Slot('theme',('Noel','Leon'))))]
    for n in (1,2,3):
        graph=GrammarDAG(specs,n);paths,receipt=exact_grammar_palindromes(graph)
        bank=enumerate_sentences(specs);expected=[]
        for rows in itertools.product(bank,repeat=n):
            t=' '.join(r['text'] for r in rows);s=norm(t)
            if s==s[::-1]:expected.append(t)
        assert sorted(graph.materialize(p)['text'] for p in paths)==sorted(expected)
        assert receipt['complete']
    assert GrammarDAG(specs).lexical_completions('A','a','subject')[0]['word']=='Anna'

def test_coverage_and_budget_are_reported_separately():
    vocab=vocabulary_receipt(frames())
    assert 'reviled' in vocab['words_absent_from_original_fragment_bank']
    assert next(x for x in vocab['words'] if x['word']=='reviled')['source']['rule']=='past_or_adjective_d'
    graph=GrammarDAG([Frame('a',(Slot('role',('Anna',)),))])
    with pytest.raises(SearchBudgetExceeded) as error:exact_grammar_palindromes(graph,max_work=1)
    assert not error.value.receipt['complete']

def test_argument_seam_changes_noun_to_adjective_without_salvaged_body():
    specs=[f for f in frames(include_food=True) if f.name in ('food_delivery_request','stressed_criticism')]
    graph=GrammarDAG(specs,2);paths,receipt=exact_grammar_palindromes(graph)
    target='Noel, deliver Anna desserts. Stressed Anna reviled Leon.'
    assert target in [graph.materialize(p)['text'] for p in paths]
    assert all(norm(graph.materialize(p)['text'])==norm(graph.materialize(p)['text'])[::-1] for p in paths)
    assert graph.lexical_completions('str','ssed','human_adjective')[0]['word']=='stressed'
    assert not graph.accepts_words(('Noel','deliver','desserts','Anna','stressed','Anna','reviled','Leon'))
    assert len(enumerate_sentences(frames()))==248
    assert len(enumerate_sentences(frames(include_food=True)))==280
