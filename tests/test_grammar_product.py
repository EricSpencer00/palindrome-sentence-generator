from llm_palindrome.grammar_product import search

def test_whole_sentence_product_reports_zero_without_repairs():
    r=search([('DET','the'),('N','reader'),('V','acts'),('DET','the'),('N','reader')])
    assert r.candidates==()
    assert r.transitions>=0 and r.pruned>0
