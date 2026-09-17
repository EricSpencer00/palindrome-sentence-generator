"""Bounded live grammar/CSP repair of the endpoint decoder.

Unlike post-hoc reranking, ``LiveFSM`` rejects a child immediately when its
displayed half can no longer be completed to an attested subject/verb shape.
The word trie supplies only character-compatible children, so the syntax FSM
and exact bilateral tape constraint are active in the same expansion step.
This is a construction experiment, not a readability certificate.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from experiments.bidirectional_attested_span_mining import common_lexicon
from experiments.whole_sentence_shared_tape import checks
from llm_palindrome.bigram import BigramModel
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.sentence_plan import SentencePlan
from llm_palindrome.syntax import brown_tables
from server.v3 import real_words
from wordfreq import zipf_frequency
from llm_palindrome.validator import is_palindrome, normalize

class LiveFSM:
    """Finite prefix/suffix grammar automaton carried by every search state."""
    def __init__(self, plan): self.plan = plan; self.rejected = 0
    def __call__(self, left, right):
        # Character compatibility has already been enforced by beam_search's
        # consume() transition; these are the live grammar states.
        ok = self.plan.suffix_possible(left) and self.plan.prefix_possible(right)
        if not ok: self.rejected += 1
        return ok

def run(*, seeds=8, vocabulary_size=5000, min_zipf=3.5, beam=80):
    table, shapes, _ = brown_tables()
    plan = SentencePlan(table, shapes, min_words=4, max_words=9)
    vocab = sorted(common_lexicon(min_zipf).intersection(plan.table),
                   key=lambda w: (-zipf_frequency(w, 'en'), w))[:vocabulary_size]
    vocab = [w for w in vocab if real_words([w])]
    tries = WordTries(vocab)
    bigrams = BigramModel.from_file(str(ROOT/'data/count_2w.txt'), vocab=vocab)
    scorer = CoherentScorer(bigrams, freq_weight=.2, length_weight=.04, short_penalty=1.5)
    rows=[]; seen=set(); fsm=LiveFSM(plan)
    for seed in range(seeds):
        words = beam_search(tries, scorer, min_letters=30, beam_width=beam,
                            max_steps=72, candidate_limit=192, seed=20260917+seed,
                            diversity=.7, allow_state=fsm, prune_every=999,
                            max_word_uses=1)
        text=' '.join(words)
        if not words or text in seen: continue
        seen.add(text); gate=checks(words)
        rows.append({'seed':seed,'text':text,'words':words,
                     'letters':len(normalize(text)), 'exact_independent':is_palindrome(text),
                     'checks':gate,'rejection_codes':[k for k,v in gate.items() if not v]})
    return {'experiment':'live-fsm-character-obligation-decoder-20260917',
            'status':'complete_bounded_live_filter',
            'config':{'seeds':seeds,'vocabulary_size':len(vocab),'min_zipf':min_zipf,'beam':beam,
                      'grammar':'SentencePlan prefix/suffix FSM during expansion',
                      'character_constraint':'search.consume exact overhang at every child'},
            'vocabulary_sha256':hashlib.sha256('\n'.join(vocab).encode()).hexdigest(),
            'records':rows,'mechanically_admitted':[r for r in rows if not r['rejection_codes']],
            'live_grammar_rejections':fsm.rejected,
            'reader_evidence':{'status':'not_run','reason':'no mechanically admitted candidate; readability requires blinded intact prose controls'},
            'next_repair':'replace corpus-shape FSM with a typed hand-authored clause automaton and retain character obligations live',
            'provenance':{'catalogue_text':False,'borrowed_text':False,'posthoc_only':False}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    result=run(); a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'records':len(result['records']),'admitted':len(result['mechanically_admitted'])}))
if __name__=='__main__': main()
