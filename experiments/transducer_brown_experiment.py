"""A small, reproducible reversible-token transducer experiment.

Each Brown-corpus span is an input tape.  The transducer accepts it only when
the reverse letter tape is another independently attested span.  This keeps
construction separate from judging: no catalogue pair is copied, and the
output is explicitly a material inventory for human reading.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.bidirectional_attested_span_mining import brown_sentences, common_lexicon
from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.validator import is_palindrome, normalize


def run(min_letters=15, max_letters=30, min_words=3, max_words=8,
        min_zipf=3.8, limit=100):
    vocab = common_lexicon(min_zipf)
    index = defaultdict(dict)
    spans = 0
    for sid, sent in enumerate(brown_sentences()):
        for start in range(len(sent)):
            words = []
            for end in range(start, min(len(sent), start + max_words)):
                w = sent[end]
                if w not in vocab: break
                words.append(w)
                tape = ''.join(words)
                if len(tape) > max_letters: break
                if len(words) >= min_words and len(tape) >= min_letters:
                    spans += 1
                    # Keep one provenance row per spelling, enough to prove
                    # independent corpus occurrence without huge output.
                    index[tape].setdefault(' '.join(words), (sid, start, end + 1))
    pairs = []
    for tape in sorted(index):
        rev = tape[::-1]
        if tape >= rev or rev not in index: continue
        for left, lp in index[tape].items():
            for right, rp in index[rev].items():
                lw, rw = left.split(), right.split()
                # No repeated units, and reject self-mirroring words (a/i etc.).
                if len(set(lw + rw)) != len(lw + rw): continue
                if any(w == w[::-1] for w in lw + rw): continue
                text = left + ' ' + right
                if not is_novel_palindrome(text): continue
                assert normalize(left) == normalize(right)[::-1]
                assert is_palindrome(text)
                pairs.append({'left': left, 'right': right, 'text': text,
                              'letters_per_side': len(tape),
                              'total_letters': 2 * len(tape),
                              'left_provenance': {'sentence_id': lp[0], 'start': lp[1], 'end': lp[2]},
                              'right_provenance': {'sentence_id': rp[0], 'start': rp[1], 'end': rp[2]},
                              'exact_palindrome': True, 'novel_catalogue': True})
                if len(pairs) >= limit: return {'config': locals_config(min_letters,max_letters,min_words,max_words,min_zipf,limit), 'indexed_spans': spans, 'vocabulary': len(vocab), 'candidates': pairs}
    return {'config': locals_config(min_letters,max_letters,min_words,max_words,min_zipf,limit), 'indexed_spans': spans, 'vocabulary': len(vocab), 'candidates': pairs}

def generated_run(limit=100, seconds=20):
    """Walk a fresh reversible token tape (no catalogue seeding)."""
    import time
    from llm_palindrome.generate import build_vocab
    from llm_palindrome.lexicon import load_lexicon
    from llm_palindrome.pairs import hunt, pair_vocabulary
    from llm_palindrome.search import WordTries
    from wordfreq import zipf_frequency
    vocab = pair_vocabulary(build_vocab(30000), lambda w: zipf_frequency(w, 'en'), load_lexicon(str(ROOT/'data/lexicon.txt')), 3.8)
    found=[]
    for left,right in hunt(WordTries(vocab), shards=500, node_budget=12000,
                            min_letters=15, max_letters=30, max_overhang=16,
                            max_units=10, min_words=3, per_family=3,
                            deadline=time.time()+seconds):
        text=' '.join(left+right)
        if is_novel_palindrome(text) and len(set(left+right)) == len(left+right) and not any(w==w[::-1] for w in left+right):
            assert is_palindrome(text)
            found.append({'left':' '.join(left),'right':' '.join(right),'text':text,
                          'letters_per_side':len(normalize(' '.join(left))),
                          'total_letters':len(normalize(text)), 'exact_palindrome':True,
                          'novel_catalogue':True, 'provenance':'fresh WordTries reversible-token walk; seed=per-shard deterministic'} )
            if len(found)>=limit: break
    return {'config': {'min_letters_per_side':15,'max_letters_per_side':30,'min_words_per_side':3,'zipf':3.8,'seconds':seconds}, 'vocabulary':len(vocab), 'candidates':found}

def locals_config(a,b,c,d,e,f):
    return {'min_letters_per_side':a,'max_letters_per_side':b,'min_words_per_side':c,'max_words_per_side':d,'min_zipf':e,'limit':f}

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--limit',type=int,default=100)
    args=ap.parse_args(); result=generated_run(limit=args.limit)
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output':args.out,'vocabulary':result['vocabulary'],'candidates':len(result['candidates'])},indent=2))
    for row in result['candidates'][:20]: print(f"{row['total_letters']:2d}  {row['left']} || {row['right']}")
