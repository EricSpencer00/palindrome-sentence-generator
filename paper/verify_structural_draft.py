"""Recheck the saved evidence used by the structural-search working draft.

Reads existing artifacts; writes nothing unless --output is supplied.
This checks saved outputs, not the historical search execution environment.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_norvig_result import audit, FUNCTION_WORDS
from llm_palindrome.sentence_plan import SentencePlan


def verify():
    paths = [
        'tools/polaris/payload/brown.json.gz',
        'tools/polaris/payload/vocab30k.txt',
        'tools/polaris/sentence_plan_debug.py',
        'tools/polaris/sentence_plan_debug.pbs',
        'llm_palindrome/sentence_plan.py',
        'llm_palindrome/exhaustive.py',
        'llm_palindrome/pairs.py',
        'llm_palindrome/syntax.py',
        'llm_palindrome/hierarchy.py',
        'server/v3.py',
        'experiments/norvig_letters.py',
        'experiments/audit_norvig_result.py',
        'runs/polaris/sentence_plan_20260904_204815/aggregate.json',
        'runs/norvig/npdict.txt',
        'runs/norvig/pal3.py',
        'runs/norvig/pal21txt.html',
        'artifacts/norvig-v3/palindrome.txt',
        'artifacts/norvig-v3/phrases.json',
        'artifacts/norvig-v3/result.json',
    ]
    report = {
        'scope': 'Current-file audit of saved artifacts; not a rerun of search or proof of historical runtime provenance.',
        'audit_checkout': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
    }
    payload = json.loads(gzip.decompress((ROOT/paths[0]).read_bytes()))
    plan = SentencePlan(payload['table'], payload['shapes'])
    vocab = (ROOT/paths[1]).read_text().split()[:30000]
    report['structure'] = {
        'raw_vocabulary_entries': len(vocab),
        'brown_known_vocabulary': sum(w in payload['table'] for w in vocab),
        'word_tag_entries': len(payload['table']),
        'raw_shapes': len(payload['shapes']),
        'retained_shapes': len(plan.shapes),
    }
    aggregate = json.loads((ROOT/'runs/polaris/sentence_plan_20260904_204815/aggregate.json').read_text())
    report['planning'] = {}
    for name, row in aggregate.items():
        pairs = row['pairs']
        keys = {(p['left'], p['right']) for p in pairs}
        assert len(keys) == len(pairs) == row['hits']
        for pair in pairs:
            left, right = pair['left'].split(), pair['right'].split()
            assert ''.join(left) == ''.join(right)[::-1]
            assert all(re.fullmatch('[a-z]+', w) for w in left + right)
            assert 20 <= sum(map(len, left + right)) <= 44
            assert len(set(left + right)) == len(left + right)
            assert plan.complete(left) and plan.complete(right)
        report['planning'][name] = {k: v for k, v in row.items() if k != 'pairs'}
        report['planning'][name].update(
            saved_pairs_verified=len(pairs),
            pairs_per_million_popped_states=row['hits']*1e6/row['nodes'],
            pairs_per_summed_rank_second=row['hits']/row['seconds'],
        )
    reference = (ROOT/'runs/norvig/pal21txt.html').read_text().split('</h1>', 1)[1].split('<hr>', 1)[0]
    reference = html.unescape(re.sub('<[^>]+>', ' ', reference))
    phrases = [re.sub('[^a-z]', '', p.lower()) for p in reference.split(',')]
    phrases = [p for p in phrases if p]
    allowed = {re.sub(r'[\W]+', '', line).lower() for line in (ROOT/'runs/norvig/npdict.txt').read_text().splitlines()}
    assert len(phrases) == 16111 and len(set(phrases)) == len(phrases)
    assert all(p in allowed for p in phrases)
    letters = ''.join(phrases)
    assert len(letters) == 90439 and letters == letters[::-1]
    words = re.findall('[a-z]+', reference.lower())
    counts = Counter(words)
    report['reference'] = {
        'letters': len(letters), 'unique_phrases': len(phrases),
        'adjacent_repeated_word_pairs': sum(a == b for a, b in zip(words, words[1:])),
        'maximum_nonexception_token_uses': max(n for w, n in counts.items() if w not in FUNCTION_WORDS),
        'exception_words': sorted(FUNCTION_WORDS),
        'word_tokenizer': '[a-z]+ after lowercasing',
    }
    report['output'] = audit(ROOT/'artifacts/norvig-v3', ROOT/'runs/norvig/npdict.txt', ROOT/'runs/norvig/pal21txt.html')
    assert report['output']['maximum_content_word_uses'] <= 3
    report['limitations'] = [
        'Only aggregate planning results are present in the copied run directory; per-rank summaries are absent.',
        'The historical planning run did not freeze a source revision or full runtime environment in its aggregate.',
        'Identical rank seeds do not preserve identical traversal after pruning changes the random call sequence.',
        'No repeated timing trials, randomized arm order, full length-objective ablation, or new human evaluation is supplied.',
    ]
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify()
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered)
