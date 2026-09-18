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
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_norvig_result import audit, FUNCTION_WORDS
from llm_palindrome.sentence_plan import SentencePlan
from paper.evidence_paths import EVIDENCE, evidence_path
from paper.provenance import snapshot, current_sources


def verify():
    if sys.flags.optimize:
        raise RuntimeError("Run audits without Python optimization; assertions must remain enabled.")
    paths = [
        'inputs/brown.json.gz', 'inputs/vocab30k.txt',
        'data/structural/aggregate.json', 'inputs/norvig/npdict.txt',
        'inputs/norvig/pal3.py', 'inputs/norvig/pal21txt.html',
        'data/length/palindrome.txt', 'data/length/phrases.json',
        'data/length/result.json',
    ]
    report = {
        'scope': 'Current-file audit of saved artifacts; not a rerun of search or proof of historical runtime provenance.',
        'source_snapshot': snapshot(ROOT, current_sources(ROOT)),
        'sha256': {p: hashlib.sha256(evidence_path(p).read_bytes()).hexdigest() for p in paths},
    }
    origin = EVIDENCE / 'SOURCE-SNAPSHOT.json'
    if origin.is_file():
        report['release_source_snapshot'] = json.loads(origin.read_text())
    payload = json.loads(gzip.decompress(evidence_path(paths[0]).read_bytes()))
    plan = SentencePlan(payload['table'], payload['shapes'])
    vocab = evidence_path(paths[1]).read_text().split()[:30000]
    report['structure'] = {
        'raw_vocabulary_entries': len(vocab),
        'brown_known_vocabulary': sum(w in payload['table'] for w in vocab),
        'word_types': len(payload['table']),
        'word_tag_associations': sum(len(tags) for tags in payload['table'].values()),
        'raw_shapes': len(payload['shapes']),
        'retained_shapes': len(plan.shapes),
    }
    aggregate = json.loads(evidence_path('data/structural/aggregate.json').read_text())
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
            eligible_closures_20_to_44_letters=row['closures'],
            pos_shape_gate_rejections=row['state_pruned'],
            summed_elapsed_seconds=row['seconds'],
        )
    reference = evidence_path('inputs/norvig/pal21txt.html').read_text().split('</h1>', 1)[1].split('<hr>', 1)[0]
    reference = html.unescape(re.sub('<[^>]+>', ' ', reference))
    phrases = [re.sub('[^a-z]', '', p.lower()) for p in reference.split(',')]
    phrases = [p for p in phrases if p]
    allowed = {re.sub(r'[\W]+', '', line).lower() for line in evidence_path('inputs/norvig/npdict.txt').read_text().splitlines()}
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
    report['output'] = audit(EVIDENCE/'data/length', evidence_path('inputs/norvig/npdict.txt'), evidence_path('inputs/norvig/pal21txt.html'))
    assert report['output']['maximum_content_word_uses'] <= 3
    report['counter_definitions'] = {
        'nodes': 'Popped states; rejected children excluded.',
        'closures': 'Eligible closures with 20--44 letters, before complete acceptance checks.',
        'state_pruned': 'POS-shape gate failures only; excludes other rejection conditions.',
        'seconds': 'Sum of per-process elapsed wall seconds, not CPU seconds.',
    }
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
