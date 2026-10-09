"""Versioned development feedback, kept separate from held-out evaluation."""
import hashlib
import json
import random
from pathlib import Path
from .admission import normalize_letters

NORMALIZATION = 'project-ascii-letters-v1'


def candidate(candidate_id, text, provenance, lineage, split='development'):
    if split not in {'development', 'held_out'}:
        raise ValueError('unknown split')
    tape = normalize_letters(text)
    if not tape or tape != tape[::-1]:
        raise ValueError('candidate must be an exact nonempty ASCII-letter palindrome')
    if not all(isinstance(x, str) and x.strip() for x in (candidate_id, provenance, lineage)):
        raise ValueError('candidate identity and provenance required')
    return dict(id=candidate_id, text=text, provenance=provenance, lineage=lineage,
                split=split, normalization=NORMALIZATION, letters=len(tape),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                normalized_sha256=hashlib.sha256(tape.encode()).hexdigest())


def validate_candidate(item):
    expected = candidate(item['id'], item['text'], item['provenance'], item['lineage'], item['split'])
    if item != expected:
        raise ValueError('candidate metadata/text mismatch')


def freeze_pairs(items, comparisons, seed):
    by_id = {item['id']: item for item in items}
    if len(by_id) != len(items):
        raise ValueError('duplicate candidate IDs')
    for item in items:
        validate_candidate(item)
        if item['split'] != 'development':
            raise ValueError('held-out items cannot enter development feedback')
    rng = random.Random(seed)
    pairs = []
    for i, (a, b) in enumerate(comparisons):
        if a == b or a not in by_id or b not in by_id:
            raise ValueError('unknown or repeated pair member')
        order = [a, b]; rng.shuffle(order)
        pairs.append(dict(id=f'pair-{i+1:03d}', A=order[0], B=order[1]))
    rng.shuffle(pairs)
    packet = dict(schema_version=1, normalization=NORMALIZATION, seed=seed,
                  candidates=items, pairs=pairs)
    packet['packet_sha256'] = packet_hash(packet)
    return packet


def packet_hash(packet):
    payload = {k: v for k, v in packet.items() if k != 'packet_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_feedback(packet, feedback, *, human_response=None):
    if packet['packet_sha256'] != packet_hash(packet):
        raise ValueError('packet changed since freezing')
    if feedback['packet_sha256'] != packet['packet_sha256']:
        raise ValueError('feedback is for another packet')
    by_id = {item['id']: item for item in packet['candidates']}
    for item in by_id.values():
        validate_candidate(item)
        if item['split'] != 'development':
            raise ValueError('held-out label cannot enter development feedback')
    pair = next((p for p in packet['pairs'] if p['id'] == feedback['pair_id']), None)
    if pair is None or feedback['preference'] not in {'A', 'B', 'neither', 'tie'}:
        raise ValueError('invalid pair preference')
    kind = feedback['rater_kind']
    if kind not in {'human', 'machine'} or not feedback['rater_id'].strip():
        raise ValueError('explicit rater identity required')
    if kind == 'human':
        if not human_response or feedback.get('raw_response') != human_response:
            raise ValueError('actual human response required; no inferred human labels')
    elif 'raw_response' in feedback:
        raise ValueError('machine labels cannot carry a human response')
    scores = feedback.get('scores', {})
    if not isinstance(scores, dict) or (scores and set(scores) != {pair['A'], pair['B']}):
        raise ValueError('scores must describe precisely the two displayed items')
    if kind == 'machine' and not scores:
        raise ValueError('machine labels require separate grammar and meaning ratings')
    for rating in scores.values():
        for key in ('grammar', 'meaning'):
            if type(rating[key]) is not int or not 0 <= rating[key] <= 3:
                raise ValueError('scores must be separate integer scales from 0 to 3')
        if not isinstance(rating['rationale'], str) or not rating['rationale'].strip():
            raise ValueError('parse/rationale required')
        if not isinstance(rating['flags'], list) or not all(isinstance(f, str) for f in rating['flags']):
            raise ValueError('flags must be strings')
    return dict(feedback)


def append_feedback(path, packet, feedback, *, human_response=None):
    validated = validate_feedback(packet, feedback, human_response=human_response)
    path = Path(path)
    key = tuple(validated[k] for k in ('packet_sha256', 'pair_id', 'rater_kind', 'rater_id'))
    if path.exists():
        for line in path.read_text().splitlines():
            previous = json.loads(line)
            if tuple(previous[k] for k in ('packet_sha256', 'pair_id', 'rater_kind', 'rater_id')) == key:
                raise ValueError('feedback already recorded; preserve original response')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(validated, ensure_ascii=False, sort_keys=True) + '\n')
