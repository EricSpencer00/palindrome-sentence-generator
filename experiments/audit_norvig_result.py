"""Independent standard-library audit of a searched Norvig-dictionary result."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re

FUNCTION_WORDS = set('a an the and or of to in on at for with by'.split())


def audit(directory, dictionary, reference):
    text = (directory/'palindrome.txt').read_text()
    phrases = json.loads((directory/'phrases.json').read_text())
    letters = re.sub('[^a-z]', '', text.lower())
    keys = [re.sub('[^a-z]', '', phrase.lower()) for phrase in phrases]
    allowed = {re.sub(r'[\W]+', '', line).lower() for line in dictionary.read_text().splitlines()}
    words = re.findall('[a-z]+', text.lower())
    word_counts = Counter(words)
    # The heading and author footer are not part of the reference palindrome.
    original = reference.read_text().split('</h1>', 1)[1].split('<hr>', 1)[0]
    original = html.unescape(re.sub('<[^>]+>', ' ', original))
    original_letters = re.sub('[^a-z]', '', original.lower())
    assert len(original_letters) == 90439 and original_letters == original_letters[::-1]
    assert letters == letters[::-1]
    assert ''.join(keys) == letters
    assert len(keys) == len(set(keys))
    assert all(key in allowed for key in keys)
    assert all(a != b for a,b in zip(words, words[1:]))
    assert keys[:2] == ['aman','aplan'] and keys[-2:] == ['acanal','panama']
    reference_words = len(re.findall('[a-z]+', original.lower()))
    result = dict(letters=len(letters), ascii_word_tokens=len(words), phrases=len(phrases),
                  unique_phrases=len(set(keys)), all_phrases_in_original_dictionary=True,
                  exact_letter_palindrome=True, adjacent_repeated_words=0,
                  maximum_content_word_uses=max(n for w,n in word_counts.items() if w not in FUNCTION_WORDS),
                  reference_letters=len(original_letters), reference_ascii_word_tokens=reference_words,
                  published_reference_words=21012,
                  norvig_style_word_count=len(phrases)+sum(p.count(" ") for p in phrases),
                  norvig_style_word_gain=len(phrases)+sum(p.count(" ") for p in phrases)-21012,
                  letter_gain=len(letters)-len(original_letters),
                  ascii_word_token_gain=len(words)-reference_words,
                  sha256=hashlib.sha256((directory/'palindrome.txt').read_bytes()).hexdigest())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--dictionary', type=Path, default=Path('runs/norvig/npdict.txt'))
    parser.add_argument('--reference', type=Path, default=Path('runs/norvig/pal21txt.html'))
    args = parser.parse_args()
    result = audit(args.directory, args.dictionary, args.reference)
    (args.directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
