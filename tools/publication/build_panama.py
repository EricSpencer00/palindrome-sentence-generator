"""Publish the verified palindrome as one continuous, naturally wrapping paragraph."""
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'artifacts/norvig-long'
out = ROOT / 'web/public/panama'
out.mkdir(parents=True, exist_ok=True)
text = ' '.join((source / 'palindrome.txt').read_text().split())
meta = json.loads((source / 'result.json').read_text())
letters = re.sub('[^a-z]', '', text.lower())
assert letters == letters[::-1] and len(letters) == meta['letters']
assert len(re.findall('[a-z]+', text.lower())) == meta['words']
page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Panama | Eric Spencer</title>'
        '<link rel="canonical" href="https://palindrome.ericspencer.us/panama">'
        '<style>html,body{margin:0;min-height:100%;background:#fff;color:#111}'
        'body{font-family:Georgia,"Times New Roman",serif}'
        'main{width:50%;margin:0 auto;padding:4rem 0;box-sizing:border-box;'
        'font-size:18px;line-height:1.6;overflow-wrap:break-word}'
        '@media (max-width:800px){main{width:calc(100% - 2rem);padding:2rem 0}}'
        '</style></head><body><main>'
        + html.escape(text) + '</main></body></html>')
(out / 'index.html').write_text(page)
print(f'One flowing paragraph: {len(letters):,} verified palindrome letters.')
