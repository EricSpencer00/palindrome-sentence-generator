"""Package the visually checked manuscript and editable figure source."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
OUT = ROOT / 'output/pdf/paper-revision'
BASE = 'e1deffe01d256330886a8306ab53fdfcffbc7b80'


def build():
    pdf = OUT/'palindrome-paper-revised-20261009.pdf'
    if (OUT/'naacl2027.pdf').is_file():
        shutil.copyfile(OUT/'naacl2027.pdf',pdf)
    if not pdf.is_file():
        raise FileNotFoundError('Build the manuscript PDF before packaging')
    files = {f'paper/{name}':PAPER/name for name in
             ('naacl2027.tex','refs.bib','acl.sty','acl_natbib.bst','figure-requirements.txt')}
    for name in ('fragment-penalty-tradeoff','fragment-penalty-motifs',
                 'strict-screen-strata','model-score-distribution'):
        files[f'paper/fig/{name}.pdf'] = PAPER/f'fig/{name}.pdf'
    for name in ('analyze_records.py','build_figures.py','build_bundle.py',
                 'derived-data.json','README.md'):
        files[f'paper/revision/{name}'] = HERE/name
    for path in sorted((HERE/'evidence').iterdir()):
        files[f'paper/revision/evidence/{path.name}'] = path
    contents = {name:path.read_bytes() for name,path in files.items()}
    contents['README.md'] = (HERE/'README.md').read_bytes()
    contents['BUILD-MANIFEST.json'] = (json.dumps({
        'base_commit':BASE,'revision_branch':'codex/palindrome-paper-figures-20261009',
        'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'manuscript_sha256':hashlib.sha256((PAPER/'naacl2027.tex').read_bytes()).hexdigest(),
        'figures':4,'verified_examples':6,'pages':5,
        'main_content_limit_pages':4,'conference_submission':False,
        'analysis':'frozen-record reanalysis only; no search or inference',
        'build':'Tectonic using locally cached resources',
        'checks':['all pages visually inspected','independent claim and interval review',
                  'example and printed-stream exactness','unchanged frozen evidence hashes']
    },indent=2)+'\n').encode()
    contents['FILES-SHA256.json'] = (json.dumps({
        name:hashlib.sha256(payload).hexdigest()
        for name,payload in sorted(contents.items())},indent=2)+'\n').encode()
    archive = OUT/'palindrome-paper-editable-sources-20261009.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,payload in sorted(contents.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,9,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            z.writestr(info,payload)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        hashes=json.loads(z.read('FILES-SHA256.json'))
        assert all(hashlib.sha256(z.read(name)).hexdigest()==sha for name,sha in hashes.items())
    print(json.dumps({'pdf':str(pdf),'source_bundle':str(archive),
                      'bundle_files':len(contents),'bundle_bytes':archive.stat().st_size,
                      'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':
    build()
