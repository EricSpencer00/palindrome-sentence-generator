"""Provenance-tagged variable-granularity island bank, known controls only."""
import hashlib,json,time
from pathlib import Path
from llm_palindrome.palindrome_islands import full_island,partial_island,lexical_atom
from llm_palindrome.block_seams import Piece,Seam
from llm_palindrome.admission import normalize_letters

ROOT=Path(__file__).resolve().parents[1]


def build():
    source=ROOT/'data/readable_palindrome_centres.json';existing={r['id']:r['text'] for r in json.loads(source.read_text())}
    known_path=ROOT/'data/known_palindromes.json';known=set(json.loads(known_path.read_text()))
    full=[]
    controls=[('eva',existing['eva'],'data/readable_palindrome_centres.json#eva','vocative + modal question; I see bees; location cave',['Eva','I','bees','cave'],'see'),
      ('rat',existing['rat'],'data/readable_palindrome_centres.json#rat','copular question + object-relative I saw; antecedent rat',['I','rat'],'identify'),
      ('basil',existing['basil'],'data/readable_palindrome_centres.json#basil','proper-name subject + past transitive ate + quantified mass NP no basil',['Lisa Bonet','basil'],'eat'),
      ('war',existing['war'],'data/readable_palindrome_centres.json#war','Now adjunct; sir vocative; a war is won, passive finite clause',['sir','war'],'win'),
      ('dennis','Dennis sinned.','data/known_palindromes.json#dennissinned','proper-name subject + past intransitive sinned',['Dennis'],'sin'),
      ('step','Step on no pets.','data/known_palindromes.json#steponnopets','imperative step + on + quantified plural NP no pets',['implicit addressee','pets'],'step')]
    for bid,text,locator,parse,entities,relation in controls:
        assert normalize_letters(text) in known
        full.append(full_island(bid,text,locator,parse,entities,relation))
    specs=[('eva-prefix','eva','Eva, can I see bees','modal question parse; optional location pending',['attach PP in a cave']),
           ('eva-clause','eva','I see bees','complete subject + transitive predicate + plural patient',['source-anchored mirror needs i on right; adding bare I is not grammar-certified']),
           ('rat-prefix','rat','Was it a rat','copular interrogative clause',['object-relative I saw attaches to rat']),
           ('basil-prefix','basil','Lisa Bonet ate','subject + past eating predicate',['source patient no basil; source-indexed completion']),
           ('war-prefix','war','Now, sir, a war','adjunct + vocative + subject NP; incomplete finite clause',['finite passive is won']),
           ('dennis-verb-fragment','dennis','Dennis sin','subject + incomplete source past-verb lexeme',['finish source verb sinned with letters ned']),
           ('step-prefix','step','Step on no','imperative + PP with incomplete quantified NP',['plural noun pets completes on no pets'])]
    by_id={r['id']:r for r in full};partial=[]
    for bid,parent_id,text,parse,requirements in specs:
        parent=by_id[parent_id]['text'];start=parent.index(text)
        partial.append(partial_island(bid,parent,(start,start+len(text)),by_id[parent_id]['source'],parse,requirements))
    vocab_path=ROOT/'tools/polaris/payload/vocab30k.txt';words=set(vocab_path.read_text().splitlines())
    atoms=[]
    for word,slot,needs in [('see','base transitive verb','subject/imperative context and patient'),('bees','plural noun NP','predicate or appropriate determiner'),
      ('in','preposition','NP complement'),('a','singular determiner','singular noun'),('cave','singular noun','determiner or licensed name context'),
      ('pets','plural noun NP','appropriate predicate'),('won','past participle or past finite win','auxiliary/tense parse'),('carries','3sg finite transitive','singular subject and patient')]:
        atoms.append(lexical_atom('word-'+word,word,dict(locator=str(vocab_path.relative_to(ROOT)),sha256=hashlib.sha256(vocab_path.read_bytes()).hexdigest()),words,slot,[needs]))
    start=time.monotonic();joins=[];failed=[]
    for a in full:
        for b in full:
            if a['id']==b['id']:continue
            state=Seam((Piece(a['id'],0,a['text']),),(Piece(b['id'],0,b['text']),));d=state.debt()
            (joins if d['viable'] else failed).append(dict(left=a['id'],right=b['id'],debt=d))
    restorations=[]
    for p in partial:
        parent=p['source']['parent_text'];a,b=p['source']['surface_span'];restored=parent[:a]+p['text']+parent[b:]
        t=normalize_letters(restored);assert t==t[::-1]
        restorations.append(dict(partial=p['id'],prefix_completion=parent[:a],suffix_completion=parent[b:],
           restored_text=restored,normalized_sha256=hashlib.sha256(t.encode()).hexdigest(),
           status='known_parent_restoration_control_not_novel',grammar_context=p['grammar']))
    return dict(schema_version=1,full_islands=full,partial_islands=partial,lexical_atoms=atoms,
      compatible_distinct_full_outer_joins=joins,failed_distinct_full_outer_joins=failed,
      source_restoration_controls=restorations,novel_cohesive_paragraphs=[],human_ratings=None,
      bounded_join_seconds=time.monotonic()-start,
      source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,known_path,vocab_path,ROOT/'llm_palindrome/palindrome_islands.py',Path(__file__)]})


if __name__=='__main__':
    out=ROOT/'research/block-seams/fixtures/palindrome-island-bank-001.json'
    if out.exists():raise FileExistsError('preserve original bank')
    r=build();out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(full=len(r['full_islands']),partial=len(r['partial_islands']),words=len(r['lexical_atoms']),
      compatible_full_joins=len(r['compatible_distinct_full_outer_joins']),failed_full_joins=len(r['failed_distinct_full_outer_joins']),
      partial_examples=[{k:p[k] for k in ('id','text','core','left_edge','right_edge','mirror_completion_right')} for p in r['partial_islands']]),indent=2))
