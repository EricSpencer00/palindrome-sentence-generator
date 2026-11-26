"""Bounded development pilot comparing the reference beam and typed two-sided block search."""
import hashlib
import itertools
import json
from pathlib import Path
import time
import signal
from contextlib import contextmanager

from llm_palindrome.admission import normalize_letters
from llm_palindrome.block_seams import Piece, Seam
from llm_palindrome.scoring import FreqScorer
from llm_palindrome.search import State, WordTries, beam_search
from llm_palindrome.typed_constituents import TypedGrammar,NAMES
from llm_palindrome.block_search import BlockUnit,block_beam_search,compatible_actions,BLOCK_SEARCH_VERSION

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/block-seams/fixtures'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(text):
    return tuple(normalize_letters(w) for w in text.split() if normalize_letters(w))


def allocate_cells(total_seconds,index_seconds,cell_count):
    return max(0.0,total_seconds-index_seconds)/cell_count


def paragraph_cells():
    return [{'id':f'paragraph-s{seed}-{band[0]}-{arm}','seed':seed,'band':band,
             'arm':arm,'left':'','right':''}
            for seed in (921,922,923) for band in ([60,119],[120,239])
            for arm in ('baseline','block_seam')]


def contiguous_sources(unit,records):
    wanted=tokens(unit)
    return [r['lineage'] for r in records if wanted and any(
        tuple(r['words'][at:at+len(wanted)])==wanted
        for at in range(len(r['words'])-len(wanted)+1))]


def source_records(unit,records):
    wanted=tokens(unit);result=[]
    for r in records:
        for at in range(len(r['words'])-len(wanted)+1):
            if wanted and tuple(r['words'][at:at+len(wanted)])==wanted:
                result.append({'source_record':r,'word_span':[at,at+len(wanted)],
                               'attribution':'possible retained occurrence; not selected authorship'})
    return result


def render_paragraph(parsed):
    case={name.lower():name for name in NAMES};case['i']='I'
    clauses=[]
    for clause,_ in parsed:
        text=' '.join(case.get(w,w) for w in clause)
        clauses.append(text[0].upper()+text[1:])
    return '. '.join(clauses)+'.'


class BudgetDeadline(Exception):pass


@contextmanager
def deadline_guard(deadline):
    """Single-worker POSIX alarm bounds otherwise long grammar callbacks."""
    previous=signal.getsignal(signal.SIGALRM)
    def interrupt(signum,frame):raise BudgetDeadline()
    signal.signal(signal.SIGALRM,interrupt)
    signal.setitimer(signal.ITIMER_REAL,max(0.000001,deadline-time.monotonic()))
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)


def seeded(left, right):
    l, r = tokens(left), tokens(right)
    seam = Seam(tuple(Piece('seed', i, w) for i, w in enumerate(l)),
                tuple(Piece('seed', i, w) for i, w in enumerate(r)))
    d = seam.debt()
    if not d['viable']:
        raise ValueError('incompatible fixed start')
    return State(0.0, l, r, d['residual'], 'L' if d['owner'] != 'right' else 'R')


class WordAdditiveScorer:
    """Identical word-level scorer for single-word and multi-word actions."""
    def __init__(self, words):
        self.base = FreqScorer(words)
        self.calls = 0

    def word_delta(self, left, right, placement, unit, growth):
        self.calls += len(unit.split())
        l = list(itertools.chain.from_iterable(tokens(x) for x in left))
        r = list(itertools.chain.from_iterable(tokens(x) for x in right))
        added = list(tokens(unit))
        if placement == 'L':
            l = l[:-len(added)]
            sequence = added
        else:
            r = r[len(added):]
            sequence = added[::-1]
        score = 0.0
        for w in sequence:
            if placement == 'L':
                l.append(w)
            else:
                r.insert(0, w)
            score += self.base.word_delta(tuple(l), tuple(r), placement, w, growth)
        return score


def build_config():
    protocol = json.loads((BASE / 'heldout-block-seam-comparison-001.json').read_text())
    bank_path = BASE / 'palindrome-island-bank-001.json'
    bank = json.loads(bank_path.read_text())
    # Ordinary finite agreement/valency drafts; human acceptance is separate.
    scenes = [('the pilot', 'reads', 'the map'), ('the pilot', 'holds', 'the map'),
              ('the pilot', 'carries', 'the map'), ('a baker', 'moves', 'a tray'),
              ('a baker', 'holds', 'a tray'), ('a baker', 'carries', 'a tray'),
              ('the gardener', 'moves', 'a pot'), ('the gardener', 'holds', 'a pot'),
              ('the gardener', 'carries', 'a pot')]
    records = [{'words': tokens(' '.join(p)), 'parts': list(p),
                'lineage': 'scene-draft-' + str(i), 'human_validated': False}
               for i, p in enumerate(scenes)]
    for x in bank['full_islands']:
        records.append({'words': tokens(x['text']), 'parts': [' '.join(tokens(x['text']))],
                        'lineage': x['id'], 'human_validated': False,
                        'source': x['source'], 'known_control': True})
    evidence_records=list(records)
    for x in bank.get('partial_islands',[]):
        evidence_records.append({'words':tokens(x['text']),'parts':[x['text']],
            'lineage':x['id'],'source':x['source'],'known_control':True,
            'partial_grammar':x['grammar'],'completion_requirements':x['completion_requirements'],
            'attribution_type':'source-anchored partial; may contain unfinished lexeme',
            'human_validated':False})
    units = {w for x in records for w in x['words']}
    blocks = units | {part for x in records for part in x['parts']}
    for key in ('partial_islands', 'lexical_atoms'):
        for x in bank.get(key, []):
            if 'text' in x:
                blocks.add(' '.join(tokens(x['text'])))
    # Bank keys vary: use every source-anchored partial/atom record with text.
    for value in bank.values():
        if isinstance(value, list):
            for x in value:
                if isinstance(x, dict) and 'text' in x:
                    blocks.add(' '.join(tokens(x['text'])))
    # Shared lexical union includes fragment words; record ordinary membership.
    units |= {w for b in blocks for w in b.split()}
    vocab30k = set((ROOT / 'tools/polaris/payload/vocab30k.txt').read_text().lower().split())
    lexicon = set((ROOT / 'data/lexicon.txt').read_text().lower().split())
    vocab = vocab30k | lexicon
    missing = sorted(units - vocab)
    # A fragmented source word is not admitted as an ordinary lexical word.
    blocks = {b for b in blocks if b and all(w in vocab for w in b.split())}
    units = {w for b in blocks for w in b.split()}
    assert all(w in vocab for w in units)
    records = [x for x in records if all(w in units for w in x['words'])]
    for s in protocol['starts']:
        assert all(w in units for w in tokens(s['left'])+tokens(s['right']))
    # Constituent productions license recombinations absent from stored records.
    # Records supply inventory/provenance; they never define grammar admission.
    grammar = TypedGrammar(units)
    exclusions = set()
    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, (list, tuple)):
            for y in x:
                yield from strings(y)
        elif isinstance(x, dict):
            for y in x.values():
                yield from strings(y)
    for name in ('known_palindromes.json', 'readable_palindrome_centres.json'):
        exclusions.update(normalize_letters(x) for x in strings(json.loads((ROOT / 'data' / name).read_text())))
    exclusions.update(normalize_letters(x) for x in ['Liam sees mail.', 'No evil did live on.',
                                                   'No rider sees red iron.'])
    old_starts = set()
    scanned = []
    def visit(x):
        if isinstance(x, dict):
            if 'left_surface' in x and 'right_surface' in x:
                old_starts.add((tokens(x['left_surface']), tokens(x['right_surface'])))
            if isinstance(x.get('left'), str) and isinstance(x.get('right'), str):
                old_starts.add((tokens(x['left']), tokens(x['right'])))
            for y in x.values():
                visit(y)
        elif isinstance(x, list):
            for y in x:
                visit(y)
    for p in sorted(BASE.glob('*.json')):
        if 'heldout-block-seam-comparison' in p.name or 'block-seam-comparison-run' in p.name:
            continue
        if p.stat().st_size > 2_000_000:
            continue
        visit(json.loads(p.read_text())); scanned.append(p.name)
    overlaps = [s['id'] for s in protocol['starts'] if (tokens(s['left']), tokens(s['right'])) in old_starts]
    if overlaps:
        raise ValueError('prior start overlap: ' + repr(overlaps))
    cfg = dict(schema_version=3, scope='prospectively reserved development paragraph pilot; not final held-out evidence',
               cells=paragraph_cells(), words=sorted(units), blocks=sorted(blocks),
               unit_provenance={u:source_records(u,evidence_records) for u in sorted(blocks)},
               scene_records=records, grammar_production_count=len(grammar.paths),
               seeds=[921,922,923],diversity=0.4,scorer='WordAdditiveScorer over dependency-free FreqScorer; uniform frozen lexical counts',
               baseline='unchanged llm_palindrome.search.beam_search with existing initial_state and allow_state APIs',
               block_arm=BLOCK_SEARCH_VERSION+'; both compatible typed sides including temporary debt growth; separate from reference beam',
               expansion_policy_difference='Baseline uses original opposite-side indexed expansion. Block v1 explicitly enumerates both sides with residual and typed grammar checks.',
               action_accounting='Baseline cap counts letter-compatible callback proposals; block cap counts every side/unit attempt, including mismatches. Report these denominators separately, not equal-work claims.',
               budget=dict(aggregate_seconds=60, maximum_cell_seconds=5, workers=1, cells=12,
                           beam_width=32, max_expansions=2000, max_added_words=64,
                           letter_bands=[[60,119],[120,239]],minimum_clauses=2,maximum_clauses=4,
                           distinct_clauses=True,shared_overhead_reserve_seconds=0.25),
               preflight=dict(start_overlap=[], scanned_artifacts=scanned,
                              unavailable_lexemes_excluded=missing, vocabulary_verified=True,
                              vocabulary_sources={w: [f for f,vs in [('tools/polaris/payload/vocab30k.txt',vocab30k),('data/lexicon.txt',lexicon)] if w in vs] for w in sorted(units)},
                              grammar_source='source-independent finite NP/predicate/complement productions with agreement and article selection; no human meaning certification',
                              machine_ratings='No verified model ratings included for these pilots'),
               source_hashes={str(p.relative_to(ROOT)): digest(p) for p in
                   [ROOT/'llm_palindrome/search.py', ROOT/'llm_palindrome/scoring.py',
                    Path(__file__), ROOT/'llm_palindrome/block_search.py',ROOT/'llm_palindrome/typed_constituents.py',bank_path, ROOT/'tools/polaris/payload/vocab30k.txt', ROOT/'data/lexicon.txt',
                    BASE/'heldout-block-seam-comparison-001.json']},
               excluded_normalized=sorted(exclusions))
    return cfg, grammar


class ExpansionCap(Exception):
    pass


def run(run_id='003'):
    if not run_id.isdecimal():raise ValueError('numeric immutable run ID required')
    pre = time.monotonic()
    cfg, grammar = build_config()
    cfg['attribution_gaps']=[u for u,ps in cfg['unit_provenance'].items() if not ps]
    cfg['attribution_policy']='Any unresolved gap explicitly retained; no affected candidate promoted as original. Source-anchored word-internal fragments are not ordinary complete lexemes.'
    cfg['deadline_policy']='Cooperative beam checkpoints plus single-worker POSIX callback deadline; report interruptions and actual elapsed.'
    # Index both arms before equally sharing the remaining search allowance.
    tries_by_arm={arm:WordTries(cfg['words'] if arm=='baseline' else cfg['blocks'])
                  for arm in ('baseline','block_seam')}
    scorers=[WordAdditiveScorer(cfg['words']) for _ in cfg['cells']]
    block_inventory=tuple(BlockUnit(f'block-{i}',u,tuple(cfg['unit_provenance'][u])) for i,u in enumerate(cfg['blocks']))
    index_seconds=time.monotonic()-pre
    reserve=cfg['budget']['shared_overhead_reserve_seconds']
    share=min(5.0,allocate_cells(60,index_seconds+reserve,len(cfg['cells'])))
    cfg['budget']['allocated_cell_seconds']=share
    cfg['index_seconds']=index_seconds
    frozen=BASE/f'block-seam-comparison-run-{run_id}-config.json'
    if frozen.exists():raise RuntimeError('run already frozen; do not duplicate search')
    frozen.write_text(json.dumps(cfg,indent=2)+'\n')
    cells=[];global_deadline=pre+60
    for cell,scorer in zip(cfg['cells'],scorers):
        t0=time.monotonic();deadline=min(global_deadline,t0+share)
        arm=cell['arm'];band=cell['band'];init=seeded(cell['left'],cell['right'])
        tries=tries_by_arm[arm];log=[];closures=[];count=0
        def flatten(l,r):
            return tuple(w for x in l for w in tokens(x)),tuple(w for x in r for w in tokens(x))
        def allow(l,r):
            nonlocal count
            count+=1
            if count>2000 or time.monotonic()>=deadline:raise ExpansionCap()
            pending={'left':list(l),'right':list(r),'status':'in_progress','phase':'grammar_frontier',
                     'attribution_gaps':[u for u in l+r if not cfg['unit_provenance'].get(u)]}
            log.append(pending)
            a,b=flatten(l,r);n=normalize_letters(' '.join(a+b))
            seam=Seam(tuple(Piece('l',i,w) for i,w in enumerate(a)),tuple(Piece('r',i,w) for i,w in enumerate(b)))
            syntax=grammar.paragraph_frontier(a,b,max_sentences=4)
            pending['phase']='lexical_lookahead'
            next_options=[]
            if arm=='block_seam' and syntax:
                for side in ('left','right'):
                    for unit in cfg['blocks']:
                        if time.monotonic()>=deadline:raise ExpansionCap()
                        child=seam.add(side,Piece('option',0,unit))
                        if child is None:continue
                        ca=a+tokens(unit) if side=='left' else a
                        cb=tokens(unit)+b if side=='right' else b
                        if grammar.paragraph_frontier(ca,cb,max_sentences=4):
                            next_options.append({'side':side,'unit':unit,'debt':child.debt(),
                                                 'source_records':cfg['unit_provenance'][unit]})
            complete=grammar.paragraph(a+b,max_sentences=4) is not None
            reason=('word_cap' if len(a)+len(b)>64 else 'letter_cap' if len(n)>band[1]
                    else 'unlicensed_grammar_frontier' if arm=='block_seam' and not syntax
                    else 'no_grammar_safe_next_option' if arm=='block_seam' and not next_options and not complete else None)
            pending.update(dict(debt=seam.debt(),status='checked',phase='complete',
                            source_records=[{'unit':u,'records':cfg['unit_provenance'].get(u,[])} for u in l+r],
                            grammar_frontier_feasible=syntax,next_options=next_options,rejection=reason))
            return reason is None
        def close(l,r):
            a,b=flatten(l,r);tape=a+b;raw=' '.join(tape);n=normalize_letters(raw)
            pending_close={'raw_assembled_text':raw,'exact':n==n[::-1],'letters':len(n),
                           'status':'in_progress','accepted':False,'phase':'paragraph_parse',
                           'attribution_gaps':[u for u in l+r if not cfg['unit_provenance'].get(u)]}
            closures.append(pending_close)
            matches=grammar.paragraph(tape,max_sentences=4)
            rendered=render_paragraph(matches) if matches is not None else None
            written_parse=grammar.text_paragraph(rendered,4) if rendered else None
            known=n in cfg['excluded_normalized']
            distinct=matches is not None and len({tuple(w) for w,_ in matches})==len(matches)
            count_ok=matches is not None and 2<=len(matches)<=4
            eligible=(n==n[::-1] and band[0]<=len(n)<=band[1] and written_parse is not None
                      and distinct and count_ok and not known)
            if rendered:assert normalize_letters(rendered)==n
            features=[grammar.clause_features(w,ids) for w,ids in matches] if matches is not None else []
            pending_close.update(dict(status='checked',phase='complete',text=rendered or raw,
                rendering_version='typed-clause-punctuation-v1' if rendered else None,
                raw_text_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                exact=n==n[::-1],letters=len(n),grammar_complete_as_rendered=written_parse is not None,
                grammar_complete=written_parse is not None,clause_count=len(matches) if matches is not None else None,
                distinct_clauses=distinct,clause_features=features,
                source_records=[{'unit':u,'records':cfg['unit_provenance'].get(u,[])} for u in l+r],
                known_control=known,mechanically_eligible_for_review=eligible,
                coherence='unreviewed',independent_lineages='requires attribution review',accepted=False))
            return eligible
        def block_grammar(child):
            nonlocal count
            count+=1
            l=tuple(p.text for p in child.left);rr=tuple(p.text for p in child.right)
            a,b=flatten(l,rr)
            entry={'left':list(l),'right':list(rr),'status':'in_progress','phase':'grammar_frontier',
                   'source_records':[{'unit':u,'records':cfg['unit_provenance'].get(u,[])} for u in l+rr]}
            log.append(entry)
            syntax=grammar.paragraph_frontier(a,b,4);options=[]
            entry['phase']='two_sided_lexical_lookahead'
            if syntax:
                def frontier(candidate):
                    aa,bb=flatten(tuple(p.text for p in candidate.left),tuple(p.text for p in candidate.right))
                    return grammar.paragraph_frontier(aa,bb,4)
                options=compatible_actions(child,block_inventory,grammar_accept=frontier,deadline=deadline,max_letters=band[1],max_words=64)
            complete=grammar.paragraph(a+b,4) is not None
            reason='unlicensed_grammar_frontier' if not syntax else 'no_grammar_safe_next_option' if not options and not complete else None
            entry.update(status='checked',phase='complete',debt=child.debt(),grammar_frontier_feasible=syntax,
                         next_options=[{'side':o.side,'unit':o.unit.text,'block_id':o.unit.id,'debt':o.child.debt()} for o in options],rejection=reason)
            return reason is None
        status='completed';result=[]
        block_diagnostics=None
        if share<=0 or t0>=global_deadline:status='aggregate_timeout_before_cell'
        else:
            try:
                if arm=='baseline':
                    with deadline_guard(deadline):
                        result=beam_search(tries,scorer,min_letters=band[0],beam_width=32,max_steps=64,
                            candidate_limit=max(len(cfg['words']),len(cfg['blocks'])),per_parent=32,
                            seed=cell['seed'],diversity=cfg['diversity'],initial_state=init,
                            allow_state=allow,allow_closed=close,deadline=deadline)
                else:
                    start=Seam(tuple(Piece('initial-left',i,u) for i,u in enumerate(init.left)),
                               tuple(Piece('initial-right',i,u) for i,u in enumerate(init.right)))
                    block_diagnostics=block_beam_search(block_inventory,scorer,initial_state=start,
                        grammar_accept=block_grammar,
                        allow_closed=lambda child:close(tuple(p.text for p in child.left),tuple(p.text for p in child.right)),
                        min_letters=band[0],max_letters=band[1],max_words=64,max_actions=2000,
                        beam_width=32,max_steps=64,seed=cell['seed'],diversity=cfg['diversity'],deadline=deadline)
                    eligible=[t for t in block_diagnostics['terminals'] if t['eligible']]
                    if eligible:
                        best=max(eligible,key=lambda t:t['score']/len(normalize_letters(t['state'].text())))
                        result=[p.text for p in best['state'].left+best['state'].right]
                    status=block_diagnostics['status']
                if time.monotonic()>=deadline:status='timeout'
            except ExpansionCap:
                status='timeout' if time.monotonic()>=deadline else 'expansion_cap'
            except BudgetDeadline:
                status='hard_callback_timeout'
            for pending in log:
                if pending.get('status')=='in_progress':pending.update(status='interrupted',rejection=status)
            for pending in closures:
                if pending.get('status')=='in_progress':pending.update(status='interrupted',rejection=status)
        serialized_block=None
        if block_diagnostics is not None:
            serialized_block={k:v for k,v in block_diagnostics.items() if k not in ('terminals','remaining_beam')}
            serialized_block['terminals']=[{**{k:v for k,v in t.items() if k!='state'},'text':t['state'].text()} for t in block_diagnostics['terminals']]
            serialized_block['remaining_beam']=[{'score':score,'left':[p.text for p in st.left],'right':[p.text for p in st.right],'debt':st.debt()} for score,st in block_diagnostics['remaining_beam']]
        cells.append(dict(cell_id=cell['id'],arm=arm,seed=cell['seed'],band=band,status=status,
            allocated_seconds=share,elapsed_seconds=time.monotonic()-t0,expanded_actions=count,
            word_score_calls=scorer.calls,result=result,proposed_actions=log,closures=closures,
            block_search_diagnostics=serialized_block,
            note='Existing exact trie omits incompatible word actions. Source records preserve spans/order/multiplicity; no originality or coherence admission.'))
    receipt=dict(schema_version=1,config_sha256=digest(frozen),index_seconds=index_seconds,
                 aggregate_search_seconds=time.monotonic()-pre,cells=cells,
                 human_acceptance_count=0,paragraph_success_count=0,
                 novelty_claim=False,machine_ratings_imported=False)
    out=BASE/f'block-seam-comparison-run-{run_id}-results.json';out.write_text(json.dumps(receipt,indent=2)+'\n')
    timing={'execution_elapsed_including_first_artifact_write':time.monotonic()-pre,
            'search_index_and_cell_overhead_seconds':receipt['aggregate_search_seconds'],
            'aggregate_search_overrun_seconds':max(0,receipt['aggregate_search_seconds']-60),
            'maximum_cell_overrun_seconds':max([0]+[c['elapsed_seconds']-c['allocated_seconds'] for c in cells]),
            'index_seconds':index_seconds,'allocated_cell_seconds':share}
    (BASE/f'block-seam-comparison-run-{run_id}-timing.json').write_text(json.dumps(timing,indent=2)+'\n')
    print(json.dumps(dict(aggregate_search_seconds=receipt['aggregate_search_seconds'],
                         index_seconds=index_seconds,cells=[{k:c[k] for k in
                         ('cell_id','arm','seed','band','status','elapsed_seconds','expanded_actions','word_score_calls','result')}
                         | {'closures':len(c['closures']),'grammar_complete_closures':sum(x.get('grammar_complete',False) for x in c['closures'])}
                         for c in cells]),indent=2))


if __name__=='__main__':
    run()
