"""Bounded source inventory audit; declared development grammar heuristics."""
import hashlib
import json
from pathlib import Path
import time
from llm_palindrome.admission import normalize_letters
from llm_palindrome.block_seams import Piece,Seam
from llm_palindrome.typed_constituents import TypedGrammar,copula

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/block-seams/fixtures'

def normwords(text):return tuple(normalize_letters(w) for w in text.split())
def state(left,right):return Seam((Piece('l',0,' '.join(left)),),(Piece('r',0,' '.join(right)),))

def parse(words):
    # Finite explicit past-tense/auxiliary frames; no POS tagger or human ratings.
    if words[:2]==('no','one'):cut=2;number='singular'
    elif words[0] in ('i','we','she','you','they','he'):cut=1;number='plural' if words[0] in ('we','you','they') else 'singular'
    elif words[0] in ('a','the'):
        cut=3 if len(words)>3 and words[1]=='old' else 2
        number='plural' if words[cut-1] in ('men','dogs') else 'singular'
    else:return None
    vp=words[cut:]
    if not vp:return None
    verb=vp[0]
    copular={'was','were','is','are','am'}
    # Every admitted verb appears in these authored sentences; past forms only.
    past={'lost','fell','had','sat','held','set','saw','ran','came','put','left','made',
          'told','drew','let','read','met','sold','lit','took','found','kept','were',
          'was','walked'}
    if verb not in copular|past|{'can'}:return None
    if verb in copular and verb!=copula(words[:cut],'past' if verb in ('was','were') else 'present'):return None
    # Original authored VP is preserved intact. Subject swapping does not certify meaning.
    return {'subject':words[:cut],'vp':vp,'number':number,'copular':verb in copular}

def run_legacy_source_bound():
    """Historical subject/whole-VP/source-bound method; not lexical scarcity evidence."""
    start=time.monotonic();deadline=start+60
    paths=[ROOT/'data/authored_sentences.txt',ROOT/'data/readable_palindrome_centres.json',
           ROOT/'data/novel_pairs.json',ROOT/'data/mirror_pairs.json',ROOT/'data/composed_sentences.json']
    authored=(paths[0]).read_text().splitlines();centres=json.loads(paths[1].read_text())
    novel=json.loads(paths[2].read_text());mirrors=json.loads(paths[3].read_text())
    composed=json.loads(paths[4].read_text())['sentences']
    config={'scope':'development inventory audit; heuristics introduced transparently, not a frozen paper method',
            'budget_seconds':60,'workers':1,'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'heuristics':['Subject boundary: I/we/she, no one, determiner+noun, determiner+old+noun.',
                          'Retain source VP intact; explicit past/copular/modal frame allowlist and subject-number agreement.',
                          'Mine all actual word-boundary prefixes/suffixes of admitted authored rows; mark subject/VP cut as constituent boundary.',
                          'Cross-source recombination is grammar-template licensing only; selectional meaning and coherence unreviewed.',
                          'One-step lookahead extends only the shorter letter tape, or either tape when balanced, using next grammar word.',
                          'Exclude identical source lineages from cross-lineage counts; known full sentences remain controls.',
                          'Mirror attested/reads flags and composed bonus values are retained source metadata, not grammar evidence.'],
            'no_search_or_model':True}
    freeze=BASE/'block-inventory-coverage-001-config.json'
    if freeze.exists():raise RuntimeError('already audited; preserve frozen run')
    freeze.write_text(json.dumps(config,indent=2)+'\n')
    parsed=[];rejected=[];prefixes={};suffixes={};units=[]
    for i,text in enumerate(authored):
        w=normwords(text);p=parse(w)
        if p is None:rejected.append({'index':i,'text':text,'reason':'outside finite declared grammar'});continue
        p.update(index=i,text=text,words=w);parsed.append(p)
        for cut in range(1,len(w)):
            for kind,part,table in [('prefix',w[:cut],prefixes),('suffix',w[cut:],suffixes)]:
                provenance={'source':f'data/authored_sentences.txt#L{i+1}','parent_text':text,
                            'word_cut':cut,'constituent_boundary':cut==len(p['subject']),
                            'lineage':f'authored-{i}','kind':kind}
                table.setdefault(part,[]).append(provenance)
                units.append({'text':' '.join(part),'provenance':provenance})
    # License finite cross-lineage complete sentence templates, then index boundaries.
    targets=[];targetprefix={};targetsuffix={}
    for a in parsed:
        for b in parsed:
            if a['index']==b['index']:continue
            vp=b['vp'];v=vp[0]
            if v in ('was','is') and a['number']=='plural':continue
            if v in ('were','are') and a['number']!='plural':continue
            # I takes am; reject cross-source is with I. Other present agreement limited.
            if v=='is' and a['subject']==('i',):continue
            w=a['subject']+vp
            j=len(targets);targets.append({'words':w,'subject_source':a['index'],'vp_source':b['index'],
                                         'text':' '.join(w),'meaning':'unreviewed'})
            for cut in range(1,len(w)):
                if w[:cut] in prefixes:targetprefix.setdefault(w[:cut],set()).add(j)
                if w[cut:] in suffixes:targetsuffix.setdefault(w[cut:],set()).add(j)
    rows=[];compatible=0;gram=0;nextviable=0;cross=0;exact=[];checked=0;timeout=False
    # Bounded actual interface enumeration; prefixes/suffixes deduplicated by text.
    for l,lprov in prefixes.items():
        for r,rprov in suffixes.items():
            if time.monotonic()>=deadline:timeout=True;break
            checked+=1;s=state(l,r);d=s.debt()
            if not d['viable']:continue
            compatible+=1
            ids=targetprefix.get(l,set()) & targetsuffix.get(r,set())
            ids={j for j in ids if len(l)+len(r)<=len(targets[j]['words'])}
            if not ids:continue
            gram+=1;opts=[];terminal=[]
            for j in sorted(ids):
                if time.monotonic()>=deadline:timeout=True;break
                t=targets[j];w=t['words'];gap=w[len(l):len(w)-len(r)]
                # Require prefix/VP units really trace to the two target source lineages.
                authentic=any(p['lineage']==f"authored-{t['subject_source']}" for p in lprov) and any(p['lineage']==f"authored-{t['vp_source']}" for p in rprov)
                if not authentic:continue
                if not gap:
                    if s.exact():terminal.append(j)
                    continue
                for side in (['right'] if d['owner']=='left' else ['left'] if d['owner']=='right' else ['left','right']):
                    word=gap[0] if side=='left' else gap[-1]
                    child=s.add(side,Piece('grammar-next',0,word))
                    if child is not None:
                        opts.append({'side':side,'word':word,'target_id':j,'new_debt':child.debt()})
            if opts:nextviable+=1;cross+=1
            if terminal:exact.extend(terminal)
            rows.append({'left':' '.join(l),'right':' '.join(r),'debt':d,
                         'left_provenance':lprov,'right_provenance':rprov,
                         'grammar_target_ids':sorted(ids),'cross_lineage_next_options':opts,
                         'exact_cross_lineage_target_ids':terminal})
        if timeout:break
    # Audit phrase resource exact paired tapes, without conferring grammar.
    phrase_bad=[]
    for name,pairs in [('novel_pairs',novel),('mirror_pairs',mirrors)]:
        for i,p in enumerate(pairs):
            if normalize_letters(' '.join(p['left']))!=normalize_letters(' '.join(p['right']))[::-1]:phrase_bad.append([name,i])
    originals={normalize_letters(x) for x in authored}|{normalize_letters(x['text']) for x in centres}
    fullcand=[dict(targets[j],existing_resource_match=normalize_letters(targets[j]['text']) in originals,
                   novelty='unverified',grammar='declared heuristic only; no human rating') for j in sorted(set(exact))]
    res={'schema_version':1,'elapsed_seconds':time.monotonic()-start,'timed_out':timeout,
         'resources':{'authored_rows':len(authored),'readable_controls':len(centres),'novel_phrase_pairs':len(novel),
                      'mirror_phrase_pairs':len(mirrors),'composed_rows':len(composed),'nonmirror_pair_rows':phrase_bad},
         'parsed_authored_rows':len(parsed),'rejected_authored_rows':rejected,
         'full_units':[{'text':p['text'],'source':f"data/authored_sentences.txt#L{p['index']+1}",
                        'grammar':'declared heuristic admitted','palindrome':normalize_letters(p['text'])==normalize_letters(p['text'])[::-1]} for p in parsed],
         'partial_units':units,'unique_prefixes':len(prefixes),'unique_suffixes':len(suffixes),
         'cross_lineage_grammar_templates':len(targets),'targets':targets,'interfaces_checked':checked,
         'letter_compatible_interfaces':compatible,'grammar_completable_interfaces':gram,
         'cross_lineage_grammar_safe_one_step_interfaces':nextviable,'interfaces':rows,
         'exact_cross_lineage_candidates':fullcand,'human_readable_new_candidates':[],
         'coverage_interpretation':'This adds source-derived grammatical frontiers, not certified complete palindrome paths. One-step survival is not eventual closure. No comparative run warranted until a complete non-control exact path exists.',
         'minimal_expansion':'Use source-attested subject/VP boundary units with number agreement and residual lookahead before considering all word cuts. All-cut counts are diagnostic, not an optimal inventory proof.',
         'novelty_claim':False,'paragraph_success':False}
    out=BASE/'block-inventory-coverage-001-results.json';out.write_text(json.dumps(res,indent=2)+'\n')
    # Derive measured constituent-only subset without another enumeration.
    subset=[x for x in rows if x['cross_lineage_next_options'] and any(p['constituent_boundary'] for p in x['left_provenance']) and any(p['constituent_boundary'] for p in x['right_provenance'])]
    compact={k:res[k] for k in ['elapsed_seconds','timed_out','resources','parsed_authored_rows','unique_prefixes','unique_suffixes','cross_lineage_grammar_templates','interfaces_checked','letter_compatible_interfaces','grammar_completable_interfaces','cross_lineage_grammar_safe_one_step_interfaces','exact_cross_lineage_candidates']}
    compact['constituent_boundary_viable_interfaces']=subset
    (BASE/'block-inventory-coverage-001-summary.json').write_text(json.dumps(compact,indent=2)+'\n')
    print(json.dumps({**{k:v for k,v in compact.items() if k!='constituent_boundary_viable_interfaces'},'constituent_boundary_viable_interface_count':len(subset),'examples':subset[:5] or [x for x in rows if x['cross_lineage_next_options']][:5]},indent=2))

def run():
    """Corrected compositional audit; source lineage is reporting metadata only."""
    started=time.monotonic();deadline=started+60
    path=ROOT/'data/authored_sentences.txt';texts=path.read_text().splitlines()
    prefixes={};suffixes={};vocab=set()
    for i,text in enumerate(texts):
        w=normwords(text);vocab.update(w)
        for cut in range(1,len(w)):
            for kind,part,table in [('prefix',w[:cut],prefixes),('suffix',w[cut:],suffixes)]:
                table.setdefault(part,[]).append({'source':f'data/authored_sentences.txt#L{i+1}',
                                                 'lineage':f'authored-{i}','word_cut':cut,'kind':kind})
    g=TypedGrammar(vocab)
    freeze=BASE/'block-inventory-compositional-002-config.json'
    if freeze.exists():raise RuntimeError('preserve prior compositional run')
    freeze.write_text(json.dumps({'grammar':'source-independent NP/predicate/complement productions; up to three sentences',
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'grammar_sha256':hashlib.sha256((ROOT/'llm_palindrome/typed_constituents.py').read_bytes()).hexdigest(),
        'workers':1,'seconds':60,'provenance_is_admission_gate':False},indent=2)+'\n')
    rows=[];checked=compatible=grammar_count=0;timeout=False
    for l,lp in prefixes.items():
        for r,rp in suffixes.items():
            if time.monotonic()>=deadline:timeout=True;break
            checked+=1;s=state(l,r);d=s.debt()
            if not d['viable']:continue
            compatible+=1
            if not g.paragraph_frontier(l,r,3):continue
            grammar_count+=1;options=[]
            for side in ('left','right'):
                for word in sorted(vocab):
                    if time.monotonic()>=deadline:timeout=True;break
                    child=s.add(side,Piece('lexical',0,word))
                    if child is None:continue
                    a=l+(word,) if side=='left' else l
                    b=(word,)+r if side=='right' else r
                    if g.paragraph_frontier(a,b,3):options.append({'side':side,'word':word,'debt':child.debt()})
                if timeout:break
            complete=g.paragraph(l+r,3)
            rows.append({'left':' '.join(l),'right':' '.join(r),'debt':d,
                         'left_provenance':lp,'right_provenance':rp,'next_options':options,
                         'source_lineage_options':{'left':sorted({p['lineage'] for p in lp}),'right':sorted({p['lineage'] for p in rp})},
                         'exact_complete':s.exact() and complete is not None,
                         'clause_features':[g.clause_features(w,ids) for w,ids in complete] if complete is not None else [],
                         'meaning':'unreviewed','originality':'unverified'})
            if timeout:break
        if timeout:break
    result={'elapsed_seconds':time.monotonic()-started,'timed_out':timeout,'interfaces_checked':checked,
            'letter_compatible':compatible,'grammar_frontier_compatible':grammar_count,'interfaces':rows,
            'provenance_is_admission_gate':False,'human_readability_claim':False,'paragraph_success_claim':False}
    (BASE/'block-inventory-compositional-002-results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':run()
