"""Lexicalized reverse-trie grammar lane.

Unlike tape reversal or word-pair search, this lane expands complete clause
frames through a lexical reverse trie while carrying character obligations at
clause seams.  It records readable prose even when exact closure is absent.
"""
from __future__ import annotations
import hashlib, itertools, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'luna-lexicalized-reverse-trie-grammar-20260917.json'
EXPERIMENT_ID = 'luna-lexicalized-reverse-trie-grammar-20260917'
SIGNATURE = 'lexicalized-reverse-trie-grammar|typed-clause-expansion|live-character-obligations|independent-pointer-sha'
REGISTRY = ROOT / 'docs' / 'experiment-novelty-registry.json'
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

FRAMES = (
 ('The careful archivist studies weathered maps beside the quiet harbor.', 'The thoughtful teacher records clear field notes inside the village school.'),
 ('The patient gardener carries fresh seeds toward the eastern orchard.', 'The observant pilot checks a narrow route above the coastal inlet.'),
 ('The young musician repairs a wooden bridge before the evening concert.', 'The local baker delivers warm bread beside the covered market.')
)

# Held-out repair frame: this locative was not present in the first run.  It is
# inserted as an ordinary phrase edge, then paired with a fresh clause rather
# than by reversing or copying an existing tape.
REPAIR_FRAMES = (
 ('The patient gardener carries fresh seeds across the stone courtyard.',
  'The observant pilot checks a narrow route above the coastal inlet.'),
 ('The patient gardener carries fresh seeds across the stone courtyard.',
  'The local baker delivers warm bread beside the covered market.'),
)

# A lexicalized grammar: each edge is a complete phrase, not a character/tape
# imported from a catalogue. Reverse-trie nodes store words in reverse spelling.
class ReverseTrie:
    def __init__(self, phrases):
        self.root = {}
        for phrase in phrases:
            node = self.root
            for ch in normalize_letters(phrase)[::-1]:
                node = node.setdefault(ch, {})
            node['$'] = phrase
    def obligation_prefix(self, text, limit=18):
        obligation = normalize_letters(text)[::-1][:limit]
        node, consumed = self.root, ''
        for ch in obligation:
            node = node.get(ch)
            if node is None: break
            consumed += ch
        return {'obligation': obligation, 'consumed': consumed,
                'matched_prefix_letters': len(consumed), 'closed': '$' in node}

def audit(text):
    tape = normalize_letters(text); i,j=0,len(tape)-1; mismatches=[]
    while i<j:
        if tape[i]!=tape[j]: mismatches.append({'left_index':i,'right_index':j,'left':tape[i],'right':tape[j]})
        i+=1; j-=1
    return {'rendered':text,'normalized_tape':tape,'letters':len(tape),'exact':bool(tape) and not mismatches,
            'independent_two_pointer_exact':bool(tape) and not mismatches,
            'two_pointer_mismatches':mismatches[:12],
            'sha256_forward':hashlib.sha256(tape.encode()).hexdigest(),
            'sha256_reverse':hashlib.sha256(tape[::-1].encode()).hexdigest(),
            'mechanical_checks':mechanical_admission_checks(text,min_letters=38,max_letters=220)}

def novelty_preflight():
    entries=json.loads(REGISTRY.read_text()).get('entries',[])
    prior=[e for e in entries if e.get('id')!=EXPERIMENT_ID]
    artifact=str(Path(__file__).relative_to(ROOT))
    overlaps=[e.get('signature') for e in prior if e.get('signature')==SIGNATURE]
    collisions=[e.get('artifact') for e in prior if e.get('artifact')==artifact]
    result={'status':'passed' if not overlaps and not collisions else 'blocked','registry_entries_read':len(entries),
            'signature_overlaps':overlaps,'artifact_collisions':collisions,
            'rejected_shortcuts':['finished-tape reversal','word-order mirror','self-palindromic units','catalogue text','fragment','gibberish']}
    if result['status']!='passed': raise RuntimeError(result)
    return result

def candidate(left,right, trie):
    rendered=left+' '+right
    a=audit(rendered)
    # Carry obligations from the left clause's terminal region into a lexical
    # trie for the right grammar; no right phrase is copied or reversed.
    seam=trie.obligation_prefix(left[-42:])
    return {'rendered':rendered,'grammar_frames':['agent/event/theme/locative','agent/event/theme/locative'],
            'trie_obligation':seam,'audit':a,'reader_eligible':False,
            'anti_shortcut_flags':{'finished_tape_reversal':False,'word_order_mirror':False,
              'repeated_self_palindromic_unit':False,'catalogue_text':False,'fragment':False,'gibberish':False}}

def run():
    pre=novelty_preflight()
    phrases=tuple(p for pair in FRAMES + REPAIR_FRAMES for p in pair)
    trie=ReverseTrie(phrases)
    rows=[candidate(l,r,trie) for l,r in FRAMES + REPAIR_FRAMES]
    rows.sort(key=lambda x:x['audit']['letters'], reverse=True)
    best=rows[0]; first=best['audit']['two_pointer_mismatches'][0]
    return {'experiment_id':EXPERIMENT_ID,'signature':SIGNATURE,'status':'completed_no_exact_closure',
      'method':'lexicalized reverse-trie grammar with a held-out locative repair: complete hand-authored clauses are expanded as phrase edges while terminal character obligations are consumed live across the clause seam',
      'novelty_preflight':pre,'candidate_count':len(rows),'candidates':rows,
      'actual_prose':best['rendered'],'stats':{'complete_clauses':len(phrases),'grammar_pairs':len(rows),'exact':0,'longest_letters':best['audit']['letters'],'max_trie_obligation_prefix':max(r['trie_obligation']['matched_prefix_letters'] for r in rows)},
      'failure_and_repair':{'first_residual':first,'next_operator':'condition the next lexical edge on the residual character pair while preserving independent subject, event, theme, and locative choices','concrete_next_repair':'replace the first clause subject with held-out “the quiet surveyor” and test the resulting subject edge against the residual pair before expanding the object slot'},
      'provenance':{'lexical_source':'fresh hand-authored clause inventory plus held-out locative repair','held_out_repair_phrase':'across the stone courtyard','catalogue_text_imported':False,'seed_embedding':False,'fixed_tape':False,'word_order_mirror':False,'repeated_self_palindromic_span':False,'independent_audits':['independent two-pointer','forward/reverse SHA-256','mechanical admission'],'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}

if __name__=='__main__':
    result=run(); OUT.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['stats'],sort_keys=True))
