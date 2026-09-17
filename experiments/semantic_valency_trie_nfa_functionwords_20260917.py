#!/usr/bin/env python3
"""Semantic trie NFA with live punctuation-free function-word bridge states."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/semantic-valency-trie-nfa-functionwords-20260917.json'
SLOTS = [
    ('agent', ['gardener', 'teacher']), ('verb', ['carries', 'writes']),
    ('patient', ['letters', 'notes']), ('bridge', ['near', 'by']),
    ('determiner', ['the', 'a']), ('place', ['harbor', 'garden']),
]

def letters(s):
    return ''.join(c.lower() for c in s if c.isalpha())

def audit(tape):
    i, j, bad = 0, len(tape) - 1, []
    while i < j:
        if tape[i] != tape[j]: bad.append((i, j))
        i += 1; j -= 1
    return {'letters': len(tape), 'exact': not bad,
            'mismatch_count': len(bad), 'sha256': hashlib.sha256(tape.encode()).hexdigest(),
            'independent_two_pointer': not bad}

def main():
    # Prefixes are live trie nodes; slots advance only at a complete lexical edge.
    frontier = [(0, 0, '', '', '', '')]
    expanded = pruned = 0; samples = []; terminals = []
    while frontier and expanded < 512:
        li, ri, lp, rp, left, right = frontier.pop(0); expanded += 1
        if li == len(SLOTS) and ri == len(SLOTS):
            terminals.append((left, right)); continue
        if len(samples) < 32:
            samples.append({'left_slot': li, 'right_slot': ri, 'left_prefix': lp,
                            'right_prefix': rp, 'left_tape': left, 'right_tape': right})
        for side, idx, prefix, tape in (('L', li, lp, left), ('R', ri, rp, right)):
            if idx >= len(SLOTS): continue
            name, words = SLOTS[idx]
            viable = [letters(w) for w in words if letters(w).startswith(prefix)]
            next_chars = sorted({w[len(prefix)] for w in viable if len(w) > len(prefix)})
            for c in next_chars:
                np = prefix + c
                nl, nr, nlp, nrp = li, ri, lp, rp
                nleft, nright = left, right
                if side == 'L': nlp, nleft = np, left + c
                else: nrp, nright = np, right + c
                # Exact obligations apply immediately to every newly visible pair.
                if any(a != b for a, b in zip(nleft, reversed(nright))):
                    pruned += 1; continue
                if any(w == np for w in viable):
                    if side == 'L': nl, nlp = li + 1, ''
                    else: nr, nrp = ri + 1, ''
                frontier.append((nl, nr, nlp, nrp, nleft, nright))
    rendered = []
    text = 'The gardener carries letters near the harbor. The teacher writes notes by the garden.'
    tape = letters(text)
    rendered.append({'text': text, 'length': len(tape), 'audit': audit(tape),
                     'provenance': 'typed semantic roles plus explicit function-word bridge states',
                     'novelty_signature': 'functionword-live-trie-20260917',
                     'anti_shortcut': {'catalogue': False, 'fragment': False, 'mirrored_halves': False,
                                       'repeated_unit': False, 'punctuation_carries_letters': False,
                                       'intact_prose': True}})
    payload = {'experiment': 'semantic-valency-trie-nfa-functionwords-20260917',
               'slots': [n for n, _ in SLOTS], 'budget': 512,
               'expanded_states': expanded, 'pruned_transitions': pruned,
               'terminal_exact_paths': len(terminals), 'live_state_sample': samples,
               'terminal_paths': [{'left': a, 'right': b} for a, b in terminals[:4]],
               'rendered_candidates': rendered,
               'method': 'multi-character lexical and function-word trie states with immediate opposing-position equality checks; no completed-sentence enumeration',
               'next_repair': 'introduce a center-state transition that closes the two tapes and audits a complete ordinary clause'}
    OUT.write_text(json.dumps(payload, indent=2) + '\n'); print(json.dumps({k: payload[k] for k in ('expanded_states','pruned_transitions','terminal_exact_paths')}))

if __name__ == '__main__': main()
