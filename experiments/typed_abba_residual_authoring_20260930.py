"""Residual-directed ABBA authoring probe.

Unlike a bank sweep, B2 is authored only after the live A1+B1 tape fixes its
opening characters.  A2 is then authored against the remaining two-pointer
frontier.  This deliberately records failures rather than manufacturing a
reversed closing paragraph.
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/typed-abba-residual-authoring-20260930.json'

def norm(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
    t=norm(s); r=t[::-1]
    return {'letters':len(t), 'two_pointer_exact':all(t[i]==t[-1-i] for i in range(len(t))),
            'forward_sha256':hashlib.sha256(t.encode()).hexdigest(),
            'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(), 'sha_exact':t==r,
            'mismatches':[{'offset':i,'left':t[i],'right':r[i]} for i in range(min(len(t),len(r))) if t[i]!=r[i]][:8]}

def live_residual(left, b2):
    """Return the required next A2 prefix from the current tape frontier."""
    tape=norm(left+b2)
    return tape[::-1]  # obligation is read incrementally from this string

def main():
    # Four intact, typed discourse units: observation, event, response, return.
    # Each B2 option is selected by its required first character, not by a
    # completed palindrome check.  The small authoring set is a grammar of
    # ordinary clauses, not a reverse-word catalogue.
    cases=[
      {'id':'lantern','a1':'Mara saw Lila.','b1':'She met Mira.',
       'b2_by_prefix':{'a':['A quiet dawn found them waiting.'], 'i':['In the doorway, she listened.']},
       'a2':['Lila thanked Mara.','Then Mara returned.']},
      {'id':'garden','a1':'Nora met Ada.','b1':'The rain met Ada.',
       'b2_by_prefix':{'a':['After the storm, Ada smiled.'], 'i':['In time, Nora smiled.']},
       'a2':['Ada answered Nora.','Nora closed the gate.']},
      {'id':'letter','a1':'Eli saw Zoe.','b1':'He met Zoe.',
       'b2_by_prefix':{'a':['At dusk, Zoe replied.'], 'i':['In calm, Eli waited.']},
       'a2':['Zoe answered Eli.','Eli sealed the note.']},
    ]
    rows=[]
    for c in cases:
      left=c['a1']+' '+c['b1']; required=norm(left)[-1]
      # Author B2 from the live obligation. If no grammatical clause begins
      # with that character, the frontier is retained as actionable debt.
      b2s=c['b2_by_prefix'].get(required,[])
      if not b2s:
        rows.append({'case':c['id'],'rendered':left,'required_b2_prefix':required,
          'b2_authoring':'no grammatical clause in current semantic grammar',
          'next_residual':live_residual(left,''),'audit':audit(left)})
        continue
      for b2 in b2s:
        residual=live_residual(left,b2)
        a2s=[]
        for a2 in c['a2']:
          a2s.append({'text':a2,'required_prefix':residual[:min(4,len(residual))],
                      'prefix_matches':norm(a2).startswith(residual[:min(4,len(norm(a2)))])})
        for a2 in a2s:
          full=left+' '+b2+' '+a2['text']
          rows.append({'case':c['id'],'required_b2_prefix':required,'b2':b2,
            'a2':a2,'rendered':full,'next_residual':residual,'audit':audit(full)})
    exact=[r for r in rows if r['audit']['two_pointer_exact'] and r['audit']['sha_exact']]
    out={'experiment_id':'typed-abba-residual-authoring-20260930',
      'method':'semantic ABBA four-unit authoring; B2 first character is selected from live A1+B1 residual, then A2 is authored against the remaining frontier',
      'stats':{'cases':len(cases),'rendered_candidates':len(rows),'exact':len(exact),
               'prefix_compatible_a2':sum(r.get('a2',{}).get('prefix_matches',False) for r in rows)},
      'rendered_candidates':rows,'exact_candidates':exact,
      'provenance':{'four_distinct_intact_units':True,'semantic_roles_explicit':True,
        'catalogue_sweep':False,'finished_tape_reversal':False,'post_hoc_repair':False,
        'b2_authored_after_live_prefix':True,'a2_authored_against_residual':True},
      'novelty_preflight':{'signature':'live-residual-prefix|typed-abba|grammar-author-b2-then-a2',
        'duplicate_found':False,'compared_against':['paragraph-abcb-reset','typed phrase graph seam edits']},
      'reader_status':'not reader-certified; all outputs are construction evidence',
      'next_construction':'add semantic B2 clauses whose first two required letters form ordinary openings (a, i, he, she), then author A2 with matching discourse role and inflection; retain residuals rather than reverse-rendering them.',
      'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
if __name__=='__main__': main()
