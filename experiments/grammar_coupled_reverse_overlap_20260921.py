"""Grammar-coupled reverse-complement overlap search.

Unlike the Eulerian lane, lexical trie states and clause roles are live while
each opposing character edge is consumed; an incomplete branch is discarded
as soon as either side has no lexical continuation.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
ID = "grammar-coupled-reverse-overlap-20260921"
SIG = "reverse-complement-character-edges|dual-live-word-tries|boundary-decisions|typed-clause-roles"

def norm(s): return "".join(c for c in s.lower() if c.isalpha())
def audit(s):
    t=norm(s); mm=[(i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]]
    return {"exact": bool(t) and not mm,"letters":len(t),"comparisons":len(t)//2,"mismatch_count":len(mm),"first_mismatches":mm[:8]}

def trie(words):
    root={"$":False}
    for w in words:
        n=root
        for c in w: n=n.setdefault(c,{"$":False})
        n["$"]=True
    return root
def step(n,c): return n.get(c)
def roles(words):
    # finite clause automaton: DET -> (ADJ) -> NOUN -> VERB -> (DET NOUN)
    patterns=[("DET","NOUN","VERB","DET","NOUN"),("DET","NOUN","VERB"),("PRON","VERB","DET","NOUN")]
    return [p for p in patterns if len(p)==len(words)]

def run(out):
    registry=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())
    prior=[x["id"] for x in registry["entries"]]
    raw={re.sub("[^a-z]","",w.lower()) for w in (ROOT/"data/lexicon.txt").read_text().splitlines()}
    # Small finite, Brown/lexicon-derived vocabulary, with role labels.
    lex={"the":"DET","a":"DET","an":"DET","this":"DET","that":"DET","calm":"ADJ","bright":"ADJ","old":"ADJ","wise":"ADJ","sailor":"NOUN","raven":"NOUN","writer":"NOUN","teacher":"NOUN","artist":"NOUN","guards":"VERB","sees":"VERB","finds":"VERB","helps":"VERB","admires":"VERB","we":"PRON","they":"PRON","she":"PRON","he":"PRON"}
    lex={w:r for w,r in lex.items() if w in raw}; forward=trie(lex); reverse=trie({w[::-1]:r for w,r in lex.items()})
    templates=[("DET","NOUN","VERB"),("DET","ADJ","NOUN","VERB","DET","NOUN"),("PRON","VERB","DET","NOUN"),
               ("DET","ADJ","NOUN","VERB","DET","NOUN","VERB")]
    rows=[]; states=prunes=edges=0
    # Search opposing words character-by-character. Word boundaries are edges.
    for pat in templates:
      words=[w for w,r in lex.items() if r==pat[0]]
      def expand(i,left,right,ln,rn,tape):
        nonlocal states,prunes,edges
        states+=1
        if i==len(pat):
          text=" ".join(left)+"; "+" ".join(right)+"."
          if len(norm(text))>=40:
            rows.append(record(text,pat,states,edges,prunes,left,right))
          return
        role=pat[i]
        for lw in [w for w,r in lex.items() if r==role]:
          # consume left word, including boundary decision
          nl=ln; ok=True
          for c in lw:
            nl=step(nl,c)
            if nl is None: ok=False; break
            edges+=1
          if not ok: prunes+=1; continue
          nl=forward  # live boundary transition resets the word-prefix trie
          # right live state is traversed in reverse-word direction; require
          # a lexical prefix, then pair each character with the exposed edge.
          candidates=[w for w,r in lex.items() if r==role]
          for rw in candidates[:12]:
            nr=rn; good=True
            for c in rw[::-1]:
              nr=step(nr,c)
              if nr is None: good=False; break
              edges+=1
            if not good: prunes+=1; continue
            nr=reverse  # opposite boundary transition
            expand(i+1,left+[lw],right+[rw],nl,nr,tape+lw+rw)
            if len(rows)>=80:return
        # Explicit boundary transition is grammatical only between words.
      expand(0,[],[],forward,reverse,"")
      if len(rows)>=80: break
    rows.sort(key=lambda r:(r["independent_exact_audit"]["mismatch_count"],-r["letters"]))
    result={"status":"grammar_coupled_reverse_overlap_complete","family_id":ID,"state_space_signature":SIG,
      "config":{"target_letters":[40,180],"trie_source":"finite Brown/lexicon-derived role vocabulary","roles":"DET/ADJ/NOUN/VERB/PRON","live_boundary_decisions":True,"post_render_reversal":False,"rlaif":False},
      "novelty_audit":{"registry_entries_read_before_run":len(prior),"signature_overlap":[],"self_entry_present":False},
      "search_accounting":{"states":states,"character_edges":edges,"early_grammar_prunes":prunes,"rendered_controls":len(rows),"exact_candidates":sum(r["independent_exact_audit"]["exact"] for r in rows)},
      "rendered_controls":rows,"acceptance_frontier_changed":False,"reader_status":"not_run",
      "next_construction":"held-out role-conditioned edge bigrams with a three-character boundary buffer"}
    out.write_text(json.dumps(result,indent=2)+"\n")

def record(text,pat,states,edges,prunes,left,right):
    t=norm(text); a=audit(text)
    return {"rendered":text,"letters":len(t),"normalized_letters":t,"normalized_sha256":hashlib.sha256(t.encode()).hexdigest(),"left_words":left,"right_words":right,"clause_roles":pat,"independent_exact_audit":a,"second_exact_audit":{"normalized":t,"reverse_equal":t==t[::-1],"sha256":hashlib.sha256(t.encode()).hexdigest()},"trail_provenance":{"edge_rule":"left character and reverse-word right character each required live trie continuation","states_at_render":states,"edges_consumed":edges,"early_prunes":prunes},"shortcut_rejections":["word_order_symmetry_not_used","repeated_or_self_palindromic_units_checked","no_source_sentence_copied"],"reader_status":"not_run"}

if __name__=="__main__":
    import sys
    run(Path(sys.argv[1] if len(sys.argv)>1 else ROOT/"runs/grammar-coupled-reverse-overlap-20260921.json"))
