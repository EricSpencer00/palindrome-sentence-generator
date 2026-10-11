"""Minimal replacement for proposal-private path compilation.
The input is independently licensed phrase productions under one abstract
semantic plan, not pre-rendered complete candidate strings. Callers own parse
licensing. Role labels alone are not evidence of grammaticality.
"""
from dataclasses import dataclass
from hashlib import sha256
from collections import Counter
import re
from llm_palindrome.bilateral_seams import Chunk

@dataclass(frozen=True)
class PhraseRule:
    entry: str
    exit: str
    text: str
    role: str
    bindings: tuple
    source: str

def compile_phrase_rules(plan_id, rules, licensed_interfaces):
    """Keep failed donors' local productions; never prefilter full donors.
    Entity IDs describe referents, not their current wording or donor IDs.
    Grammar interfaces and agreement constraints must be explicitly licensed.
    """
    out={}
    for r in rules:
        if (r.entry,r.exit) not in licensed_interfaces:
            raise ValueError('unlicensed syntactic interface')
        binding=dict(r.bindings)
        if len(binding)!=len(r.bindings):
            raise ValueError('duplicate feature key')
        key=(r.entry,r.exit,r.text,tuple(sorted(r.bindings)))
        cid='rule-'+sha256(repr((plan_id,key)).encode()).hexdigest()[:16]
        entry=r.entry if r.entry=='START' else plan_id+':'+r.entry
        exit=r.exit if r.exit=='END' else plan_id+':'+r.exit
        out[key]=Chunk(cid,r.text,entry,exit,r.role,r.source,r.bindings)
    return list(out.values())

def repetition_penalty(text,token_weight=1.0,phrase_weight=0.5):
    """A finite cost, never admission. Weights are explicit experimental knobs.
    Apply after the finite exact search for this compiler pilot, without making
    payoff a hard cutoff or changing the existing quality-review rubric.
    """
    w=re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
    token_excess=sum(n-1 for n in Counter(w).values())
    phrase_excess=sum(n-1 for size in (2,3,4)
                      for n in Counter(tuple(w[i:i+size]) for i in range(len(w)-size+1)).values())
    return token_weight*token_excess+phrase_weight*phrase_excess
