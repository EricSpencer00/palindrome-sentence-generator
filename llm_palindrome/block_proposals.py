"""Unconnected decoded-text transformer proposal interface; deterministic checks."""
from dataclasses import dataclass
import math
from typing import Protocol
from .block_seams import Piece,Seam


@dataclass(frozen=True)
class Proposal:
    side:str
    decoded_text:str
    block_id:str
    score:float
    provenance:str


class Proposer(Protocol):
    connected:bool
    def propose(self,context:dict)->list[Proposal]:...


class UnconnectedTransformer:
    connected=False
    def propose(self,context):
        raise RuntimeError('No transformer connected or invoked; choose an approved local runtime first')


def context_for(state,meaning,grammar_slots):
    return dict(left_surface=' '.join(p.text for p in state.left),right_surface=' '.join(p.text for p in state.right),
      seam_debt=state.debt(),intended_meaning=meaning,grammar_slots=grammar_slots(state) if callable(grammar_slots) else grammar_slots,
      output_contract='Return decoded block text and side; no token-level palindrome mask')


def checked_beam(state,proposer,grammar_accept,*,meaning,grammar_slots,width=8,steps=1):
    """Retain multiple decoded proposals; exact letters and grammar are separate.

    grammar_accept(state,proposal) must license lexical/syntactic context;
    it is not an automatic human meaning or readability certification.
    No inference occurs unless caller explicitly supplies a connected proposer.
    """
    if not proposer.connected:raise RuntimeError('Transformer interface is unconnected')
    if width<1 or steps<1:raise ValueError('positive beam bounds required')
    beam=[(0.0,state)];rejections=[]
    for _ in range(steps):
        pool={}
        for score,parent in beam:
            for p in proposer.propose(context_for(parent,meaning,grammar_slots)):
                if p.side not in {'left','right'} or not p.decoded_text.strip() or not math.isfinite(p.score):
                    rejections.append((p.block_id,'malformed proposal'));continue
                try:child=parent.add(p.side,Piece(p.block_id,0,p.decoded_text))
                except ValueError:child=None
                if child is None:
                    rejections.append((p.block_id,'letter infeasible'));continue
                if not grammar_accept(parent,p):
                    rejections.append((p.block_id,'unlicensed grammar'));continue
                rank=score+p.score
                if child not in pool or rank>pool[child]:pool[child]=rank
        ranked=sorted(((score,state) for state,score in pool.items()),key=lambda x:-x[0])
        # Reserve slots across debt orientation/size and grammar frontiers.
        beam=[];buckets=set()
        for item in ranked:
            d=item[1].debt();frontier=grammar_slots(item[1]) if callable(grammar_slots) else grammar_slots
            bucket=(d['owner'],len(d['residual'])//4,repr(frontier))
            if bucket not in buckets and len(beam)<width:
                buckets.add(bucket);beam.append(item)
        for item in ranked:
            if len(beam)>=width:break
            if item not in beam:beam.append(item)
        if not beam:break
    return dict(states=beam,rejections=rejections,model_identity=getattr(proposer,'identity','not supplied'),
                human_readability_verified=False)
