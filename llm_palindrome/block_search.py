"""Versioned two-sided typed block beam, distinct from reference word search.

Both sides may grow whenever exact residual cancellation remains viable.
Grammar is supplied independently of provenance. All limits are explicit;
finite-beam truncation is reported and never described as exhaustive search.
"""
from dataclasses import dataclass
import math
import random
import time
import signal
import threading
from contextlib import contextmanager
from .admission import normalize_letters
from .block_seams import Piece,Seam

BLOCK_SEARCH_VERSION='typed-two-sided-block-beam-v1'


class _HardDeadline(Exception):pass


@contextmanager
def _deadline_guard(deadline):
    if deadline is None:yield;return
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('hard callback deadline requires the single main worker')
    if signal.getitimer(signal.ITIMER_REAL)[0]:raise ValueError('existing timer must not be overwritten')
    previous=signal.getsignal(signal.SIGALRM)
    def stop(signum,frame):raise _HardDeadline()
    signal.signal(signal.SIGALRM,stop)
    try:
        signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)


@dataclass(frozen=True)
class BlockUnit:
    id:str
    text:str
    provenance:tuple=()


@dataclass(frozen=True)
class BlockAction:
    side:str
    unit:BlockUnit
    child:Seam


def units(inventory):
    result=[];ids=set()
    for i,x in enumerate(inventory):
        u=x if isinstance(x,BlockUnit) else BlockUnit(f'unit-{i}',x)
        if u.id in ids:raise ValueError('unique block IDs required')
        if not isinstance(u.text,str) or not normalize_letters(u.text):raise ValueError('nonempty ASCII-letter block required')
        ids.add(u.id);result.append(u)
    return tuple(result)


def _action(state,side,unit,grammar_accept):
    seq=state.left if side=='left' else state.right
    child=state.add(side,Piece(unit.id,len(seq),unit.text))
    if child is None:return None,'letter_mismatch'
    if grammar_accept is not None and not grammar_accept(child):return None,'unlicensed_grammar_frontier'
    return BlockAction(side,unit,child),None


def compatible_actions(state,inventory,*,grammar_accept=None,deadline=None,max_letters=None,max_words=None):
    """Actual two-sided search menu, including grammar-safe debt growth."""
    result=[]
    for side in ('left','right'):
        for unit in units(inventory):
            if deadline is not None and time.monotonic()>=deadline:return result
            if max_letters is not None and len(normalize_letters(state.text()))+len(normalize_letters(unit.text))>max_letters:continue
            if max_words is not None and sum(len(p.text.split()) for p in state.left+state.right)+len(unit.text.split())>max_words:continue
            action,_=_action(state,side,unit,grammar_accept)
            if action is not None:result.append(action)
    return result


def block_beam_search(inventory,scorer,*,initial_state=None,grammar_accept=None,
                      allow_closed=None,beam_width=32,max_steps=64,max_actions=2000,
                      min_letters=1,max_letters=239,max_words=64,seed=921,diversity=.4,
                      deadline=None,frontier_key=None):
    if beam_width<1 or max_steps<0 or max_actions<0 or min_letters<1 or max_letters<min_letters or max_words<1:
        raise ValueError('invalid resource bounds')
    if not math.isfinite(diversity) or diversity<0:raise ValueError('invalid diversity')
    inventory=units(inventory);byid={x.id:x for x in inventory}
    start=initial_state or Seam()
    if not start.debt()['viable']:raise ValueError('initial state must have viable residual')
    if sum(len(p.text.split()) for p in start.left+start.right)>max_words:
        raise ValueError('initial state exceeds word cap')
    if len(normalize_letters(start.text()))>max_letters:
        raise ValueError('initial state exceeds letter cap')
    rng=random.Random(seed);beam=[(0.0,start)];log=[];terminals=[];attempted=0
    status='completed';truncated=False;visited=0
    def expired():return deadline is not None and time.monotonic()>=deadline
    def provenance(state):
        return [{'block_id':p.sentence_id,'text':p.text,
                 'records':list(byid[p.sentence_id].provenance) if p.sentence_id in byid else [],
                 'attribution_status':'retained' if p.sentence_id in byid and byid[p.sentence_id].provenance else 'initial_or_unattributed'}
                for p in state.left+state.right]
    try:
        with _deadline_guard(deadline):
            for depth in range(max_steps+1):
                if expired():status='deadline';truncated=True;break
                pool={}
                stop=False
                for score,state in beam:
                    if expired():status='deadline';truncated=True;stop=True;break
                    visited+=1;tape=normalize_letters(state.text());n=len(tape)
                    if min_letters<=n<=max_letters and state.exact():
                        accepted=allow_closed is None or allow_closed(state)
                        terminals.append({'state':state,'score':score,'eligible':accepted,
                                          'eligibility_scope':'mechanical closure callback; no human acceptance',
                                          'source_records':provenance(state),'depth':depth})
                        if expired():status='deadline';truncated=True;stop=True;break
                    if depth==max_steps:continue
                    for side in ('left','right'):
                        for unit in inventory:
                            if expired():status='deadline';truncated=True;stop=True;break
                            if attempted>=max_actions:status='action_cap';truncated=True;stop=True;break
                            attempted+=1
                            entry={'parent_left':[p.text for p in state.left],
                                   'parent_right':[p.text for p in state.right],
                                   'side':side,'block_id':unit.id,'text':unit.text,
                                   'source_records':list(unit.provenance),'status':'in_progress'}
                            log.append(entry)
                            # Bounds precede potentially expensive grammar callbacks.
                            new_letters=n+len(normalize_letters(unit.text))
                            new_words=sum(len(p.text.split()) for p in state.left+state.right)+len(unit.text.split())
                            if new_letters>max_letters:entry.update(status='rejected',reason='letter_cap');continue
                            if new_words>max_words:entry.update(status='rejected',reason='word_cap');continue
                            action,reason=_action(state,side,unit,grammar_accept)
                            if action is None:entry.update(status='rejected',reason=reason);continue
                            if expired():entry.update(status='interrupted',reason='deadline');status='deadline';truncated=True;stop=True;break
                            child=action.child
                            l=tuple(p.text for p in child.left);r=tuple(p.text for p in child.right)
                            delta=0.0 if scorer is None else scorer.word_delta(l,r,'L' if side=='left' else 'R',unit.text,'append' if side=='left' else 'prepend')
                            if not math.isfinite(delta):raise ValueError('finite scorer delta required')
                            total=score+delta;priority=total+rng.random()*diversity
                            entry.update(status='retained_proposal',debt=child.debt(),score=total,priority=priority)
                            previous=pool.get(child)
                            if previous is None or total>previous[0]:pool[child]=(total,priority,entry)
                        if stop:break
                    if stop:break
                if stop:break
                if depth==max_steps:
                    if beam:status='step_cap'
                    break
                ranked=sorted(((score,priority,child,entry) for child,(score,priority,entry) in pool.items()),key=lambda x:-x[1])
                selected=[];buckets=set()
                for item in ranked:
                    d=item[2].debt();bucket=(d['owner'],len(d['residual'])//4,
                                              repr(frontier_key(item[2])) if frontier_key else None)
                    if bucket not in buckets and len(selected)<beam_width:
                        buckets.add(bucket);selected.append(item)
                chosen={x[2] for x in selected}
                for item in ranked:
                    if len(selected)>=beam_width:break
                    if item[2] not in chosen:selected.append(item);chosen.add(item[2])
                if len(ranked)>beam_width:truncated=True
                for _,_,child,entry in ranked:
                    entry['beam_selected']=child in chosen
                    if child not in chosen:entry['status']='beam_pruned'
                beam=[(score,state) for score,_,state,_ in selected]
                if not beam:break
    except _HardDeadline:
        status='hard_deadline';truncated=True
        for entry in log:
            if entry.get('status')=='in_progress':entry.update(status='interrupted',reason='hard_deadline')
    return {'version':BLOCK_SEARCH_VERSION,'status':status,'truncated':truncated,
            'attempted_actions':attempted,'visited_states':visited,'terminals':terminals,
            'bounded_horizon_steps':max_steps,
            'action_log':log,'remaining_beam':beam,'human_readability_verified':False,
            'coherence_assessment':'unreviewed','originality_assessment':'unverified',
            'scheduling':'both compatible sides, including temporary debt growth; reference baseline unchanged'}
