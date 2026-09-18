"""Validated center-out chart for an ordinary ``at noon`` event."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

# Fixed event graph: “The patient guard rests at noon before dusk.”
SLOTS=(('det',('the',)),('subject_adj',('patient','careful')),('subject',('guard','teacher')),
       ('verb',('rests','waits')),('prep',('at',)),('center',('noon','level')),
       ('time_prep',('before','after')),('time',('dusk','dawn','rain')))

def replay(ledger):
    residual=''; cancellations=0
    for e in ledger:
        c=e['char']
        if residual:
            if c != residual[0]: return {'ok':False,'events_replayed':len(ledger),'cancellations':cancellations,'residual':residual}
            residual=residual[1:]; cancellations+=1
        else: residual=c
    return {'ok':not residual,'events_replayed':len(ledger),'cancellations':cancellations,'residual':residual}

def render(words): return ' '.join(words).capitalize()+'.'
def parse(text):
    w=tokenize(text)
    return len(w)==8 and w[0]=='the' and w[1] in SLOTS[1][1] and w[2] in SLOTS[2][1] and w[3] in SLOTS[3][1] and w[4]=='at' and w[5] in SLOTS[5][1] and w[6] in SLOTS[6][1] and w[7] in SLOTS[7][1]
def audit(text,kind,prov):
    tape=normalize_letters(text); gate=mechanical_admission_checks(text,min_letters=30,max_letters=220); codes=[k for k,v in gate.items() if not v]
    if not parse(text): codes.append('independent_complete_reparse_failed')
    return {'record_kind':kind,'rendered':text,'provenance':prov,'independent_exact_audit':{'exact':bool(tape) and tape==tape[::-1],'letters':len(tape),'normalized_sha256':sha256(tape.encode()).hexdigest()},'independent_parse':parse(text),'central_admission':gate,'mechanically_admitted':not codes,'rejection_codes':codes,'reader_status':'unreviewed; programmatic checks do not certify readability'}

def run(*,state_limit=100000):
    stats=Counter(states=0,center_choices=0,seam_choices=0,matched_emissions=0); exact=[]; deepest={'ledger':[],'rejection':None}; words=['']*len(SLOTS)
    center=5
    for ci,cword in enumerate(SLOTS[center][1]):
        chars=normalize_letters(cword)
        for pivot in range(1,len(chars)):
            left,right=chars[:pivot][::-1],chars[pivot:]
            if not left or not right or left[0]!=right[0]: continue
            stats['center_choices']+=1; stats['seam_choices']+=1; words[center]=cword; ledger=[]; pos=0
            # Emit the central equality through the same scheduler used later.
            while pos<min(len(left),len(right)) and left[pos]==right[pos]:
                ledger.extend(({'side':'left','slot':center,'word':cword,'char':left[pos],'action':'open' if not ledger else 'cancel'}, {'side':'right','slot':center,'word':cword,'char':right[pos],'action':'cancel'})); stats['matched_emissions']+=1; pos+=1
            # Expand actual slot indices outward; choose words only when their
            # boundary is exposed, and append a pair only after it matches.
            l,r=center-1,center+1
            while l>=0 and r<len(SLOTS) and stats['states']<state_limit:
                stats['states']+=1; left_options=SLOTS[l][1]; right_options=SLOTS[r][1]; advanced=False
                for lw in left_options:
                    for rw in right_options:
                        la,ra=normalize_letters(lw)[::-1],normalize_letters(rw)
                        if la[0]!=ra[0]:
                            deepest={'ledger':ledger[:],'rejection':{'left_slot':l,'right_slot':r,'left_word':lw,'right_word':rw,'left_char':la[0],'expected':ra[0],'action':'boundary_contradiction'}}; continue
                        advanced=True; words[l],words[r]=lw,rw
                        ledger.extend(({'side':'left','slot':l,'word':lw,'char':la[0],'action':'open'}, {'side':'right','slot':r,'word':rw,'char':ra[0],'action':'cancel'})); stats['matched_emissions']+=1
                        break
                    if advanced: break
                if not advanced: break
                l-=1; r+=1
            row=audit(render(tuple(words)),'centerout_noon_chart_control',tuple(words)) if all(words) else None
            if row and row['independent_exact_audit']['exact']:
                row['replay']=replay(ledger)
                if row['replay']['ok'] and row['mechanically_admitted']: exact.append(row)
            deepest['independent_replay']=replay(ledger); deepest['emissions_including_rejection']=len(ledger)+(bool(deepest.get('rejection')))
            words=['']*len(SLOTS)
    return {'status':'centerout_noon_event_chart','repair_operator':'equal_internal_seam_then_lexical_boundary_chart','config':{'event_fixed_before_search':True,'starts_at_center':True,'equal_center_required':True,'expands_slot_indices_outward':True,'multiple_legal_seams':True,'production_choices_boundary_exposed':True,'constructed_suffix':False,'independent_reparse':True},'event_graph':SLOTS,'seed_control':audit('The patient guard rests at noon before dusk.','authored_noon_event_control',('the','patient','guard','rests','at','noon','before','dusk')),'stats':dict(stats),'deepest_live_frontier':deepest,'exact_closures':exact,'admitted_closures':exact,'provenance':{'generator_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'material':'authored noon event graph; no catalogue text'},'reader_facing_next_operator':'Replace the complete event graph with a grammatical event offering a longer equal center seam; do not alter a single suffix token.','reader_status':'unreviewed; no programmatic result certifies readability'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--state-limit',type=int,default=100000); a=p.parse_args()
    if a.out.exists(): p.error(f'refusing to overwrite {a.out}')
    r=run(state_limit=a.state_limit); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({'out':str(a.out),'states':r['stats']['states'],'exact':len(r['exact_closures'])},indent=2))
if __name__=='__main__': main()
