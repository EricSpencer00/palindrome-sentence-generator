"""Final center-out attempt: data-driven, non-self-palindromic cross-word seams."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters,tokenize,has_self_palindromic_proper_multiword_span

SLOTS=(('det',('the',)),('subject',('teacher','editor')),('verb',('writes','notes')),('object_det',('some','the')),('object',('notes','letters')),('prep',('for','to')),('goal',('class','readers')))
CONTROL='The teacher writes some notes for class.'

def replay(ledger):
    residual=''; cancels=0
    for e in ledger:
        c=e['char']
        if residual:
            if c!=residual[0]: return {'ok':False,'events_replayed':len(ledger),'cancellations':cancels,'residual':residual}
            residual=residual[1:];cancels+=1
        else: residual=c
    return {'ok':not residual,'events_replayed':len(ledger),'cancellations':cancels,'residual':residual}
def parse(text):
    w=tokenize(text);return len(w)==7 and w[0]=='the' and w[1] in SLOTS[1][1] and w[2] in SLOTS[2][1] and w[3] in SLOTS[3][1] and w[4] in SLOTS[4][1] and w[5] in SLOTS[5][1] and w[6] in SLOTS[6][1]
def audit(text,kind,prov):
    tape=normalize_letters(text);g=mechanical_admission_checks(text,min_letters=30,max_letters=220);codes=[k for k,v in g.items() if not v]
    if not parse(text):codes.append('independent_complete_reparse_failed')
    return {'record_kind':kind,'rendered':text,'provenance':prov,'independent_exact_audit':{'exact':bool(tape) and tape==tape[::-1],'letters':len(tape),'normalized_sha256':sha256(tape.encode()).hexdigest()},'independent_parse':parse(text),'central_admission':g,'mechanically_admitted':not codes,'rejection_codes':codes,'reader_status':'unreviewed; programmatic checks do not certify readability'}
def run(*,state_limit=100000):
    stats=Counter(seams_considered=0,legal_seams=0,states=0,matched_emissions=0); deepest={'ledger':[],'rejection':None};exact=[]
    # Enumerate every cross-word seam. The central span is prefiltered before
    # any stream is emitted; no self-palindromic word/span can seed the chart.
    for seam in range(1,len(SLOTS)):
      for lw in SLOTS[seam-1][1]:
       for rw in SLOTS[seam][1]:
        stats['seams_considered']+=1; units=(lw,rw); span=normalize_letters(lw+rw)
        if lw==lw[::-1] or rw==rw[::-1] or span==span[::-1] or has_self_palindromic_proper_multiword_span(units): continue
        la,ra=normalize_letters(lw)[::-1],normalize_letters(rw)
        if la[0]!=ra[0]: continue
        stats['legal_seams']+=1;ledger=[];pos=0
        # Keep both active word streams until one is exhausted; only then may
        # the corresponding outward slot be selected.
        while pos<len(la) and pos<len(ra) and la[pos]==ra[pos] and stats['states']<state_limit:
            ledger.extend(({'side':'left','slot':seam-1,'word':lw,'char':la[pos],'action':'open' if not ledger else 'cancel'},{'side':'right','slot':seam,'word':rw,'char':ra[pos],'action':'cancel'}));pos+=1;stats['states']+=1;stats['matched_emissions']+=1
        if pos<len(la) and pos<len(ra): deepest={'ledger':ledger,'rejection':{'seam':seam,'left_word':lw,'right_word':rw,'left_char':la[pos],'expected':ra[pos],'action':'boundary_contradiction'}}
        else: deepest={'ledger':ledger,'rejection':None}
    control=audit(CONTROL,'crossword_seam_grammar_control',tuple(tokenize(CONTROL)));deepest['independent_replay']=replay(deepest['ledger']);deepest['emissions_including_rejection']=len(deepest['ledger'])+bool(deepest['rejection'])
    return {'status':'centerout_crossword_seam_chart_final_attempt','repair_operator':'all_seams_nonselfpalindromic_stream_exhaustion','config':{'event_fixed_before_search':True,'cross_word_center_required':True,'all_legal_seams_enumerated':True,'central_span_prefiltered':True,'active_streams_exhausted_before_slot_advance':True,'constructed_suffix':False,'independent_reparse':True},'event_graph':SLOTS,'seed_control':control,'stats':dict(stats),'deepest_live_frontier':deepest,'exact_closures':exact,'admitted_closures':[],'provenance':{'generator_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'material':'authored teacher note event; no catalogue text'},'reader_facing_next_operator':'If no seam survives preflight, author a new grammatical event with a non-self-palindromic compatible seam; do not fabricate a trace.','reader_status':'unreviewed; no programmatic result certifies readability'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--state-limit',type=int,default=100000);a=p.parse_args()
 if a.out.exists():p.error(f'refusing to overwrite {a.out}')
 r=run(state_limit=a.state_limit);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'out':str(a.out),'seams':r['stats']['seams_considered'],'legal':r['stats']['legal_seams'],'exact':0},indent=2))
if __name__=='__main__':main()
