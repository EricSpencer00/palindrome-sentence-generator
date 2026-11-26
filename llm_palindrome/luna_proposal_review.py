"""Validate native Luna proposal receipts, without invoking any model."""
import hashlib,json,math
from pathlib import Path
from .block_seams import Seam,Piece
from .block_proposals import Proposal
from .admission import normalize_letters


def decode_response(payload,request):
    if set(payload)!={'request_id','proposals'}:raise ValueError('unexpected response fields')
    if payload.get('request_id')!=request['request_id']:raise ValueError('request ID mismatch')
    rows=payload.get('proposals')
    if not isinstance(rows,list) or not 1<=len(rows)<=5:raise ValueError('supply one to five proposals')
    out=[];ids=set()
    for row in rows:
        if set(row)!={'side','decoded_text','block_id','score','provenance'}:raise ValueError('unexpected proposal fields')
        if row['side'] not in request['admissible_sides']:raise ValueError('proposal side not allowed')
        if any(not isinstance(row[k],str) or not row[k].strip() for k in ('decoded_text','block_id','provenance')):
            raise ValueError('nonempty proposal text/identity/provenance required')
        if row['block_id'] in ids:raise ValueError('duplicate proposal ID')
        ids.add(row['block_id'])
        if type(row['score']) not in (int,float) or not math.isfinite(row['score']):raise ValueError('finite machine rank score required')
        out.append(Proposal(**row))
    return out


def review_response(request,payload,grammar_accept=None):
    proposals=decode_response(payload,request)
    left=request['left_surface'];right=request['right_surface']
    st=Seam((Piece('fixed-left',0,left),) if left else (), (Piece('fixed-right',0,right),) if right else ())
    if st.debt()!=request['seam_debt']:raise ValueError('request surfaces/debt mismatch')
    rows=[]
    for p in proposals:
        row=dict(block_id=p.block_id,side=p.side,decoded_text=p.decoded_text,
                 text_sha256=hashlib.sha256(p.decoded_text.encode()).hexdigest(),score=p.score,provenance=p.provenance,
                 evidence_kind='machine_proposal',label='native Luna proposal reviewed by checker')
        try:child=st.add(p.side,Piece(p.block_id,0,p.decoded_text));tape=normalize_letters(p.decoded_text)
        except ValueError:child=None;tape=None
        row['normalized_proposal']=tape
        if child is None or not tape:
            row.update(status='rejected',reason='unsupported alphabet, empty tape, or irreversible outer-letter mismatch')
        elif request.get('forbidden_left_proposals') and p.side=='left' and tape in request['forbidden_left_proposals']:
            row.update(status='rejected',reason='known-control restoration explicitly excluded from this request')
        else:
            row.update(new_debt=child.debt(),global_reversal_exact=child.exact(),grammar_verified=False,coherence_verified=False)
            if grammar_accept is None:row.update(status='letter_feasible_grammar_pending',reason='No syntax reviewer supplied; not a grammatical acceptance')
            elif not grammar_accept(st,p):row.update(status='rejected',reason='unlicensed grammar continuation')
            else:row.update(status='letter_and_grammar_accepted',reason='Independent grammar callback passed; coherence remains pending',grammar_verified=True)
        rows.append(row)
    return dict(request_id=request['request_id'],request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
                raw_response=payload,reviewed_proposals=rows,human_feedback=None,model_invoked_by_checker=False)


def save_review(path,request,payload,grammar_accept=None):
    result=review_response(request,payload,grammar_accept)
    path=Path(path)
    if path.exists():raise FileExistsError('preserve previous receipt; choose a new version')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    return result
