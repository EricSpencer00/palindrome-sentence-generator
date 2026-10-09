"""Small inspectable newly authored block bank and bounded seam chart."""
import hashlib
import itertools
import json
from pathlib import Path
from llm_palindrome.block_seams import Piece,Seam,paragraph_gates

ROOT=Path(__file__).resolve().parents[1]
BANK={
 'carry-gem':dict(parts=['Tara','carries','the gem.'],entities=['Tara','gem'],event='carry'),
 'weigh-gem':dict(parts=['The gem','weighs','a','carat.'],entities=['gem'],event='weight'),
 'open-case':dict(parts=['Nora','opens','the case.'],entities=['Nora','case'],event='open'),
 'place-gem':dict(parts=['Tara','places','the gem','inside the case.'],entities=['Tara','gem','case'],event='place'),
}
for value in BANK.values():
 value['provenance']='Newly authored finite subject/verb/complement block bank, 2026-10-09; no Diana/corpus/catalogue import'
 value['grammar_evidence']='Explicit declarative parse inventory; independent Luna/human judgments pending'


def next_pieces(state,side):
    seq=state.left if side=='left' else state.right
    used={p.sentence_id for p in state.left+state.right}
    if seq:
        end=seq[-1] if side=='left' else seq[0]
        index=end.index+(1 if side=='left' else -1)
        if 0<=index<len(BANK[end.sentence_id]['parts']):
            return [Piece(end.sentence_id,index,BANK[end.sentence_id]['parts'][index])]
    return [Piece(sid,0 if side=='left' else len(v['parts'])-1,
                  v['parts'][0 if side=='left' else -1]) for sid,v in BANK.items() if sid not in used]


def run(max_states=500):
    endpoints=[]
    for left,right in itertools.product(BANK,BANK):
        if left==right:continue
        state=Seam((Piece(left,0,BANK[left]['parts'][0]),),
                   (Piece(right,len(BANK[right]['parts'])-1,BANK[right]['parts'][-1]),))
        endpoints.append(dict(left=left,right=right,debt=state.debt()))
    queue=[Seam()];seen={Seam()};failed=[];witnesses=[];visited=0
    while queue and visited<max_states:
        state=queue.pop(0);visited+=1
        gates=paragraph_gates(state,BANK)
        # Human coherence is deliberately not a machine admission boolean.
        if all(gates[k] for k in ('exact_palindrome','sentence_grammar_complete',
               'independent_nonpalindromic_blocks','no_duplicated_sentences','discourse_topic_linkage')):
            witnesses.append(dict(text=state.text(),gates=gates))
        if len(state.left)+len(state.right)>=14:continue
        for side in ('left','right'):
            for piece in next_pieces(state,side):
                child=state.add(side,piece)
                if child is None:
                    trial=Seam(state.left+(piece,),state.right) if side=='left' else Seam(state.left,(piece,)+state.right)
                    failed.append(dict(left=[p.__dict__ for p in trial.left],right=[p.__dict__ for p in trial.right],debt=trial.debt()))
                elif child not in seen:seen.add(child);queue.append(child)
    return dict(schema_version=1,bank=BANK,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                status='state_cap' if queue else 'finite_chart_exhausted',visited_states=visited,
                endpoint_pairs=endpoints,compatible_endpoint_pairs=sum(p['debt']['viable'] for p in endpoints),
                failed_joins=failed,constructions=witnesses,human_ratings=None,
                remaining_frontier_states=len(queue),
                scope='Bounded development finite inventory chart; cap exhaustion is not inventory infeasibility or a readability result')


if __name__=='__main__':
    out=ROOT/'research/block-seams/fixtures/block-seams-002.json'
    if out.exists():raise FileExistsError('preserve previous chart result')
    result=run();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','visited_states','compatible_endpoint_pairs','constructions')},indent=2))
