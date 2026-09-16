"""Preflight block for repairing exact rejected tapes into clauses."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={'experiment':'exact-rejected-tape-clause-repair-preflight-20260916','status':'preflight_blocked','proposed_signature':'exact-rejected-tape-inventory|lexical-boundary-repair|finite-state-clause-realization|semantic-role-filter|independent-audit','overlaps':['grammar-boundary-resegmentation-repair','lexical-trie-segmentation-repair-20260916','induced-grammar-reverse-decoder-20260916','luna-constrained-reverse-decode','semantic-frame-slots'],'reason':'The registry already retains exact-tape boundary resegmentation, held-out trie segmentation, induced reverse grammar decoding, and semantic-frame fragment repair. Repairing the 72 rejected tapes by lexical boundaries and clause filtering would replay these states.','pivot':'No tape repair run; preserve the block and require a new construction state rather than another fixed-tape decoder/segmentation repair.'}
 (ROOT/'runs/exact-rejected-tape-clause-repair-preflight-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p))
if __name__=='__main__': main()
