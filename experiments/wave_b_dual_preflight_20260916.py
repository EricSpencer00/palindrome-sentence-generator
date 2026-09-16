"""Preflight record for two requested but already-covered constructive routes."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={'experiment':'wave-b-dual-preflight-20260916','status':'preflight_blocked','routes':{'B1':{'signature':'dependency-completion-boundary-state|live-semantic-scoring|character-obligation','overlaps':['dependency-completion-csp-20260916','bilateral-slot-synchronizer-adjunct-repair-20260916','variable-boundary-tape-ilp']},'B2':{'signature':'morphology-inflectional-seam-growth|arbitrary-target-length|semantic-role-preservation|character-obligation','overlaps':['morphology-semantic-template-csp-20260916','inflectional-fst-clitic-tape-20260916','orthographic-derivational-composition']}},'reason':'Both requested state spaces are already retained: B1 is dependency completion plus boundary-state search; B2 is morphology/inflectional seam growth. Running them would duplicate prior artifacts.','pivot':'No candidate searches run or counted; preserve this dual preflight and require genuinely new linguistic states.'}
 (ROOT/'runs/wave-b-dual-preflight-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p))
if __name__=='__main__': main()
