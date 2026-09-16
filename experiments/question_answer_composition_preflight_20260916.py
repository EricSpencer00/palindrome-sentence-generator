"""Preflight record for a proposed scalable question/answer composer."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={'experiment':'question-answer-composition-preflight-20260916','status':'preflight_blocked','proposed_signature':'discourse-question-answer-realization|ordinary-prose-composition|live-character-debt|scalable-clause-composition|independent-exact-audit','overlaps':['dialogue-speech-act-residual-20260916','dialogue-acknowledgment-residual-inventory','dialogue-elliptical-ack-residual-inventory','interrogative-quantifier-fsm'],'reason':'The registry already contains question/answer speech-act grammar, acknowledgment and elliptical-answer inventories, and interrogative dependency state. A new composer would replay these dimensions.','pivot':'No candidates were generated or counted; preserve this block and choose a linguistic state not represented by the dialogue/question families.'}
 (ROOT/'runs/question-answer-composition-preflight-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p))
if __name__=='__main__': main()
