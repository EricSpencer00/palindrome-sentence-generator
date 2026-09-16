"""Preflight block for corpus weighted reverse segmentation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={'experiment':'b3-corpus-weighted-reverse-preflight-20260916','status':'preflight_blocked','proposed_signature':'brown-author-corpus-half-sentences|weighted-wordfreq-bigram-dp|reverse-character-segmentation|grammar-clause-filter|independent-exact-audit','overlaps':['brown-attested-residual-lattice-20260916','corpus-span-boundary-dp-20260916','rank-partitioned-corpus-sentence-gram-phrase-lattice','wordfreq-bigram-centerout','attested-phrase-pair-wrapper'],'reason':'The registry already contains Brown/corpus span mining, reverse segmentation, wordfreq-bigram DP, and grammar filtering. Changing weights does not change the generated construction state.','pivot':'No corpus search run; preserve this preflight and require a new linguistic state rather than another weighted reverse decoder.'}
 (ROOT/'runs/b3-corpus-weighted-reverse-preflight-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p))
if __name__=='__main__': main()
