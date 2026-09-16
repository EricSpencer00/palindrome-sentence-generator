import json
from pathlib import Path
def test_repair_rejects_fragments():
 p=json.loads((Path(__file__).parents[1]/'runs/reversible-constituent-clause-repair-20260916.json').read_text())
 assert p['exact_count']==0 and p['complete_sentence_count']==0 and p['reader_eligible_count']==0
