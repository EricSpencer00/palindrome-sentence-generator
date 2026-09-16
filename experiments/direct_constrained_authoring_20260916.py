"""Direct local-model authoring probe; every proposal is independently audited."""
import hashlib, json, re, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/direct-constrained-authoring-20260916.json'
PROMPTS=[
"Write one original, grammatical English sentence of at least 100 letters whose letters, ignoring spaces and punctuation and case, form an exact palindrome. Do not repeat clauses or words, do not quote known examples, and use ordinary concrete prose. Output only the sentence.",
"Invent a single coherent English sentence (at least 100 letters) that is an exact character palindrome after lowercasing and removing nonletters. It must have normal subject-verb-object syntax, distinct words, and no list, quotation, or mirrored word order. Output only the sentence.",
"Author fresh readable prose of 100+ letters that is a letter-level palindrome. Keep one scene and natural English syntax; no repeated/self-palindromic chunks, no copied catalogue text, no fragments. Output only the sentence."
]
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
    t=letters(s); words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",s)
    return {'text':s,'letters':len(t),'exact':bool(t) and t==t[::-1],
            'no_repeated_units':len([w.lower().replace("'",'') for w in words])==len(set(w.lower().replace("'",'') for w in words)),
            'complete_sentence':bool(re.search(r'[.!?]$',s.strip())) and len(words)>=8,
            'borrowed_catalogue':False,'reader_eligible':False}
def ask(prompt):
    p=subprocess.run(['ollama','run','gpt-oss:20b',prompt],capture_output=True,text=True,timeout=20)
    return p.stdout.strip().splitlines()[0].strip() if p.stdout.strip() else ''
def main():
    rows=[]
    model_hash=hashlib.sha256(subprocess.check_output(['ollama','show','gpt-oss:20b','--modelfile'])).hexdigest()
    for i,prompt in enumerate(PROMPTS):
        try: text=ask(prompt); err=None
        except Exception as e: text=''; err=type(e).__name__+': '+str(e)
        rows.append({'id':f'direct-{i+1}','prompt':prompt,'model':'gpt-oss:20b','model_hash':model_hash,'provenance':'local-ollama-direct-authoring','attempt':audit(text) if text else None,'error':err})
    # Concrete repair: ask the model to repair the first mismatch while preserving its scene.
    seed=next((r['attempt']['text'] for r in rows if r['attempt'] and r['attempt']['text']), '')
    repair_prompt=("Rewrite this sentence as one original coherent English sentence of 100+ letters, "
                   "making its letters (case/punctuation ignored) an exact palindrome. Do not repeat words or clauses. Output only sentence.\n"+seed)
    try: repaired=ask(repair_prompt); err=None
    except Exception as e: repaired=''; err=type(e).__name__+': '+str(e)
    rows.append({'id':'direct-repair-1','prompt':repair_prompt,'model':'gpt-oss:20b','model_hash':model_hash,'provenance':'local-ollama-direct-authoring|mismatch-directed-repair','attempt':audit(repaired) if repaired else None,'error':err})
    out={'experiment':'direct-constrained-authoring-20260916','signature':'local-model-whole-sentence-authoring|single-scene-contract|live-palindrome-instruction|independent-letter-audit|mismatch-directed-repair','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'exact_count':sum(bool(r['attempt'] and r['attempt']['exact']) for r in rows),'repair_action':'direct model rewrite of first nonempty proposal with exact-tape constraint'}
    OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'candidates':len(rows),'exact_count':out['exact_count'],'output':str(OUT)}))
if __name__=='__main__': main()
