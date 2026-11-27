"""Blinded AWS Bedrock quality audit; credentials stay inside the AWS CLI.

Reuses the previous rubric and its exact 12 Brown prose/12 shuffle controls.
Every distinct new candidate is rated once, irrespective of condition. Twelve
hidden repeats check consistency. Corpus surfaces exist only in memory and a
temporary local request file, which is removed after each AWS call.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fragment_penalty_20261002 import OUT, sha, write
from experiments import aws_qwen3_readability_audit_20260926 as rubric
from experiments.local_llm_readability_audit_20260926 import normalized_case
from experiments.static_language_verifier_20260925 import sha256_text, shuffle_surface

AWS = "/Users/eric/.local/bin/aws"
REGION = "us-east-2"
OLD = ROOT / "runs/aws-qwen3-32b-readability-audit-20260926.json"
MANIFEST = OUT / "quality-manifest.json"
RESULT = OUT / "quality-results.json"


def materialize():
    prior = json.loads(OLD.read_text())
    controls = []
    for item in prior["presentations"]:
        if "brown" not in item["condition"] or item.get("repeat_of"):
            continue
        lo, hi = item["brown_sentence_indices_half_open"]
        corpus = Path(os.environ.get('PALINDROME_BROWN_DIR', str(Path.home()/'nltk_data/corpora/brown')))
        lines = (corpus/item["brown_fileid"]).read_text().splitlines()
        sentences = [[w.rsplit('/', 1)[0] for w in line.split()] for line in lines if line.strip()]
        text = " ".join(" ".join(s) for s in sentences[lo:hi])
        if item["condition"] == "heldout_brown_same_word_shuffle":
            text = shuffle_surface(text, item["shuffle_seed"])
        assert sha256_text(text) == item["source_text_sha256"]
        text = normalized_case(text)
        assert sha256_text(text) == item["text_sha256"]
        controls.append({"condition": item["condition"], "text": text,
                         "text_sha256": sha256_text(text), "prior_presentation_id": item["presentation_id"]})
    search = json.loads((OUT / "results.json").read_text())
    assert search["status"] == "complete" and len(search["results"]) == 360
    candidates = {}
    for row in search["results"]:
        if row.get("text"):
            assert sha256_text(row["text"]) == row["text_sha256"]
            candidates[row["text_sha256"]] = {"condition": "candidate", "text": row["text"], "text_sha256": row["text_sha256"]}
    candidate_rows = [candidates[k] for k in sorted(candidates)]
    prose = sorted([r for r in controls if r["condition"] == "heldout_brown_prose"], key=lambda r: r["text_sha256"])
    repeats = [{**r, "condition": "hidden_repeat", "repeat_source_condition": r["condition"]} for r in candidate_rows[:6]+prose[:6]]
    first = sorted(controls, key=lambda r: sha256_text("control-order:"+r["text_sha256"]))
    rest = sorted(candidate_rows+repeats, key=lambda r: sha256_text("test-order:"+r["condition"]+r["text_sha256"]))
    rows = first+rest
    for i, row in enumerate(rows):
        row["id"] = "V-"+sha256_text(f"20261002:{row['condition']}:{row['text_sha256']}")[:16]
        row["order"] = i+1
    return rows


def prepare():
    rows = materialize()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": rubric.MODEL_ID, "region": REGION,
        "search_manifest_sha256": sha(OUT / "manifest.json"),
        "search_results_sha256": sha(OUT / "results.json"),
        "prior_audit_sha256": sha(OLD),
        "script_sha256": sha(__file__),
        "rubric_source_sha256": sha(ROOT/"experiments/local_llm_readability_audit_20260926.py"),
        "request_source_sha256": sha(ROOT/"experiments/aws_qwen3_readability_audit_20260926.py"),
        "system_prompt": rubric.SYSTEM_PROMPT,
        "user_template": rubric.USER_PROMPT_TEMPLATE,
        "decoding": rubric.DECODING,
        "timing": "search design and rubric frozen before generation; complete presentation inventory frozen after search completion and before any new judge call",
        "controls": "same 12 prose and 12 shuffles as prior audit; pass gate >=8 prose and <=3 shuffles jointly >=2 on both scales; candidates assessed only if gate passes",
        "blinding": "one text only; no arm, method, score, source, paired output, or repeat label in model messages; identical text shared across arms receives the same rating",
        "repeats": "six candidates and six prose controls selected by ascending text hash, repeated once and interspersed with candidate calls; excluded from main counts",
        "meaning": "model evidence, not independent human readability; no provider seed or model-weight digest",
        "counts": dict(Counter(r['condition'] for r in rows)),
        "presentations": [{k:v for k,v in r.items() if k!='text'} for r in rows],
    }
    if MANIFEST.exists(): raise FileExistsError(MANIFEST)
    write(MANIFEST, protocol)
    print(json.dumps({"manifest_sha256":sha(MANIFEST),"counts":protocol['counts']}))


def execute():
    protocol = json.loads(MANIFEST.read_text())
    assert protocol['script_sha256'] == sha(__file__)
    assert protocol['search_results_sha256'] == sha(OUT/'results.json')
    assert protocol['rubric_source_sha256'] == sha(ROOT/'experiments/local_llm_readability_audit_20260926.py')
    assert protocol['request_source_sha256'] == sha(ROOT/'experiments/aws_qwen3_readability_audit_20260926.py')
    rows = materialize()
    assert [{k:v for k,v in r.items() if k!='text'} for r in rows] == protocol['presentations']
    budget = json.loads((OUT/'aws-budget.json').read_text())
    identity = json.loads(subprocess.run([AWS,'sts','get-caller-identity','--output','json'],capture_output=True,text=True,check=True).stdout)
    assert hashlib.sha256(identity['Account'].encode()).hexdigest() == budget['account_id_sha256']
    rates = budget['rates']
    input_rate = rates['qwen_input']['rate']/1000
    output_rate = rates['qwen_output']['rate']/1000
    worst = 32768*input_rate+128*output_rate
    result = json.loads(RESULT.read_text()) if RESULT.exists() else {
        'manifest_sha256':sha(MANIFEST),'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'responses':[], 'failed_attempts':[], 'reserved_cost_usd':0.0, 'in_flight':None}
    assert result['manifest_sha256'] == sha(MANIFEST)
    if result.get('in_flight'):
        raise RuntimeError('unresolved request: inspect before resuming; no blind duplicate call')
    done = {r['id'] for r in result['responses']}
    last_start = 0.0
    for row in rows:
        if row['id'] in done: continue
        if row['order'] > 24:
            lookup = {r['id']:r['scores'] for r in result['responses']}
            passed = {kind:sum(all(lookup[r['id']][k]>=2 for k in ('grammaticality','coherent_meaning')) for r in rows[:24] if r['condition']==kind) for kind in ('heldout_brown_prose','heldout_brown_same_word_shuffle')}
            result['control_pass_counts'] = passed
            result['control_gate_passed'] = passed['heldout_brown_prose']>=8 and passed['heldout_brown_same_word_shuffle']<=3
            if not result['control_gate_passed']:
                result['status']='control_gate_failed';write(RESULT,result);return
        if result['reserved_cost_usd']+worst > budget['bedrock_total_cap_usd']:
            result['status']='cost_cap';write(RESULT,result);return
        body = rubric.request_body(row['text'], row['id'])
        body['requestMetadata']['experiment'] = 'fragment-penalty-20261002'
        request = json.dumps(body,sort_keys=True)
        assert len(request.encode()) < 20000, 'unexpectedly large prompt'
        time.sleep(max(0.0,5.0-(time.monotonic()-last_start)))
        result['in_flight']={'id':row['id'],'request_sha256':sha256_text(request),'reserved_usd':worst}
        result['reserved_cost_usd']+=worst
        write(RESULT,result)
        with tempfile.NamedTemporaryFile(mode='w',suffix='.json',prefix='palindrome-request-',delete=False) as handle:
            handle.write(request); path=handle.name
        try:
            last_start=time.monotonic()
            env={**os.environ,'AWS_MAX_ATTEMPTS':'1'}
            call=subprocess.run([AWS,'bedrock-runtime','converse','--region',REGION,'--cli-input-json','file://'+path,'--output','json','--no-cli-pager','--cli-read-timeout','180'],capture_output=True,text=True,env=env,timeout=200)
        finally:
            Path(path).unlink(missing_ok=True)
        if call.returncode:
            result['failed_attempts'].append({**result['in_flight'],'error':call.stderr[-1500:]})
            result['in_flight']=None;result['status']='request_error';write(RESULT,result)
            raise RuntimeError('Bedrock request failed; details retained without prompt text')
        response=json.loads(call.stdout)
        usage=response['usage']; actual=usage['inputTokens']*input_rate+usage['outputTokens']*output_rate
        assert actual <= worst, 'request exceeded the frozen cost reservation'
        result['reserved_cost_usd']+=actual-worst
        raw=''.join(x.get('text','') for x in response['output']['message']['content'])
        scores=json.loads(raw)
        assert set(scores)=={'grammaticality','coherent_meaning'}
        assert all(type(x) is int and 0<=x<=3 for x in scores.values())
        result['responses'].append({'id':row['id'],'text_sha256':row['text_sha256'],'scores':scores,
            'request_sha256':result['in_flight']['request_sha256'],'response':response,'cost_usd':actual,'completed_utc':datetime.now(timezone.utc).isoformat()})
        result['in_flight']=None
        write(RESULT,result)
        print(json.dumps({'completed':len(result['responses']),'total':len(rows),'cost_usd':round(result['reserved_cost_usd'],5)}),flush=True)
    result['status']='complete';result['completed_utc']=datetime.now(timezone.utc).isoformat();write(RESULT,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
    prepare() if args.action=='prepare' else execute()
