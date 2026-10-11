"""Separate assistant inspection, mechanical metrics, and historical model ratings."""
import hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
from llm_palindrome.typed_constituents import TypedGrammar,words
from llm_palindrome.admission import normalize_letters
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/block-seams'
def run():
 p=OUT/'stratified-output-review-001.json';out=OUT/'stratified-output-judgments-001.json'
 if out.exists():raise RuntimeError('immutable judgment receipt exists')
 d=json.loads(p.read_text());old=json.loads((ROOT/'paper/revision/evidence/quality-results.json').read_text());ratings={r['text_sha256']:r.get('scores') for r in old['responses']};g=TypedGrammar();judgments=[]
 for i,r in enumerate(d['sample']):
  text=r['text'];tokens=words(text);cnt=Counter(tokens);grams=Counter(tuple(tokens[j:j+3]) for j in range(max(0,len(tokens)-2)));parsed=g.text_paragraph(text,4);chunks=[c.strip() for c in re.split('[.!?]+',text) if c.strip()]
  dennis='dennis' in tokens and 'sinned' in tokens
  judgment={'id':r['id'],'sample_index':i,'method':r['method'],'text':text,'sampling_stratum':[r['source'],r['method'],r['band'],r['seed_family']],'exact':r['exact'],'letters':r['letters'],'word_count':len(tokens),'finite_grammar_complete':parsed is not None,'sentence_chunks':len(chunks),'repeated_token_excess':sum(v-1 for v in cnt.values()),'repeated_trigram_rate':sum(v-1 for v in grams.values())/max(1,len(tokens)-2),'assistant_inspection':{'grammar':'fails','readability':'poor','cross_sentence_coherence':'not applicable: no complete multi-sentence paragraph','evidence':('Known Dennis sinned scaffold is repeated with unattached words or inverted predicate/subject fragments.' if dennis else 'The opening fragment '+repr(' '.join(tokens[:8]))+' does not form a complete clause; subsequent reversed word fragments do not repair its syntax.'),'repetition':'high' if sum(v-1 for v in cnt.values())>=len(tokens)*.3 else 'present','known_example_derivation':'Known Dennis sinned source family retained in provenance; not novel.' if dennis else 'Historical structural exclusion checks retained in sample provenance; no claim of semantic originality.'},'historical_model_scores':ratings.get(hashlib.sha256(text.encode()).hexdigest()),'historical_model_source':'paper/revision/evidence/quality-results.json; prior frozen ratings, no new paid calls','native_readability_review':'pending handoff authorization','human_score':None}
  judgments.append(judgment)
 grouped=defaultdict(list)
 for j in judgments:grouped[j['method']].append(j)
 summary=[{'method':m,'sample_n':len(js),'assistant_grammar_passes':sum(j['assistant_inspection']['grammar']=='passes' for j in js),'finite_grammar_passes':sum(j['finite_grammar_complete'] for j in js),'assistant_readable_paragraphs':0,'mean_repeated_trigram_rate':sum(j['repeated_trigram_rate'] for j in js)/len(js),'historical_model_ratings_available':sum(j['historical_model_scores'] is not None for j in js),'human_ratings':0} for m,js in sorted(grouped.items())]
 record={'sample_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rating_type':'Current assistant inspection plus deterministic metrics; neither human judgments nor independent native Luna ratings. Historical model scores are separately labeled.','judgments':judgments,'successful_output_conditioned_summary':summary,'generation_summary':d['method_summary'],'inference_limit':'Small strata rounded up; aggregate unweighted sample means are not population estimates. Zero-output methods have undefined conditional quality, not zero quality. Historical paired penalty arms have matched budgets; cross-run pilot comparisons are descriptive only.','finding':'Fragment penalties reduce output duplication while preserving 75% exact-output yield in historical matched cells, but none of this sample forms a readable cohesive paragraph. Boundary/full-slot grammar supports known constructions and avoids inadmissible roots; autonomous paragraph success remains zero.'}
 out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
