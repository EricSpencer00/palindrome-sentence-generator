"""Indexed typed phrase-graph walks for readable palindrome construction.

Unlike clause-product searches, this lane composes a scene as a walk through
an authored graph.  Edges carry syntactic/semantic obligations; a reverse
walk is admitted only when its character tape is present in the forward walk
index.  No language-model reward is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, subprocess
from pathlib import Path
from collections import defaultdict
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).resolve().parents[1]
ID = "typed-phrase-graph-walk-20260919"

@dataclass(frozen=True)
class Edge:
    src: str; dst: str; text: str; kind: str; number: str | None = None
    valency: str | None = None

EDGES = (
    Edge("open", "agent_sg", "an aide", "subject", "sg"),
    Edge("open", "agent_sg", "a poet", "subject", "sg"),
    Edge("open", "agent_pl", "some men", "subject", "pl"),
    Edge("open", "agent_pl", "the sailors", "subject", "pl"),
    Edge("agent_sg", "action_sg", "rips", "verb", "sg", "transitive"),
    Edge("agent_sg", "action_sg", "inspires", "verb", "sg", "transitive"),
    Edge("agent_pl", "action_pl", "inspire", "verb", "pl", "transitive"),
    Edge("agent_pl", "action_pl", "guard", "verb", "pl", "transitive"),
    Edge("action_sg", "object", "nine memos", "object"),
    Edge("action_sg", "object", "a sonnet", "object"),
    Edge("action_pl", "object", "Diana", "object"),
    Edge("action_pl", "object", "new songs", "object"),
    Edge("object", "close", "at dawn", "adjunct"),
    Edge("object", "close", "near the river", "adjunct"),
    # A second beat allows the graph to make a coherent scene without
    # reversing or copying a finished sentence.
    Edge("close", "agent_sg", "a keeper", "subject", "sg"),
    Edge("close", "agent_pl", "some sailors", "subject", "pl"),
)

def tape(s: str) -> str: return normalize_letters(s)
def words(s: str) -> frozenset[str]: return frozenset(tape(x) for x in tokenize(s) if tape(x) not in {"a","an","the","at","near"})

def walks(max_edges=6):
    out=[]
    def dfs(node, edges, used):
        if node == "close" and len(edges) >= 3:
            out.append(tuple(edges))
        if len(edges) == max_edges: return
        for e in EDGES:
            if e.src != node: continue
            if e.text in used: continue
            # Keep agreement/valency local: verbs are only reached from the
            # matching subject state, and objects only follow transitive verbs.
            if e.kind == "verb" and not edges: continue
            dfs(e.dst, edges + [e], used | {e.text})
    dfs("open", [], set())
    return out

def render(path): return " ".join(e.text for e in path) + "."
def audit(text):
    t=tape(text); ok=all(t[i]==t[-1-i] for i in range(len(t)//2))
    f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"normalized":t,"letters":len(t),"two_pointer_exact":ok,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

def run():
    paths=walks()
    by_tape=defaultdict(list)
    for p in paths: by_tape[tape(render(p))].append(p)
    exact=[]
    for p in paths:
        text=render(p); t=tape(text); rev=by_tape.get(t[::-1], [])
        if rev:
            a=audit(text); admission=mechanical_admission_checks(text,min_letters=30,max_letters=240)
            exact.append({"rendered":text,"length":a["letters"],"provenance":"typed authored phrase-graph walk; indexed reverse tape","audit":a,"mechanical_admission":admission,"graph_edges":[e.__dict__ for e in p],"reader_status":"unreviewed"})
    exact.sort(key=lambda x:x["length"], reverse=True)
    registry=ROOT/"docs/experiment-novelty-registry.json"
    payload={"experiment_id":ID,"method":"typed phrase graph walk with reverse-tape index; bounded DFS, no reward model","stats":{"walks":len(paths),"indexed_tapes":len(by_tape),"exact_rows":len(exact),"longest_exact":exact[0]["length"] if exact else 0},"actual_candidates":exact[:12],"novelty_preflight":{"fresh_graph":True,"catalogue_imported":False,"finished_tape_reversal_used":False,"prior_lane_overlap":"graph composition is new; audit primitive shared"},"next_repair":"Add a second independently authored transitive beat whose edge boundary supplies the first unresolved reverse character, then extend indexed walks to 8 edges; retain role agreement and unique content words.","provenance":{"source":str(Path(__file__).relative_to(ROOT)),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"registry_sha256":hashlib.sha256(registry.read_bytes()).hexdigest(),"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()},"strict_gate":{"readable_over_38":0,"human_readability_test":"not performed","note":"programmatic metrics do not certify readability"}}
    return payload

if __name__ == "__main__":
    out=ROOT/"runs"/(ID+".json"); out.write_text(json.dumps(run(),indent=2)+"\n"); print(json.dumps(run()["stats"],indent=2))
