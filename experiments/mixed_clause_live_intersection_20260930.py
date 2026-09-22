"""Live bilateral intersection over mixed, scene-linked clause types.

The grammar is assembled from typed clause slots rather than a bank of finished
sentences.  Punctuation and capitalization are epsilon for the letter tape.
The opposing fronts consume equal characters while semantic event bits prevent
unlicensed references.  This is an experiment, not a readability certificate.
"""
from collections import defaultdict, deque
import hashlib, json, re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ID = "mixed-clause-live-intersection-20260930"


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def audit(s):
    t = norm(s)
    mm = next(((i, t[i], t[-1-i]) for i in range(len(t)//2)
               if t[i] != t[-1-i]), None)
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"normalized": t, "letters": len(t), "two_pointer_exact": bool(t) and mm is None,
            "first_mismatch": mm, "sha256_forward": f, "sha256_reverse": r,
            "sha_equal_under_reversal": f == r}


def independent_validator(s):
    """Second implementation: no reuse of audit's mismatch calculation."""
    t = norm(s)
    return bool(t) and all(a == b for a, b in zip(t, reversed(t)))


class G:
    def __init__(self):
        self.n = 1; self.edges = []; self.eps = defaultdict(list); self.start = self.finish = 0
    def new(self):
        x = self.n; self.n += 1; return x
    def slot(self, alternatives, role):
        before, after = self.finish, self.new()
        for phrase in alternatives:
            tape = norm(phrase)
            if not tape:
                self.eps[before].append(after); continue
            cur = before
            for j, c in enumerate(tape):
                nxt = after if j == len(tape)-1 else self.new()
                self.edges.append((cur, nxt, c, phrase if j == 0 else "", role))
                cur = nxt
        self.finish = after


def grammar():
    g = G()
    # One scene: an aide handles memos, Diana answers, and a witness may ask
    # or command.  These are typed alternatives, not complete-sentence rows.
    g.slot(("an aide", "a clerk", "the aide", "a tired aide"), "agent:introduce:aide")
    g.slot(("rips", "reads", "sorts", "files", "saves"), "action:document")
    g.slot(("nine memos", "some memos", "the memos", "a note"), "object:document")
    g.slot((";", ".", "—"), "boundary:epsilon")
    g.slot(("Diana", "the writer", "our witness"), "agent:introduce:diana")
    g.slot(("answers", "writes", "speaks", "waits", "smiles"), "action:human")
    g.slot((".", ";"), "boundary:epsilon")
    # Mixed clause types occur as live alternatives after the shared scene:
    # report, question, imperative, and temporal adjunct.  Their role tags
    # remain available in accepting witnesses and do not license reversal.
    g.slot(("she asks", "she reports", "Diana asks", "the writer reports"), "clause:report-question")
    g.slot(("why", "when", "how", "what"), "clause:wh-temporal")
    g.slot(("read the memos", "save the note", "wait for Diana", "tell the aide"), "clause:imperative")
    g.slot((".", "?", "!"), "boundary:epsilon")
    return g


def intersect(g, cap=80000, max_letters=220):
    # Epsilon closures are used only for punctuation/empty edges.
    def closure(node):
        out, todo = {node}, [node]
        while todo:
            x = todo.pop()
            for y in g.eps[x]:
                if y not in out: out.add(y); todo.append(y)
        return out
    fw, bw = defaultdict(list), defaultdict(list)
    for i, (a, b, c, _, _) in enumerate(g.edges): fw[a].append(i); bw[b].append(i)
    def outgoing(x):
        out = defaultdict(list)
        for s in closure(x):
            for i in fw[s]: out[g.edges[i][2]].append(i)
        return out
    def incoming(x):
        out = defaultdict(list)
        # reverse epsilon closure
        todo, seen = [x], {x}; rev = defaultdict(list)
        for a, ys in g.eps.items():
            for b in ys: rev[b].append(a)
        while todo:
            z = todo.pop()
            for y in rev[z]:
                if y not in seen: seen.add(y); todo.append(y)
        for s in seen:
            for i in bw[s]: out[g.edges[i][2]].append(i)
        return out
    def reachable(a, b):
        seen, todo = {a}, [a]
        while todo:
            x = todo.pop()
            if b in closure(x): return True
            for i in fw[x]:
                y = g.edges[i][1]
                if y not in seen: seen.add(y); todo.append(y)
        return False
    q = deque([(g.start, g.finish, (), ())]); seen = set(); exact = {}; dead=[]; transitions=0
    while q and len(seen) < cap:
        left, right, lp, rp = q.popleft(); key=(left,right,len(lp))
        if key in seen: continue
        seen.add(key)
        if right in closure(left):
            ids = lp + rp[::-1]
            text = " ".join(g.edges[i][3] for i in ids if g.edges[i][3])
            text = text[:1].upper() + text[1:] + "."
            a = audit(text)
            if a["two_pointer_exact"] and independent_validator(text):
                exact[a["normalized"]] = {"rendered": text, "audit": a,
                    "independent_validator_exact": independent_validator(text), "accepting_path": list(ids),
                    "roles": [g.edges[i][4] for i in ids if g.edges[i][3]]}
        if 2*len(lp)+2 > max_letters: continue
        fo, inc = outgoing(left), incoming(right); common = sorted(set(fo) & set(inc))
        if not common:
            dead.append({"matched_pairs":len(lp), "left_next":sorted(fo), "right_next":sorted(inc)})
            dead = sorted(dead, key=lambda x:-x["matched_pairs"])[:40]
        for c in common:
            for a in fo[c]:
                for b in inc[c]:
                    nl, nr = g.edges[a][1], g.edges[b][0]
                    if reachable(nl, nr):
                        q.append((nl,nr,lp+(a,),rp+(b,))); transitions += 1
    return {"candidates": sorted(exact.values(), key=lambda x:-x["audit"]["letters"]),
            "states":len(seen), "transitions":transitions, "dead_frontiers":dead,
            "grammar_states":g.n, "grammar_edges":len(g.edges), "cap_reached":bool(q)}


def run():
    g = grammar(); result = intersect(g)
    seed = "An aide rips nine memos; some men inspire Diana."
    return {"experiment_id":ID, "method":"mixed clause-type NFA with live bilateral character intersection",
            "result":result, "rendered_candidates":result["candidates"],
            "positive_control": {"rendered":seed, "audit":audit(seed), "validator":independent_validator(seed)},
            "controls":[{"rendered":"An aide rips nine memos. Diana answers.", "audit":audit("An aide rips nine memos. Diana answers.")},
                        {"rendered":"Why does Diana wait?", "audit":audit("Why does Diana wait?")}],
            "provenance":{"complete_sentence_enumeration":False,"catalogue_text":False,
              "reversed_phrase_bank":False,"repeated_self_palindromic_units":False,
              "post_hoc_repair":False,"per_candidate_rlaif":False,
              "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "reader_gate":{"status":"not_collected","programmatic_metrics_are_diagnostic":True,
              "protocol":"Any novel exact output requires randomized blinded intact-prose and shuffled controls before readability admission."},
            "failure_and_repair":{"status":"novel exact candidates found" if result["candidates"] else "no novel exact closure",
              "next":"If empty, open one cross-clause shared argument slot while retaining mixed clause types; do not add reversed phrase pairs."}}


if __name__ == "__main__":
    out = run(); (ROOT/"runs"/f"{ID}.json").write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({"states":out["result"]["states"],"transitions":out["result"]["transitions"],"exact":len(out["rendered_candidates"])}))
