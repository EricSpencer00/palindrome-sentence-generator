"""Variable clause-sequence grammar with live mirrored character propagation.

This is a topology pivot from the fixed-frame REGULAR lane: accepted strings
contain one, two, or three independently authored clauses.  Clause boundaries
and coordination are grammar edges, not a post-render splice.  Mirrored
character domains are propagated while the unfinished NFA path is still being
searched; no completed tape is reversed and no repair is applied.
"""
from __future__ import annotations
import hashlib, json, re, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "variable-clause-sequence-20260921"
SEED = "An aide rips nine memos; some men inspire Diana."

def letters(s): return re.sub("[^a-z]", "", s.lower())

def audit(text):
    a = letters(text); b = ''.join(c.lower() for c in text if c.isalpha() and c.isascii())
    i,j=0,len(b)-1
    while i<j and b[i]==b[j]: i,j=i+1,j-1
    return {"letters":len(a),"normalized":a,"exact":bool(a) and a==a[::-1],
            "independent_pointer_exact":bool(b) and i>=j,
            "normalizers_agree":a==b,"sha256":hashlib.sha256(a.encode()).hexdigest(),
            "reverse_sha256":hashlib.sha256(a[::-1].encode()).hexdigest()}

class NFA:
    def __init__(self): self.edges=defaultdict(list); self.size=2; self.frames=[]
    def state(self): n=self.size; self.size+=1; return n
    def path(self, name, tokens):
        cur=0; path=[]
        for word,role in tokens:
            nxt=self.state(); chars=letters(word); st=cur
            for k,ch in enumerate(chars):
                dst=nxt if k==len(chars)-1 else self.state()
                self.edges[st].append((ch,dst, {'word':word,'role':role,'frame':name} if k==len(chars)-1 else None)); st=dst
            cur=nxt; path.append((word,role))
        self.edges[cur].append(("",1,None)); self.frames.append({"name":name,"tokens":path})
    def compile(self):
        closure={}
        for s in range(self.size):
            seen={s}; todo=[s]
            while todo:
                for ch,d,_ in self.edges[todo.pop()]:
                    if not ch and d not in seen: seen.add(d); todo.append(d)
            closure[s]=seen
        tr=defaultdict(list)
        for s in range(self.size):
            for q in closure[s]:
                tr[s].extend((c,d) for c,d,_ in self.edges[q] if c)
        return tr,{s for s in range(self.size) if 1 in closure[s]}
    def render(self,tape):
        q=[(0,0)]; parent={(0,0):None}
        while q:
            s,p=q.pop()
            if s==1 and p==len(tape):
                out=[]; key=(s,p)
                while parent[key] is not None:
                    prev,label=parent[key]
                    if label: out.append(label)
                    key=prev
                out.reverse(); return out
            for ch,d,label in self.edges[s]:
                if ch and (p>=len(tape) or tape[p]!=ch): continue
                key=(d,p+bool(ch))
                if key not in parent: parent[key]=((s,p),label); q.append(key)
        return None

def clauses():
    # Each clause is a complete transitive proposition with number agreement.
    sg=[("a", "det_sg"),("poet", "subject_sg"),("reads", "verb_sg"),("a", "obj_det_sg"),("map", "object_sg"),
        ("the", "det_sg"),("artist", "subject_sg"),("marks", "verb_sg"),("the", "obj_det_sg"),("letter", "object_sg"),
        ("a", "det_sg"),("writer", "subject_sg"),("keeps", "verb_sg"),("a", "obj_det_sg"),("book", "object_sg")]
    pl=[("some", "det_pl"),("men", "subject_pl"),("read", "verb_pl"),("the", "obj_det_pl"),("maps", "object_pl"),
        ("many", "det_pl"),("poets", "subject_pl"),("mark", "verb_pl"),("the", "obj_det_pl"),("letters", "object_pl")]
    return [sg[i:i+5] for i in range(0,len(sg),5)] + [pl[i:i+5] for i in range(0,len(pl),5)]

def build():
    g=NFA(); base=clauses(); connectors=[("and","coord"),("while","coord"),("as","coord")]
    # Explicit sequence topology: 1, 2, or 3 distinct complete clauses.
    # Connector choice is part of the live path and preserves clause edges.
    from itertools import permutations, product
    for count in (1,2,3):
        for ids in permutations(range(len(base)), count):
            for conns in product(connectors, repeat=count-1):
                toks=[]
                for k,idx in enumerate(ids):
                    if k: toks.append(conns[k-1])
                    toks.extend(base[idx])
                g.path(f"sequence_{count}_{ids}_{tuple(c[0] for c in conns)}", toks)
    return g

def propagate(dom, tr, finals, stats):
    dom=list(dom); n=len(dom)
    while True:
        stats["propagation_rounds"]+=1; fw=[{0}]
        for d in dom: fw.append({q for s in fw[-1] for c,q in tr[s] if c in d})
        if not fw[n]&finals: return None
        bw=[set() for _ in range(n+1)]; bw[n]=fw[n]&finals; sup=[set() for _ in range(n)]
        for i in range(n-1,-1,-1):
            for s in fw[i]:
                for c,q in tr[s]:
                    if c in dom[i] and q in bw[i+1]: bw[i].add(s); sup[i].add(c)
        changed=False
        for i in range((n+1)//2):
            j=n-1-i; keep=frozenset(sup[i]&sup[j])
            if not keep: return None
            if keep!=dom[i] or keep!=dom[j]:
                stats["domain_values_removed"]+=len(dom[i]-keep)+(len(dom[j]-keep) if i!=j else 0)
                dom[i]=dom[j]=keep; changed=True
        if not changed: return tuple(dom)

def solve(g,n,cap=250):
    tr,finals=g.compile(); st={"nodes":0,"propagation_rounds":0,"domain_values_removed":0,"conflicts":0,"cap_reached":False}; out=[]; frontier=[]
    def visit(dom):
        if st["nodes"]>=cap: st["cap_reached"]=True; return
        st["nodes"]+=1; dom=propagate(dom,tr,finals,st)
        if dom is None: st["conflicts"]+=1; return
        openp=[i for i in range((n+1)//2) if len(dom[i])>1]
        if not openp:
            tape=''.join(next(iter(x)) for x in dom); path=g.render(tape)
            if path:
                text=' '.join(w for w,_ in path).capitalize()+'.'; toks=re.findall('[a-z]+',text.lower())
                flags=[]
                if len(toks)!=len(set(toks)): flags.append('repeated_word')
                out.append({"text":text,"audit":audit(text),"lexical_path":path,"shortcut_flags":flags,"human_readability":"not_tested"})
            return
        # Preserve a real frontier rather than silently discarding unresolved parses.
        if len(frontier)<12:
            frontier.append({"prefix_domains":[ ''.join(sorted(x)) for x in dom[:min(8,n)] ],"open_positions":len(openp)})
        p=min(openp,key=lambda i:(len(dom[i]),abs(n/2-i)))
        for c in sorted(dom[p]):
            b=list(dom); b[p]=b[n-1-p]=frozenset(c); visit(tuple(b))
    visit(tuple(frozenset('abcdefghijklmnopqrstuvwxyz') for _ in range(n)))
    return {"target_letters":n,"stats":st,"candidates":out,"frontier":frontier}

def main():
    t=time.monotonic(); g=build(); results=[]
    # Actual grammar lengths are finite; target sweep includes every requested
    # length >=39 and records root conflicts/frontiers rather than fabricating output.
    for n in range(39,101): results.append(solve(g,n))
    controls=[]
    for frame in g.frames[:6]:
        text=' '.join(w for w,_ in frame['tokens']).capitalize()+'.'
        controls.append({'text':text,'audit':audit(text),
                         'provenance':'independently authored variable-sequence grammar control',
                         'status':'intact_grammar_control_not_palindrome_candidate'})
    out={"experiment_id":ID,"method":"variable_clause_sequence_NFA_live_mirrored_domains",
         "results":results,"forward_controls":controls,"seconds":time.monotonic()-t,"grammar_frames":len(g.frames),
         "provenance":{"lexicon":"hand_authored_complete_transitive_clauses","catalogue_text":False,
                         "finished_tape_reverse_for_construction":False,"repair":False,
                         "variable_clause_counts":[1,2,3],"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
         "exact_audit":"independent letter normalizer + pointer + SHA-256",
         "novelty":{"implementation_distinction":"variable clause-count topology with live connector and clause-boundary NFA edges",
                    "prior_checked":["regular-shared-character-20260921","recursive-clause-pair-constructor-20260916"]},
         "next_operator":"add a live center nonterminal connecting two independently licensed clause sequences; do not add another modifier bank",
         "reader_gate":"closed until exact original plausible prose and blinded ratings"}
    (ROOT/'runs'/f'{ID}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({"frames":len(g.frames),"targets":len(results),"exact":sum(len(x['candidates']) for x in results),"frontier":sum(len(x['frontier']) for x in results),"seconds":out['seconds']},indent=2))
if __name__=='__main__': main()
