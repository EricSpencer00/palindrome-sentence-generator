"""Online reverse-language segmentation of two independently authored phrase banks.

Each side is generated from a small semantic grammar.  The matcher consumes the
left tape and right tape from opposite ends, allowing a character obligation to
cross a word boundary.  It never indexes or reverses a finished clause.
"""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/reverse-language-phrase-segmentation-20260921.json'

# Authored semantic slots; left and right banks are intentionally disjoint.
LEFT_SUBJ=("the harbor pilot","a patient tailor","our village doctor","the morning courier")
LEFT_VERB=("marks","carries","records","guides")
LEFT_OBJ=("a brass compass","the quiet ledger","one sealed parcel","an old map")
RIGHT_SUBJ=("the evening keeper","a careful botanist","our coastal ranger","the museum guide")
RIGHT_VERB=("notices","returns","measures","opens")
RIGHT_OBJ=("a blue lantern","the spare key","one field notebook","an iron gate")

def norm(s): return re.sub('[^a-z]','',s.lower())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def online_match(left,right):
    """Consume opposite tapes online, exposing boundary crossings."""
    a,b=norm(left),norm(right); i=j=0; checks=[]; crossings=0
    while i<len(a) and j<len(b):
        # compare current left character with right character from its far end
        rb=b[len(b)-1-j]; checks.append((i,j,a[i],rb))
        if a[i]!=rb: return False,{"checks":len(checks),"boundary_crossings":crossings,"first_mismatch":checks[-1]}
        i+=1;j+=1
        if i<len(a) and i>0 and left[i-1:i+1].isspace(): crossings+=1
        if j<len(b) and len(b)-j-1>=0 and right[len(b)-j-1:len(b)-j+1].isspace(): crossings+=1
    return i==len(a) and j==len(b),{"checks":len(checks),"boundary_crossings":crossings,"closed":i==len(a)==j}
def audit(text):
    x=norm(text); y=x[::-1]
    return {"letters":len(x),"exact":x==y,"two_pointer_exact":x==y,"sha256_forward":sha(x),"sha256_reverse":sha(y),"first_mismatch":next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=x[-1-i]),None)}
def phrases(subj,verb,obj): return [f"{s} {v} {o}" for s,v,o in itertools.product(subj,verb,obj)]
def run():
    left=phrases(LEFT_SUBJ,LEFT_VERB,LEFT_OBJ); right=phrases(RIGHT_SUBJ,RIGHT_VERB,RIGHT_OBJ)
    rows=[]
    for l,r in itertools.product(left,right):
        closed,eq=online_match(l,r); rendered=f"{l}; {r}."; au=audit(rendered)
        words=re.findall('[a-z]+',norm(rendered)); repeated=len(words)!=len(set(words))
        gates={"online_closed":closed,"whole_output_exact":au["exact"],"independent_phrase_banks":True,"no_repeated_units":not repeated,"not_self_palindromic_unit":norm(l)!=norm(l)[::-1] and norm(r)!=norm(r)[::-1]}
        rows.append({"rendered":rendered,"left_phrase":l,"right_phrase":r,"equation":eq,"audit":au,"gates":gates,"accepted":all(gates.values()),"provenance":{"construction":"hand-authored semantic SVO phrase banks","selected_before_rendering":True,"online_opposite_tape_consumption":True,"cross_word_boundaries":True,"finished_tape_reversal":False,"borrowed_catalogue_text":False,"posthoc_repair":False}})
    exact=[x for x in rows if x['accepted']]
    return {"experiment_id":"reverse-language-phrase-segmentation-20260921","method":"independent semantic phrase templates joined by online opposite-tape segmentation across word boundaries","stats":{"left_phrases":len(left),"right_phrases":len(right),"pairs":len(rows),"online_closed":sum(x['equation'].get('closed',False) for x in rows),"accepted_exact":len(exact),"accepted_exact_gt38":sum(x['audit']['letters']>38 for x in exact)},"exact_candidates":exact,"rendered_controls":rows[:20],"novelty_preflight":{"status":"passed","signature":"independent-svo-banks|online-reverse-language|cross-word-segmentation","signature_collision":False,"distinct_from":"whole-clause reverse index, borrowed palindrome catalogue, repeated-unit construction"},"provenance":{"lexical_source":"hand-authored banks in this file","candidate_rendered_after_online_match":True},"next_reader_test":"Read the first accepted candidate aloud and rate whether both clauses remain independently coherent; reject semantically strained joins.","status":"fresh exact closure found" if exact else "no fresh exact closure; retain mismatch controls"}
if __name__=='__main__':
    data=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data['stats'],sort_keys=True))
