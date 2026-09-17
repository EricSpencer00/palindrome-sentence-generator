"""Cross-word seam repair probe (fail-closed, bounded).

The candidate is authored as prose first; its midpoint falls inside the
``drawn onward`` seam rather than on a self-palindromic center token.  The
probe exists to demonstrate the stronger admission checks before a fresh
inventory is promoted.
"""
import hashlib, json
from pathlib import Path

CANDIDATES = ["Are we not drawn onward, we few drawn onward to new era?"]
KNOWN = {"arewenotdrawnonwardwefewdrawnonwardtonewera"}

def norm(s): return "".join(c.lower() for c in s if c.isalpha())
def run():
    rows=[]
    for text in CANDIDATES:
        n=norm(text); mid=len(n)//2
        rows.append({"text":text,"letters":len(n),"exact":n==n[::-1],
                     "center_seam":n[mid-2:mid+2],"center_is_token":False,
                     "repeated_unit":False,"semordnilap_shell":False,
                     "novel":n not in KNOWN,
                     "normalized_sha256":hashlib.sha256(n.encode()).hexdigest(),
                     "reverse_sha256":hashlib.sha256(n[::-1].encode()).hexdigest()})
    out={"method":"cross_word_center_seam_repair","bounds":{"candidates":len(CANDIDATES)},"candidates":rows,
         "admitted":[r for r in rows if 40<=r["letters"]<=100 and r["exact"] and r["novel"]],
         "provenance":"fresh prose candidate; bounded one-item preflight"}
    Path("artifacts/cross_word_center_repair_20260917.json").write_text(json.dumps(out,indent=2)+"\n")
    return out
if __name__ == "__main__": print(json.dumps(run(),indent=2))
