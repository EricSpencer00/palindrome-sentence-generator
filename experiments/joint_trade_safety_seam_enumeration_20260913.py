"""Typed, jointly enumerated trade/safety seam feasibility before generation.

This is a lexical/structural preflight, not a palindrome generator or a human
readability certificate. It enumerates complete ordinary trade-role NPs and
human safety-role/event phrases together, records their exact outer matching
letters, and identifies the first contradiction. No retained rendered phrase
may be installed as a frozen half: a successor must expose and lexicalize its
own connected grammar leaves and reproduce the pair with the exact emitter.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path


@dataclass(frozen=True)
class Trade:
    commodity: str
    venue: str
    head: str
    relation: str

    def words(self):
        return tuple((self.commodity+" "+self.venue+" "+self.head).split())


@dataclass(frozen=True)
class Safety:
    modifier: str
    head: str

    def words(self):
        return tuple((self.modifier+" "+self.head).split())


# These are independent category values, not catalogue phrases. Venue/head
# compatibility prevents combinations like a street shopkeeper.
COMMODITIES=("dessert","desserts","coffee","decaf","food","fruit","flower","pastry","bread",
             "snack","plant","clothing","furniture","wood","tile","cart")
VENUE_HEADS={
    "":("vendor","seller","retailer","dealer","trader","supplier","shopkeeper","sales assistant","sales clerk","sales representative"),
    "street":("vendor","seller"),"market":("vendor","trader","stallholder"),
    "shop":("owner","assistant"),"railway":("vendor",),"rail":("vendor",),
    "station":("vendor",),"kiosk":("vendor","owner"),"stall":("owner",),"counter":("assistant",),
}
SAFETY_MODIFIERS={
    "expert":("safety","materials","testing"),
    "tester":("safety","materials","quality"),
    "inspector":("safety","quality"),
    "engineer":("safety","structural","materials","testing"),
    "technician":("testing","laboratory","quality"),
    "assessor":("safety","quality"),
    "auditor":("safety",),
    "designer":("product","furniture"),
    "specialist":("safety","testing"),
    "cabinetmaker":("skilled","experienced"),
    "carpenter":("skilled","experienced"),
    "apprentice":("trained","skilled","experienced"),
}
EVENTS=("stressed","tested","inspected","examined","opened","closed","checked","moved","shook",
        "bent","broke","damaged","repaired","restored","painted","finished","sanded","cut","set",
        "reset","loaded","unloaded","filled","emptied","jammed","fixed","tried","fitted","altered",
        "refaced","refilled","removed","pulled","pushed","slid","lifted","lowered","drilled","struck",
        "split","sawed","hammered","hit","scratched","scraped","measured","weighed","sealed","covered",
        "assembled","installed","leveled","beveled","fastened","tightened","widened","faced","scanned")


def letters(text):
    return "".join(c.lower() for c in text if "a"<=c.lower()<="z")


def matched_prefix(left,right):
    a,b=letters(left),letters(right)[::-1]
    i=0
    while i<min(len(a),len(b)) and a[i]==b[i]:i+=1
    return {"matched_pairs":i,"left_matched":a[:i],"right_matched":b[:i],
            "next_left":a[i:i+1],"next_right":b[i:i+1],
            "fully_consumed_one_phrase":i==min(len(a),len(b))}


def enumerate_seams(limit=25):
    trades=[Trade(c,v,h,"person sells the named commodity at the named venue" if v else "person sells the named commodity")
            for c in COMMODITIES for v,heads in VENUE_HEADS.items() for h in heads]
    safety=[Safety(m,h) for h,modifiers in SAFETY_MODIFIERS.items() for m in modifiers]
    best=[]
    evaluated=0
    for trade,role,event in itertools.product(trades,safety,EVENTS):
        left="reward a "+" ".join(trade.words())
        right="the "+" ".join(role.words())+" "+event+" a drawer"
        match=matched_prefix(left,right)
        evaluated+=1
        if match["matched_pairs"]<15:continue
        row={"left_trade_words":list(trade.words()),"right_safety_words":list(role.words()),
             "artifact_event":event,"trade":asdict(trade),"safety":asdict(role),**match,
             "agent_type":"person","patient_type":"physical_drawer","candidate":False,
             "reader_status":"unreviewed typed phrase feasibility only"}
        best.append(row)
    best.sort(key=lambda row:(-row["matched_pairs"],row["left_trade_words"],row["right_safety_words"],row["artifact_event"]))
    return {"pairs_examined":evaluated,"ordinary_typed_pair_count":len(trades)*len(safety)*len(EVENTS),
            "maximum_matched_pairs":best[0]["matched_pairs"] if best else 0,
            "top_pairs":best[:limit],"beyond_37_emitted_letters_possible_in_preflight":any(r["matched_pairs"]>=19 for r in best),
            "generator_run_authorized_by_this_preflight":False}


def run():
    return {"method":"joint_trade_safety_semantic_seam_preflight","diagnostic_only":True,
        "prior_structural_blocker":{"left_span":"dessert street","right_event":"stressed a drawer",
            "required_pre_event_suffix":"teert","reason":"matching the full street frontier forces a subject suffix teert; no available ordinary human title supplies it"},
        "rules":{"drawer_sense":"physical_furniture_only","human_heads":"ordinary occupational roles",
            "phrase_pair_is_never_a_frozen_generation_seed":True,"central_admission_required_for_any_complete_candidate":True},
        "provenance":{"source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),"catalogue_construction_material":False},
        **enumerate_seams()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():parser.error("refusing to overwrite existing artifact")
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"out":str(args.out),"pairs_examined":result["pairs_examined"],
                      "maximum_matched_pairs":result["maximum_matched_pairs"],"top":result["top_pairs"][:1]}))


if __name__=="__main__":main()
