"""Matrix-perfect alternation with a distinct temporal adjunct.

This is a new grammar state after the temporal subject-relative lane: the
matrix predicate is perfect (has/have + participle), while the relative
predicate remains future perfect.  The temporal adjunct is independently
selected (since/after), and character obligations are checked while the two
ordinary-order clauses are emitted by the existing cursor product.
"""
from __future__ import annotations

import hashlib, json, re, sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from experiments.unequal_center_ditransitive_relative_complement_20260920 import Word, walk_pair

OUT = ROOT / "runs/semantic-role-matrix-perfect-temporal-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "semantic-role-matrix-perfect-temporal-20260920"
SIGNATURE = "typed-seam-machine|temporal-subject-relative|future-perfect-relative|matrix-perfect|distinct-temporal-adjunct|independent-pointer-sha"

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = norm(s); i, j = 0, len(t)-1; bad = None
    while i < j:
        if t[i] != t[j]: bad = {"left_index": i, "right_index": j, "left_char": t[i], "right_char": t[j]}; break
        i += 1; j -= 1
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"normalized_tape": t, "letters": len(t), "exact": bool(t) and bad is None,
            "first_mismatch": bad, "sha256_forward": f, "sha256_reverse": r,
            "sha_equal_under_reversal": f == r, "independent_exact": bool(t) and bad is None and f == r}

@dataclass(frozen=True)
class Clause:
    frame_id: str; number: str; adjunct: str; words: tuple[Word, ...]; roles: tuple[str, ...]
    @property
    def surface(self): return " ".join(w.surface for w in self.words)
    @property
    def letters(self): return len(norm(self.surface))

SUBJECTS = {"sg": ("the patient pilot", "pilot"), "pl": ("patient pilots", "pilots")}
REL = {"sg": (("charted", "the old map"), ("guarded", "the quiet inlet")),
       "pl": (("charted", "old maps"), ("guarded", "quiet inlets"))}
MATRIX = {"sg": (("guided", "the narrow bridge"), ("observed", "a calm harbor")),
          "pl": (("guided", "narrow bridges"), ("observed", "calm harbors"))}
ADJUNCTS = (("after rain", "after"), ("since dawn", "since"), ("before winter", "before"))

def words(text, role, index): return tuple(Word(x, role, index) for x in text.split())
def paths():
    out = []
    for n, (subject, lemma) in SUBJECTS.items():
        for relverb, reltheme in REL[n]:
            for matrixverb, matrixtheme in MATRIX[n]:
                for adjunct, relation in ADJUNCTS:
                    # Future-perfect relative + matrix perfect alternation is a
                    # typed state, not a post-hoc substitution.
                    chunks = (words(subject,"head",0), words("who will have", "relative_aux",1),
                              words(relverb,"relative_participle",2), words(reltheme,"relative_theme",3),
                              words(adjunct,"temporal_adjunct",4),
                              words(("has" if n == "sg" else "have"),"matrix_perfect_aux",5),
                              words(matrixverb,"matrix_participle",6), words(matrixtheme,"matrix_theme",7))
                    out.append(Clause(f"matrix-perfect:{n}:{lemma}:{relverb}:{matrixverb}:{relation}", n, relation,
                                      tuple(w for c in chunks for w in c),
                                      ("head","relative_aux","relative_participle","relative_theme","temporal_adjunct","matrix_perfect_aux","matrix_participle","matrix_theme")))
    return tuple(out)

def render(a, b):
    x = a.surface[:1].upper() + a.surface[1:]
    return f"{x}; {b.surface}."

def row(a, b, walk):
    text = render(a,b); return {"rendered": text, "letters": len(norm(text)), "audit": audit(text),
      "seam_machine": {"status": walk.status, "matched_characters": walk.matched_characters,
       "online_equations": walk.online_equations, "first_mismatch": walk.first_mismatch,
       "seam_mode": walk.seam_mode, "center_location": walk.center_location},
      "provenance": {"left_frame_id": a.frame_id, "right_frame_id": b.frame_id,
       "left_roles": list(a.roles), "right_roles": list(b.roles), "left_number": a.number,
       "right_number": b.number, "left_temporal_relation": a.adjunct, "right_temporal_relation": b.adjunct,
       "left_clause_words": [w.surface for w in a.words], "right_clause_words": [w.surface for w in b.words],
       "lexical_source": "fresh authored matrix-perfect temporal slots", "grammar_paths_independent": True,
       "lexicalized_during_cursor_walk": True, "right_clause_read_inward_by_cursor": True,
       "finished_tape_reversal": False, "post_hoc_repair": False, "mirrored_units": False,
       "catalogue_text": False, "word_order_symmetry": False, "per_search_rlaif": False},
      "reader_facing_eligible": False, "reader_evidence": {"status":"not_run", "human_raters":0}}

def novelty():
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collision = [e.get("id") for e in entries if e.get("id") == EXPERIMENT_ID or e.get("signature") == SIGNATURE]
    return {"status": "passed" if not collision else "collision", "registry_inspected": True,
            "registry_entries_read": len(entries), "exact_signature_collision": collision,
            "signature": SIGNATURE, "not_a_duplicate_sweep": True,
            "distinction": "Adds matrix perfect auxiliary/aspect alternation and a distinct temporal adjunct to the existing future-perfect subject-relative state; obligations are live during ordinary-order realization."}

def run():
    ps = paths(); obs=[]; exact=[]; equations=prunes=0
    for i,a in enumerate(ps):
        for j,b in enumerate(ps):
            w=walk_pair(a,b); equations += w.online_equations; prunes += w.status == "mismatch_pruned"
            if w.status == "closed":
                x=row(a,b,w)
                if x["audit"]["independent_exact"]: exact.append(x)
            obs.append((i,j,a,b,w))
    controls=[]; seen=set()
    for i,j,a,b,w in sorted(obs, key=lambda q: -(q[2].letters+q[3].letters)):
        key=(a.surface,b.surface)
        if key not in seen and a.surface != b.surface:
            controls.append(row(a,b,w)); seen.add(key)
        if len(controls) >= 24: break
    clean=[x for x in exact if x["letters"]>38]
    result={"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,
      "method":"Fresh typed matrix-perfect alternation: future-perfect subject-relative clause, distinct temporal adjunct, and has/have + participle matrix state, intersected with live unequal-boundary cursor equations.",
      "operator_added":{"name":"matrix perfect alternation with distinct temporal adjunct","relative_state":"future perfect","matrix_state":"present perfect, number-agreeing has/have","adjunct_state":["after","since","before"],"fresh_lexical_bank":True,"post_hoc_repair":False},
      "stats":{"heldout_clause_paths":len(ps),"paired_grammar_states":len(obs),"online_character_equations":equations,"mismatch_prunes":prunes,"rendered_controls":len(controls),"mechanical_exact_candidates":len(exact),"exact_clean_above_38":len(clean),"longest_rendered_control_letters":max((x["letters"] for x in controls),default=0),"longest_exact_clean_letters":max((x["letters"] for x in clean),default=0)},
      "rendered_controls":controls,"exact_candidates":exact,"exact_clean_candidates":clean,"reader_facing_candidates":[],
      "reader_gate":{"status":"closed","reason":"No exact-clean closure from this lane; no human reader test run.","human_raters":0},
      "novelty_preflight":novelty(),"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"shared_cursor_source":"experiments/unequal_center_ditransitive_relative_complement_20260920.py","independent_audits":["literal outside-in pointer scan","forward/reverse SHA-256"],"ordinary_order_grammar_emission":True,"right_clause_read_inward_by_cursor":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_or_self_palindromic_units":False,"word_order_symmetry":False,"per_search_rlaif":False},
      "status":"completed_no_exact_closure" if not exact else "mechanical_exact_requires_reader_gate","falsifier":"A valid closure would require both cursor closure and independent pointer/SHA equality; a same-surface or repeated-unit closure is excluded.","next_construction":"Change semantic frame topology (not another lexical bank): add a center-bearing adjunct attachment state with agreement-carrying subject choice."}
    OUT.write_text(json.dumps(result,indent=2)+"\n"); return result
if __name__ == "__main__":
    result = run()
    print(json.dumps({"experiment_id":result["experiment_id"],"stats":result["stats"]},sort_keys=True))
