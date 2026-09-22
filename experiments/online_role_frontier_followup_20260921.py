"""Bounded subject/object lexical construction selected by an actual frontier."""
from pathlib import Path
import hashlib
import json
from experiments.online_regular_language_palindrome_20260921 import intersect, independent_audit

ROOT = Path(__file__).resolve().parents[1]


def main():
    # In the first experiment, the object starts with a c/q/n while the
    # opposing agent 'clerk' ends k. 'kind' repairs just one equation, then
    # i conflicts with r. This motivates jointly replacing the role heads.
    cases = {
        'adjective_transition_check': [
            ['Nora '], ['asks '], ['a kind clerk; '],
            ['a careful clerk asks '], ['Aaron.']],
        'joint_role_head_construction': [
            ['Nora '], ['asks '],
            ['a dog; ', 'a dog a riddle; ', 'a dog about the moon; ',
             'a dog whether the moon is full; '],
            ['a god asks '], ['Aaron.']],
    }
    runs = {}
    for name, slots in cases.items():
        result = intersect(slots)
        result['slots'] = slots
        result['intact_control'] = ''.join(options[-1] for options in slots)
        result['control_audit'] = independent_audit(result['intact_control'])
        for row in result['candidates']:
            row['provenance'] = 'Fresh lexical role choices in the declared finite grammar; no seed or catalogue sentence.'
            row['admitted'] = False
            row['exclusion'] = 'Only 25 letters; reciprocal dog/god and Nora/Aaron boundary construction. Diagnostic closure only, no longer-output claim and no shortcut-clean claim.'
            row['human_readability_evidence'] = None
        runs[name] = result
    payload = dict(experiment_id='online-role-frontier-followup-20260921',
        provenance=dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        source='Hand-authored fable scene; roles allow a speaking animal and a deity.'),
        runs=runs, fresh_exact_gt38=0, reader_admissions=0,
        conclusion='Real online closure is reachable, but the bounded long question-complement constructions do not close.',
        next_transition='After 13 paired letters, the riddle and about-complements require r/b but their backward frontier offers e/g/l/n. The whether-complement stops at pair 12 requiring w versus a. A new grammatical complement must supply these specific equations; merely adding more agent nouns cannot cross this frontier.',
        next_reader_facing_test='None yet: no longer candidate qualifies for a blinded reader package.')
    out = ROOT / 'runs/online-role-frontier-followup-20260921.json'
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({name: dict(stats=r['stats'], candidates=r['candidates'],
                                deepest=r['dead_frontier_examples'][:1])
                      for name, r in runs.items()}, indent=2))


if __name__ == '__main__':
    main()
