"""Independent source grammar for patient-first/locative-first adhesive repair."""
from __future__ import annotations


def parse_source(tokens):
    words = tuple(tokens)
    patients = {
        ("red", "metalwork"): ("metal", "singular", "cracks", ("epoxy",)),
        ("salvaged", "woodwork"): ("wood", "singular", "cracks", ("glue",)),
        ("red", "acrylic", "panels"): ("acrylic", "plural", "cracks", ("cement",)),
        ("stone", "sculptures"): ("stone", "plural", "cracks", ("epoxy",)),
        ("torn", "canvas", "paintings"): ("textile", "plural", "tears", ("starch", "paste")),
        ("damaged", "paper", "prints"): ("paper", "plural", "tears", ("starch", "paste")),
        ("old", "glasswork"): ("glass", "singular", "cracks", ("epoxy",)),
    }
    tools = {("paintbrushes",), ("a", "roller"), ("a", "spatula"), ("a", "spreader"), ("a", "syringe")}
    i = 0

    def expect(sequence):
        nonlocal i
        if words[i:i + len(sequence)] != tuple(sequence):
            raise ValueError("source syntax")
        i += len(sequence)

    try:
        locative = words[:1] == ("along",)
        named_defect = None
        if locative:
            expect(("along", "the"))
            named_defect = words[i]
            i += 1
            expect(("in",))
        matches = [(np, data) for np, data in patients.items() if words[i:i + len(np)] == np]
        if len(matches) != 1:
            return []
        np, (material, number, defect, adhesive) = matches[0]
        patient_span = (i, i + len(np))
        i += len(np)
        pronoun, possessive, copula = ("it", "its", "is") if number == "singular" else ("them", "their", "are")
        active, passive = ("mend", "mended") if defect == "tears" else ("repair", "repaired")
        if locative:
            expect(("from", "regional", "museums", "local"))
            agent = words[i]
            i += 1
            expect((active, pronoun, "with") + adhesive + ("using",))
        else:
            expect(("that", "arrived", "from", "regional", "museums", copula, passive, "along", possessive, defect,
                    "with") + adhesive + ("applied", "to", pronoun, "by", "local"))
            agent = words[i]
            i += 1
            expect(("using",))
        tool = words[i:]
        if agent not in {"artists", "conservators", "technicians"} or tool not in tools:
            return []
        i = len(words)
        if locative and named_defect != defect:
            return []
        return [{"complete": True, "construction": "locative_first" if locative else "patient_first_passive",
                 "patient": {"id": "patient", "surface": np, "span": patient_span, "material": material, "number": number},
                 "repair": {"action": active, "patient_id": "patient", "defect": defect,
                            "initial": 2, "predicted": 1},
                 "adhesive_application": {"adhesive": adhesive, "target_id": "patient", "agent": agent, "tool": tool},
                 "reference": {"surface": pronoun, "referent_id": "patient", "number_matches": True},
                 "semantic_relation_valid": True}]
    except (ValueError, IndexError):
        return []
