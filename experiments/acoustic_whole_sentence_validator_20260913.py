"""Third, independent whole-sentence parse and dimension/event binding check."""


def parse_acoustic_sentence(tokens):
    words = tuple(tokens)
    if not words or words[0] not in {"timekeepers", "gamekeepers", "beekeepers", "observers", "researchers", "listeners"}:
        return []
    i = 1
    relative = None
    if i < len(words) and words[i] == "who":
        start = i
        i += 1
        if i + 2 >= len(words) or words[i] not in {"carry", "use"} or words[i + 1] not in {"portable", "digital"} or words[i + 2] != "recorders":
            return []
        i += 3
        if i < len(words) and words[i] == "from":
            if i + 2 >= len(words) or words[i + 1] not in {"local", "national"} or words[i + 2] not in {"laboratories", "universities"}:
                return []
            i += 3
        relative = {"tokens": words[start:i], "subject_identity": "observer"}
    if i < len(words) and words[i] in {"carefully", "patiently"}:
        i += 1
    if i >= len(words) or words[i] not in {"record", "log", "measure"}:
        return []
    action = words[i]
    i += 1
    if i < len(words) and words[i] == "precise":
        i += 1
    if i >= len(words) or words[i] not in {"durations", "lengths", "weights"}:
        return []
    quantity = words[i]
    unit = "kilograms" if quantity == "weights" else "seconds"
    i += 1
    if i >= len(words) or words[i] != "of":
        return []
    i += 1
    if i < len(words) and words[i] in {"faint", "loud", "brief"}:
        i += 1
    if i >= len(words) or words[i] not in {"calls", "sounds", "noises"}:
        return []
    sound = words[i]
    i += 1
    if i >= len(words) or words[i] not in {"that", "which"}:
        return []
    i += 1
    if i < len(words) and words[i] in {"wild", "distant"}:
        i += 1
    if i + 1 >= len(words) or words[i] not in {"elk", "deer", "sheep", "birds", "bats", "frogs", "wolves", "whales", "crickets"}:
        return []
    animal = words[i]
    i += 1
    if words[i] not in {"emit", "produce", "make"} or i + 1 != len(words):
        return []
    # The object-relative gap denotes precisely the event whose temporal
    # extent the main predicate records; it is not an unrelated second clause.
    event = {"id": "acoustic-event", "kind": sound, "emitter": animal, "emission_verb": words[i], "temporal_extent_unit": "seconds"}
    measurement = {"event_id": "acoustic-event", "quantity": quantity, "unit": unit,
                   "observer_id": "observer", "action": action}
    valid = measurement["event_id"] == event["id"] and measurement["unit"] == event["temporal_extent_unit"]
    return [{"complete": True, "actor": words[0], "relative": relative, "event": event,
             "measurement": measurement, "semantic_relation_valid": valid,
             "semantic_witness": "measurement and relative-clause gap share the acoustic event; measurement dimension is time"}]
