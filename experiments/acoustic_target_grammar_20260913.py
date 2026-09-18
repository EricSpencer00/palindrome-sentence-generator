"""Independent target prefix grammar: lexicalized field-role nouns.

This file deliberately imports no source grammar, lexicon, transition builder,
or surface layout.  Its record contains an acoustic event and its temporal
measurement, not a preselected pair of text strings.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetState:
    position: str
    verb: str = ""
    dimension: str = ""
    sound: str = ""
    animal: str = ""


def target_transitions(s):
    def arc(token, position, verb=None, dimension=None, sound=None, animal=None):
        return token, TargetState(position, s.verb if verb is None else verb,
                                  s.dimension if dimension is None else dimension,
                                  s.sound if sound is None else sound, s.animal if animal is None else animal)
    actions = lambda: [arc(token, "measure_np", verb=token) for token in ("measure", "log", "record")]
    manners = lambda: [arc(token, "verb",) for token in ("patiently", "carefully")]
    if s.position == "initial":
        return [arc(token, "subject_done") for token in ("timekeepers", "gamekeepers", "beekeepers", "observers", "researchers", "listeners")]
    if s.position == "subject_done":
        return [arc("who", "human_relative")] + manners() + actions()
    if s.position == "human_relative":
        return [arc(token, "tool_adj") for token in ("use", "carry")]
    if s.position == "tool_adj":
        return [arc(token, "tool_noun") for token in ("digital", "portable")]
    if s.position == "tool_noun":
        return [arc("recorders", "tool_complete")]
    if s.position == "tool_complete":
        return [arc("from", "institution_adj")] + manners() + actions()
    if s.position == "institution_adj":
        return [arc(token, "institution") for token in ("national", "local")]
    if s.position == "institution":
        return [arc(token, "human_relative_done") for token in ("universities", "laboratories")]
    if s.position == "human_relative_done":
        return manners() + actions()
    if s.position == "verb":
        return actions()
    if s.position == "measure_np":
        return [arc("precise", "measure_head")] + [arc(token, "event_preposition", dimension="temporal_extent") for token in ("lengths", "durations")]
    if s.position == "measure_head":
        return [arc(token, "event_preposition", dimension="temporal_extent") for token in ("lengths", "durations")]
    if s.position == "event_preposition":
        return [arc("of", "event_np")]
    if s.position == "event_np":
        return [arc(token, "event_head") for token in ("brief", "loud", "faint")] + [arc(token, "event_relative", sound=token) for token in ("noises", "sounds", "calls")]
    if s.position == "event_head":
        return [arc(token, "event_relative", sound=token) for token in ("noises", "sounds", "calls")]
    if s.position == "event_relative":
        return [arc(token, "emitter_np") for token in ("which", "that")]
    if s.position == "emitter_np":
        return [arc(token, "emitter_head") for token in ("distant", "wild")] + [arc(token, "emit_verb", animal=token) for token in ("crickets", "whales", "wolves", "frogs", "bats", "birds", "sheep", "deer", "elk")]
    if s.position == "emitter_head":
        return [arc(token, "emit_verb", animal=token) for token in ("crickets", "whales", "wolves", "frogs", "bats", "birds", "sheep", "deer", "elk")]
    if s.position == "emit_verb":
        return [arc(token, "complete") for token in ("make", "produce", "emit")]
    return []


def target_signature(state):
    if state.position != "complete":
        return None
    return ("human_observer", state.verb, state.dimension, "seconds", state.sound, state.animal,
            "relative_gap_is_measured_acoustic_event")
