"""Independently authored source grammar: open-compound field roles."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceState:
    phase: str
    operation: str = ""
    quantity: str = ""
    event: str = ""
    emitter: str = ""


def source_transitions(s):
    def edge(word, phase, **features):
        return word, SourceState(phase, features.get("operation", s.operation),
                                 features.get("quantity", s.quantity), features.get("event", s.event),
                                 features.get("emitter", s.emitter))
    if s.phase == "start":
        return [edge(w, "compound_head") for w in ("time", "game", "bee")] + [edge(w, "agent") for w in ("observers", "researchers", "listeners")]
    if s.phase == "compound_head":
        return [edge("keepers", "agent")]
    if s.phase == "agent":
        return [edge("who", "relative_action")] + [edge(w, "manner_done") for w in ("carefully", "patiently")] + [edge(w, "quantity", operation=w) for w in ("record", "log", "measure")]
    if s.phase == "relative_action":
        return [edge(w, "equipment_modifier") for w in ("carry", "use")]
    if s.phase == "equipment_modifier":
        return [edge(w, "equipment") for w in ("portable", "digital")]
    if s.phase == "equipment":
        return [edge("recorders", "relative_complete")]
    if s.phase == "relative_complete":
        return [edge("from", "source_modifier")] + [edge(w, "manner_done") for w in ("carefully", "patiently")] + [edge(w, "quantity", operation=w) for w in ("record", "log", "measure")]
    if s.phase == "source_modifier":
        return [edge(w, "source_head") for w in ("local", "national")]
    if s.phase == "source_head":
        return [edge(w, "relative_no_source") for w in ("laboratories", "universities")]
    if s.phase == "relative_no_source":
        return [edge(w, "manner_done") for w in ("carefully", "patiently")] + [edge(w, "quantity", operation=w) for w in ("record", "log", "measure")]
    if s.phase == "manner_done":
        return [edge(w, "quantity", operation=w) for w in ("record", "log", "measure")]
    if s.phase == "quantity":
        return [edge("precise", "quantity_head")] + [edge(w, "of", quantity="temporal_extent") for w in ("durations", "lengths")]
    if s.phase == "quantity_head":
        return [edge(w, "of", quantity="temporal_extent") for w in ("durations", "lengths")]
    if s.phase == "of":
        return [edge("of", "sound")]
    if s.phase == "sound":
        return [edge(w, "sound_head") for w in ("faint", "loud", "brief")] + [edge(w, "relative_link", event=w) for w in ("calls", "sounds", "noises")]
    if s.phase == "sound_head":
        return [edge(w, "relative_link", event=w) for w in ("calls", "sounds", "noises")]
    if s.phase == "relative_link":
        return [edge(w, "animal") for w in ("that", "which")]
    if s.phase == "animal":
        return [edge(w, "animal_head") for w in ("wild", "distant")] + [edge(w, "emission", emitter=w) for w in ("elk", "deer", "sheep", "birds", "bats", "frogs", "wolves", "whales", "crickets")]
    if s.phase == "animal_head":
        return [edge(w, "emission", emitter=w) for w in ("elk", "deer", "sheep", "birds", "bats", "frogs", "wolves", "whales", "crickets")]
    if s.phase == "emission":
        return [edge(w, "done") for w in ("emit", "produce", "make")]
    return []


def source_signature(state):
    if state.phase != "done":
        return None
    return ("human_observer", state.operation, state.quantity, "seconds", state.event, state.emitter,
            "relative_gap_is_measured_acoustic_event")
