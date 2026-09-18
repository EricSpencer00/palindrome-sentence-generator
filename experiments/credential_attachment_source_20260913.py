"""Source analysis: lexical credential subject and owner-bound duty constituent."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    phase: str
    operation: str = ""
    owner: str = ""
    duty: str = ""


def source_steps(s):
    def step(word, phase, operation=None, owner=None, duty=None):
        return word, Source(phase, s.operation if operation is None else operation,
                            s.owner if owner is None else owner, s.duty if duty is None else duty)
    if s.phase == "start":
        return [step(word, "subject") for word in ("keycards", "bookplates", "armbands", "wristbands", "passcards", "numberplates")]
    if s.phase == "subject":
        return [step("that", "relative0")] + [step(w, "determiner", operation=w) for w in ("identify", "label", "describe", "distinguish", "reveal", "mark", "name", "specify")]
    relative = ("display", "identifying", "information", "supplied", "by", "authorized", "officials")
    if s.phase.startswith("relative"):
        i = int(s.phase[len("relative"):])
        return [step(relative[i], "verb" if i == len(relative) - 1 else f"relative{i + 1}")]
    if s.phase == "verb":
        return [step(w, "determiner", operation=w) for w in ("identify", "label", "describe", "distinguish", "reveal", "mark", "name", "specify")]
    if s.phase == "determiner":
        return [step("the", "owner")]
    if s.phase == "owner":
        return [step(w, "duty", owner=w) for w in ("guards", "workers", "residents", "locals", "bishops", "smiths")]
    if s.phase == "duty":
        return [step(w, "man", duty=w) for w in ("door", "boat", "watch")]
    if s.phase == "man":
        return [step("man", "complete")]
    return []


def source_meaning(s):
    if s.phase != "complete":
        return None
    return {"shared_core": ("credential", s.operation, "duty_attendant", s.duty, s.owner),
            "attachment": {"possessor": s.owner, "attaches_to": "duty_domain", "duty_domain": s.duty},
            "attendant_relation": "the man assigned to the owner-associated duty domain",
            "constituency": "[[owners' duty] man]"}
