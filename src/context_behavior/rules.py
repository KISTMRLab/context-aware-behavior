"""Labelled rule fallback over the ontology synonym lists (used when no trained checkpoint exists).

It is a transparent phrase matcher, not the paper's learned classifier:
* questions about the room ("Is the window open?") and first-person statements are conversation;
* a negation before the verb ("Don't open the window") makes the sentence conversation;
* split phrasal verbs use the particle that follows the verb ("Turn the lamp off, it is on" -> Turn off);
* "go sit on the chair" prefers the second, non-locomotion verb;
* a pronoun target ("open it") resolves to an object named elsewhere in the sentence.
"""
from __future__ import annotations

import re

from .ontology import CONVERSATION, NONE, VIRTUAL_HUMAN, Ontology

NEGATION = re.compile(r"\b(?:don'?t|do not|does not|doesn'?t|never|no need to|not|stop|shouldn'?t|won'?t)\b")
SUGGESTION = re.compile(r"\bwhy (?:don'?t you|not)\b")
QUESTION = re.compile(r"^(?:is|are|was|were|do(?! not\b| nothing\b)|does|did|when|where|what|who|whose|which|how|have you|has)\b")
STATEMENT = re.compile(r"^(?:i|i'm|im|i am|i've|it's|it is|its|this|that|there|the|my)\b")
PHRASAL = re.compile(r"\b(turn|switch|power|flip)\s+(?:(?!on\b|off\b)[a-z']+\s+){0,3}?(on|off)\b")
CLAUSE = re.compile(r"[,.;!?]|\bbut\b|\bthen\b|\bbecause\b|\bsince\b|\bwhile\b|\bso\b")
NOT_SIDE = re.compile(r"\b(?:all right|alright|right now|right away|right there|right here|that'?s right|left over)\b")
DETERMINERS = r"(?:the|a|an|my|your|this|that|its|his|her|our)\s+"
LOCOMOTION_LEAD = {"Walk"}


def _normal(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").replace("_", " ").split())


def _inflected(phrase: str) -> str:
    """Regex for a phrase whose first word may be inflected: open/opens/opening, sit/sitting, close/closing."""
    first, _, rest = phrase.partition(" ")
    forms = {first, first + "s", first + "ing", first + first[-1] + "ing"}
    if first.endswith("e"):
        forms.add(first[:-1] + "ing")
    pattern = "(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + ")"
    return r"\b" + pattern + (r"\s+" + re.escape(rest) if rest else "") + r"\b"


def _find(phrases: list[tuple[str, str]], text: str, taken: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    found = []
    for phrase, label in phrases:
        for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            taken.append(span)
            found.append((span[0], span[1], label))
    return sorted(found)


def mentioned_targets(text: str, ontology: Ontology) -> list[str]:
    """Target classes named anywhere in the text, in order of appearance."""
    return [label for _, _, label in _find(ontology.surface_index("target"), _normal(text), [])]


class RuleInterpreter:
    backend = "labelled rule fallback (ontology synonym matching, not a trained model)"

    def __init__(self, ontology: Ontology | None = None):
        self.ontology = ontology or Ontology.load()
        self.actions = [(re.compile(_inflected(p)), p, label) for p, label in self.ontology.surface_index("action")]
        self.targets = self.ontology.surface_index("target")

    def _actions(self, clause: str) -> list[tuple[int, list[tuple[int, int]], str]]:
        """Non-overlapping action mentions in order: (start, spans, class)."""
        found, taken = [], []
        for match in PHRASAL.finditer(clause):
            label = "Turn on" if match.group(2) == "on" else "Turn off"
            found.append((match.start(), [match.span(1), match.span(2)], label))
            taken += [match.span(1), match.span(2)]
        for regex, _, label in self.actions:  # longest phrases first
            for match in regex.finditer(clause):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in taken):
                    continue
                taken.append(span)
                found.append((span[0], [span], label))
        return sorted(found)

    def predict(self, text: str) -> dict:
        normal = _normal(text)
        if QUESTION.match(normal) and not re.match(r"^(?:do you mind|how about)\b", normal):
            return self._result(CONVERSATION, NONE, NONE, NONE, text)
        clauses = [part.strip() for part in CLAUSE.split(normal) if part and part.strip()]
        mentioned = mentioned_targets(text, self.ontology)
        for clause in clauses or [normal]:
            if STATEMENT.match(clause) and not re.search(r"\byou\b", clause):
                continue  # "I'm cold", "It's dark in here": context for the request, not a request
            actions = self._actions(clause)
            if not actions:
                continue
            start, spans, action = actions[0]
            if action in LOCOMOTION_LEAD and len(actions) > 1 and actions[1][2] not in LOCOMOTION_LEAD | {"Run"}:
                start, spans, action = actions[1]  # "go sit on the chair", "go to bed and lie down"
            if NEGATION.search(SUGGESTION.sub(" ", clause[:start])):
                return self._result(CONVERSATION, NONE, NONE, NONE, text, negated=True)
            taken = [span for item in actions for span in item[1]]
            targets = _find(self.targets, clause, taken)
            after = [item for item in targets if item[0] >= spans[0][0]]
            pool = after or targets
            chosen = (pool[-1] if action == "Put" and pool else pool[0]) if pool else None
            target = chosen[2] if chosen else NONE
            if target == NONE and re.search(r"\b(?:it|them|that one)\b", clause) and mentioned:
                target = mentioned[-1]  # "I need something from the drawer, open it"
            position = self._position(clause, spans, chosen)
            if target == NONE and action not in self.ontology.target_optional_actions:
                position = position if position in {"Left", "Right"} else NONE
            return self._result(VIRTUAL_HUMAN, action, position, target, text, mentioned=mentioned)
        return self._result(CONVERSATION, NONE, NONE, NONE, text)

    def _position(self, clause: str, action_spans, chosen) -> str:
        masked = list(clause)
        for start, end in action_spans:
            masked[start:end] = "#" * (end - start)
        masked = NOT_SIDE.sub(lambda m: "#" * len(m.group(0)), "".join(masked))
        side = re.search(r"\b(left|right)\b", masked)
        if side:
            return side.group(1).capitalize()
        if re.search(r"\b(?:to|towards?|over to|up to)\s+(?:me|here|us)\b", masked):
            return "To"
        if chosen:
            before = masked[: chosen[0]]
            match = re.search(r"\b(on top of|onto|on|upon|into|inside|in|towards|toward|over to|up to|to)\s+"
                              r"(?:[a-z]+\s+){0,3}$", before)
            if match:
                return self.ontology.from_surface("position", match.group(1))
        return NONE

    def _result(self, subject, action, position, target, text, negated=False, mentioned=None) -> dict:
        return {"subject": subject, "action": action, "position": position, "target": target, "negated": negated,
                "mentioned": mentioned if mentioned is not None else mentioned_targets(text, self.ontology),
                "backend": self.backend}
