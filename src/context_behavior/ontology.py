"""Table 1 interaction vocabulary: Subject (2), Action (14), Position (6), Target (11)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

HEADS = ("subject", "action", "position", "target")
ENTITY_HEADS = ("action", "position", "target")
NONE = "None"
CONVERSATION = "None"
VIRTUAL_HUMAN = "Virtual Human"
DEFAULT_PATH = Path(__file__).resolve().parent / "resources" / "ontology.json"
EXPECTED_SIZES = {"subject": 2, "action": 14, "position": 6, "target": 11}


def _key(text: str) -> str:
    return " ".join(str(text).casefold().replace("_", " ").replace("-", " ").split())


@dataclass(frozen=True)
class Ontology:
    classes: dict[str, tuple[str, ...]]
    synonyms: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    target_optional_actions: frozenset[str] = frozenset()
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(cls, raw: dict) -> "Ontology":
        classes = {head: tuple(raw["classes"][head]) for head in HEADS}
        for head, values in classes.items():
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {head} classes")
            if values[0] != NONE:
                raise ValueError(f"{head} classes must start with None")
        synonyms = {head: {name: tuple(words) for name, words in raw.get("synonyms", {}).get(head, {}).items()}
                    for head in ENTITY_HEADS}
        for head, mapping in synonyms.items():
            unknown = set(mapping) - set(classes[head])
            if unknown:
                raise ValueError(f"synonyms for unknown {head} classes: {sorted(unknown)}")
        optional = frozenset(raw.get("target_optional_actions", ()))
        if optional - set(classes["action"]):
            raise ValueError("target_optional_actions contains unknown actions")
        return cls(classes, synonyms, optional, raw)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Ontology":
        return cls.from_dict(json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return self.raw or {"classes": {k: list(v) for k, v in self.classes.items()}}

    def index(self, head: str, label: str) -> int:
        return self.classes[head].index(self.canonical(head, label))

    def canonical(self, head: str, label: str | None) -> str:
        """Return the exact Table 1 class for a class name written in any case/spacing."""
        if label is None or _key(label) in {"", "none", "o"}:
            return NONE
        if head == "subject" and _key(label) in {"conversation", "small talk", "none (small talk)"}:
            return CONVERSATION
        if head == "subject" and _key(label) in {"action", "virtual human", "vh"}:
            return VIRTUAL_HUMAN
        for name in self.classes[head]:
            if _key(name) == _key(label):
                return name
        raise ValueError(f"'{label}' is not an ontology {head} class; expected one of {list(self.classes[head])}")

    def from_surface(self, head: str, text: str) -> str:
        """Map an annotated span (e.g. 'switch on', 'light', 'turn_on') onto its class."""
        key = _key(text)
        try:
            return self.canonical(head, key)
        except ValueError:
            pass
        legacy = self.raw.get("legacy_aliases", {}).get(head, {})
        for mapping in (self.synonyms.get(head, {}), legacy):
            for name, words in mapping.items():
                if key in {_key(word) for word in words}:
                    return name
        raise ValueError(f"span '{text}' has no {head} class; add it to the ontology synonyms")

    def surface_index(self, head: str) -> list[tuple[str, str]]:
        """(phrase, class) pairs sorted longest phrase first for lexical matching."""
        pairs = [(_key(name), name) for name in self.classes[head] if name != NONE]
        pairs += [(_key(word), name) for name, words in self.synonyms.get(head, {}).items() for word in words]
        return sorted(set(pairs), key=lambda item: (-len(item[0]), item[0]))


def check_table1(ontology: Ontology) -> None:
    for head, size in EXPECTED_SIZES.items():
        if len(ontology.classes[head]) != size:
            raise ValueError(f"{head} has {len(ontology.classes[head])} classes; Table 1 lists {size}")
