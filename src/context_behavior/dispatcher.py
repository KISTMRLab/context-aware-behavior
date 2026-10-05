"""Compatibility wrapper: the earlier ``ActionDispatcher.dispatch(intent, entities)`` API over BehaviorPlanner."""
from __future__ import annotations

import json
from pathlib import Path

from .ontology import ENTITY_HEADS, Ontology
from .scene import BehaviorPlanner, Decision

DispatchResult = Decision


class ActionDispatcher:
    """Ground predictions in a declared scene instead of blindly invoking animations."""

    def __init__(self, scene: dict, ontology: Ontology | None = None):
        self.planner = BehaviorPlanner(scene, ontology)
        self.ontology = self.planner.ontology

    @classmethod
    def load(cls, path: str | Path) -> "ActionDispatcher":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _class(self, head: str, value: str | None) -> str | None:
        if value is None:
            return None
        if head == "target":
            match = next((o for o in self.planner.scene["objects"] if o["id"] == value), None)
            if match:
                return match["class"]
        try:
            return self.ontology.from_surface(head, value)
        except ValueError:
            return None

    def dispatch(self, intent: str, entities: dict[str, str], commit: bool = False) -> Decision:
        prediction = {"subject": intent, **{head: self._class(head, entities.get(head)) for head in ENTITY_HEADS}}
        if entities.get("target") and prediction["target"] is None:
            return Decision(False, "action", f"target '{entities['target']}' is not present in the scene")
        return self.planner.plan(prediction, commit=commit)
