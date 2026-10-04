from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DispatchResult:
    accepted: bool
    route: str
    reason: str
    command: dict[str, str] | None = None


class ActionDispatcher:
    """Ground predictions in a declared scene instead of blindly invoking animations."""

    def __init__(self, scene: dict):
        self.objects = {item["id"]: item for item in scene.get("objects", [])}

    @classmethod
    def load(cls, path: str | Path) -> "ActionDispatcher":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def dispatch(self, intent: str, entities: dict[str, str]) -> DispatchResult:
        if intent == "conversation":
            return DispatchResult(True, "conversation", "route to the configured dialogue adapter")
        action, target = entities.get("action"), entities.get("target")
        if not action or not target:
            return DispatchResult(False, "action", "action and target entities are required")
        obj = self.objects.get(target)
        if obj is None:
            return DispatchResult(False, "action", f"target '{target}' is not present in the scene")
        if action not in obj.get("affordances", []):
            return DispatchResult(False, "action", f"target '{target}' does not support '{action}'")
        command = {"actor": "virtual_human", "action": action, "target": target}
        if entities.get("position"):
            command["position"] = entities["position"]
        return DispatchResult(True, "action", "scene and affordance checks passed", command)

