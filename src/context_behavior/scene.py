"""Scene metadata and entity-combination -> behaviour planning.

Every object carries a name, an ontology Target class, a floor position (metres),
a facing direction (yaw in degrees; 0 faces +z, toward the camera/user) and an
affordance table ``{Action: [allowed Position classes]}``. The planner combines the
predicted Action, Position and Target with that metadata, rejects unsupported
combinations with a reason, and otherwise emits a step list that the browser
renderer executes (walk, face, reach, sit, lie, attach/place props, change state).
Left/Right are taken from the user's (camera) view.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ontology import CONVERSATION, NONE, Ontology

BEHAVIOR = {"Walk": "walk", "Run": "run", "Sit": "sit_on", "Lay": "lie_on", "Stand up": "stand_up", "Idle": "idle",
            "Open": "open", "Close": "close", "Turn on": "turn_on", "Turn off": "turn_off", "Bring": "bring",
            "Hold": "hold", "Put": "put"}
LOCOMOTION_POSITIONS = ["None", "To", "Left", "Right"]
SPEED = {"Walk": 1.1, "Run": 2.6}
BOUNDS = (-2.8, 2.8, -2.2, 1.9)
FLOOR = {"id": "floor", "name": "floor", "class": "Floor", "position": [0.0, 0.0], "size": [0, 0, 0], "virtual": True,
         "affordances": {"Walk": ["None", "To", "On"], "Run": ["None", "To", "On"], "Sit": ["On", "None"],
                         "Lay": ["On", "None"], "Put": ["On", "None", "Left", "Right"]}}
LEGACY_AFFORDANCES = {"open": ("Open", ["None"]), "close": ("Close", ["None"]), "turn_on": ("Turn on", ["None"]),
                      "turn_off": ("Turn off", ["None"]), "sit_on": ("Sit", ["On"]), "lie_on": ("Lay", ["On"]),
                      "move_to": ("Walk", ["To"]), "bring": ("Bring", ["None", "To"]), "hold": ("Hold", ["None"]),
                      "put": ("Put", ["On"])}


def _round(values):
    return [round(float(v), 3) for v in values]


def yaw_to(origin, point) -> float:
    return round(math.degrees(math.atan2(point[0] - origin[0], point[1] - origin[1])), 1)


def normalize_scene(scene: dict, ontology: Ontology) -> dict:
    """Accept the current metadata format and the earlier {id, affordances:[...], x, z} format."""
    result = {"user": scene.get("user", {"position": [0.0, 1.7]}),
              "agent": {"position": [0.0, 0.0], "facing": 0.0, "posture": "stand", "holding": None, "seat": None,
                        **scene.get("agent", {})},
              "objects": []}
    for raw in scene.get("objects", []):
        item = copy.deepcopy(raw)
        item.setdefault("name", item["id"].replace("_", " "))
        item["class"] = ontology.canonical("target", item.get("class") or ontology.from_surface("target", item["name"]))
        if "position" not in item:
            item["position"] = [((item.get("x", 50) - 50) / 16), ((item.get("z", 50) - 50) / 20)] if "x" in item else [0.0, 0.0]
        item.setdefault("size", [0.5, 0.6, 0.5])
        item.setdefault("facing", 0.0)
        item.setdefault("elevation", 0.0)
        affordances = item.get("affordances", {})
        if isinstance(affordances, list):
            converted = {}
            for name in affordances:
                action, positions = LEGACY_AFFORDANCES.get(name, (ontology.from_surface("action", name), ["None"]))
                converted.setdefault(action, [])
                converted[action] += [p for p in positions if p not in converted[action]]
            affordances = converted
        item["affordances"] = {ontology.canonical("action", k): [ontology.canonical("position", p) for p in v]
                               for k, v in affordances.items()}
        state = dict(item.get("state", {}))
        legacy = item.get("initial_state")  # earlier format: "open"/"close"/"turn_on"/"turn_off"/"idle"
        if "Open" in item["affordances"] and legacy:
            state.setdefault("open", legacy == "open")
        if "Turn on" in item["affordances"] and legacy:
            state.setdefault("on", legacy == "turn_on")
        if "Open" in item["affordances"]:
            state.setdefault("open", False)
        if "Turn on" in item["affordances"]:
            state.setdefault("on", False)
        if item.get("carryable"):
            state.setdefault("location", [*item["position"][:1], item["elevation"] + item["size"][1] / 2, item["position"][1]])
            state.setdefault("held", False)
        item["state"] = state
        result["objects"].append(item)
    return result


@dataclass
class Decision:
    accepted: bool
    route: str
    reason: str
    command: dict | None = None
    steps: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reply: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Rejected(Exception):
    pass


class BehaviorPlanner:
    def __init__(self, scene: dict, ontology: Ontology | None = None):
        self.ontology = ontology or Ontology.load()
        self.scene = normalize_scene(scene, self.ontology)
        self.floor = normalize_scene({"objects": [FLOOR]}, self.ontology)["objects"][0]

    @classmethod
    def load(cls, path: str | Path, ontology: Ontology | None = None) -> "BehaviorPlanner":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), ontology)

    # ----- state ---------------------------------------------------------------------------------------------
    def states(self) -> dict:
        return {"agent": copy.deepcopy(self.scene["agent"]),
                "objects": {o["id"]: copy.deepcopy(o["state"]) for o in self.scene["objects"]}}

    def public_scene(self) -> dict:
        return copy.deepcopy(self.scene)

    # ----- geometry ------------------------------------------------------------------------------------------
    @staticmethod
    def front(obj) -> tuple[float, float]:
        angle = math.radians(obj.get("facing", 0.0))
        return math.sin(angle), math.cos(angle)

    @staticmethod
    def extent(obj, direction) -> float:
        width, _, depth = obj["size"]
        return abs(direction[0]) * width / 2 + abs(direction[1]) * depth / 2

    def approach(self, obj, side: str = NONE, clearance: float = 0.42) -> list[float]:
        if obj.get("virtual"):
            return list(self.scene["agent"]["position"])
        x, z = obj["position"]
        if side in {"Left", "Right"}:
            direction = (-1.0, 0.0) if side == "Left" else (1.0, 0.0)
        else:
            if obj.get("approach") and side == NONE:
                return _round(obj["approach"])
            direction = self.front(obj)
        distance = self.extent(obj, direction) + clearance
        return self.clamp([x + direction[0] * distance, z + direction[1] * distance])

    def approach_near(self, support, point, clearance: float = 0.3) -> list[float]:
        """Stand at the support's front edge, opposite an item resting on or in it."""
        fx, fz = self.front(support)
        cx, cz = support["position"]
        offset = self.extent(support, (fx, fz)) + clearance - ((point[0] - cx) * fx + (point[1] - cz) * fz)
        return self.clamp([point[0] + fx * offset, point[1] + fz * offset])

    @staticmethod
    def clamp(point) -> list[float]:
        return _round([min(BOUNDS[1], max(BOUNDS[0], point[0])), min(BOUNDS[3], max(BOUNDS[2], point[1]))])

    def top(self, obj) -> float:
        return float(obj.get("surface_height", obj.get("elevation", 0.0) + obj["size"][1]))

    def handle(self, obj) -> list[float]:
        if obj.get("handle"):
            return _round(obj["handle"])
        fx, fz = self.front(obj)
        reach = self.extent(obj, (fx, fz))
        height = obj.get("elevation", 0.0) + obj["size"][1] * (0.5 if obj.get("elevation") else 0.8)
        return _round([obj["position"][0] + fx * reach, min(1.5, max(0.7, height)), obj["position"][1] + fz * reach])

    # ----- lookup --------------------------------------------------------------------------------------------
    def objects_of(self, cls: str) -> list[dict]:
        if cls == "Floor":
            return [self.floor]
        agent = self.scene["agent"]["position"]
        found = [o for o in self.scene["objects"] if o["class"] == cls]
        return sorted(found, key=lambda o: math.dist(agent, o["position"]))

    def nearest_with(self, action: str) -> dict | None:
        agent = self.scene["agent"]["position"]
        found = [o for o in self.scene["objects"] if action in o["affordances"]]
        return min(found, key=lambda o: math.dist(agent, o["position"])) if found else None

    def affordances(self, obj) -> dict[str, list[str]]:
        table = dict(obj["affordances"])
        for action, positions in (("Walk", LOCOMOTION_POSITIONS), ("Run", LOCOMOTION_POSITIONS), ("Stand up", ["None"])):
            table.setdefault(action, positions)
        return table

    # ----- planning ------------------------------------------------------------------------------------------
    def plan(self, prediction: dict, commit: bool = True) -> Decision:
        subject = self.ontology.canonical("subject", prediction.get("subject", prediction.get("intent")))
        if subject == CONVERSATION:
            reason = ("negated request: no action is taken; route to the dialogue client" if prediction.get("negated")
                      else "small talk: route to the dialogue client")
            return Decision(True, "conversation", reason)
        action = self.ontology.canonical("action", prediction.get("action"))
        position = self.ontology.canonical("position", prediction.get("position"))
        target = self.ontology.canonical("target", prediction.get("target"))
        if action == NONE:
            return Decision(False, "action", "no Action class was recognised; nothing to perform")
        saved = copy.deepcopy(self.scene)
        try:
            decision = self._plan(action, position, target, prediction.get("mentioned") or [])
        except Rejected as rejection:
            self.scene = saved
            return Decision(False, "action", str(rejection),
                            {"action": action, "position": position, "target": target})
        if not commit:
            self.scene = saved
        return decision

    def _plan(self, action: str, position: str, target: str, mentioned: list[str]) -> Decision:
        agent = self.scene["agent"]
        steps: list[dict] = []
        notes: list[str] = []
        obj = None
        if target == NONE:
            if action not in self.ontology.target_optional_actions:
                raise Rejected(f"'{action}' needs a Target (which object?)")
            if action in {"Sit", "Lay"}:
                obj = self.nearest_with(action) or self.floor
                notes.append(f"no target named; using the nearest object that affords {action}: {obj['name']}")
        else:
            candidates = self.objects_of(target)
            if not candidates:
                raise Rejected(f"there is no {target.lower()} in this room")
            supporting = [o for o in candidates if action in self.affordances(o) or
                          (action == "Put" and o.get("carryable"))]
            if not supporting:
                options = ", ".join(self.affordances(candidates[0])) or "nothing"
                raise Rejected(f"the {candidates[0]['name']} does not support '{action}' (it supports: {options})")
            obj = supporting[0]

        if obj is not None and not (action == "Put" and obj.get("carryable")):
            allowed = self.affordances(obj).get(action, [])
            if position == NONE and "None" not in allowed and allowed:
                position = allowed[0]
                notes.append(f"position defaulted to {position}")
            elif position not in allowed and not (position == NONE and not allowed):
                raise Rejected(f"'{action} {position} {obj['class']}' is not supported: {action} with the "
                               f"{obj['name']} accepts {', '.join(allowed) or 'no position'}")
        elif obj is None and position in {"In", "On"} and action != "Put":
            raise Rejected(f"'{action} {position}' needs a target object")

        handlers = {"Walk": self._walk_to, "Run": self._run_to, "Idle": self._idle, "Stand up": self._stand_up,
                    "Sit": self._sit_on, "Lay": self._lie_on, "Open": self._open, "Close": self._close,
                    "Turn on": self._turn_on, "Turn off": self._turn_off, "Bring": self._bring, "Hold": self._hold,
                    "Put": self._put}
        reply = handlers[action](obj, position, steps, notes, mentioned) or ""
        command = {"actor": "virtual_human", "behavior": BEHAVIOR[action], "action": action, "position": position,
                   "target": obj["class"] if obj else NONE, "object": obj["id"] if obj else None}
        return Decision(True, "action", "entity combination matches the scene metadata", command, steps, notes,
                        reply or f"{action} {position if position != NONE else ''} {obj['name'] if obj else ''}".strip())

    # ----- step helpers --------------------------------------------------------------------------------------
    def _ensure_standing(self, steps):
        agent = self.scene["agent"]
        if agent["posture"] != "stand":
            seat = next((o for o in self.scene["objects"] if o["id"] == agent.get("seat")), None)
            steps.append({"op": "stand"})
            agent["posture"], agent["seat"] = "stand", None
            if seat is not None:
                self._walk(self.approach(seat), steps, "Walk")

    def _walk(self, point, steps, gait="Walk"):
        agent = self.scene["agent"]
        point = self.clamp(point)
        if math.dist(agent["position"], point) > 0.05:
            agent["facing"] = yaw_to(agent["position"], point)
            steps.append({"op": "walk", "to": point, "speed": SPEED.get(gait, 1.1), "gait": gait.lower(),
                          "facing": agent["facing"]})
            agent["position"] = point

    def _face(self, steps, yaw=None, point=None):
        agent = self.scene["agent"]
        if point is not None:
            yaw = yaw_to(agent["position"], point)
        agent["facing"] = round(float(yaw) % 360, 1)
        steps.append({"op": "face", "yaw": agent["facing"]})

    def _go_to(self, obj, side, steps, gait="Walk"):
        self._ensure_standing(steps)
        steps.append({"op": "look", "object": obj["id"]})
        self._walk(self.approach(obj, side), steps, gait)
        self._face(steps, point=obj["position"])

    @staticmethod
    def _hand(position):
        return position.lower() if position in {"Left", "Right"} else "auto"

    # ----- behaviours ----------------------------------------------------------------------------------------
    def _locomote(self, obj, position, steps, gait):
        agent = self.scene["agent"]
        self._ensure_standing(steps)
        if obj is None or obj.get("virtual"):
            if position in {"Left", "Right"}:
                dx = -1.5 if position == "Left" else 1.5
                point = [agent["position"][0] + dx, agent["position"][1]]
            elif position == "To":
                raise Rejected(f"'{gait} To' needs a target (to where?)")
            else:
                point = [0.0, 0.6] if math.dist(agent["position"], [0.0, 0.6]) > 0.6 else [0.0, -0.8]
            self._walk(point, steps, gait)
            return f"{gait.lower()}ing {position.lower() if position != NONE else 'across the room'}"
        steps.append({"op": "look", "object": obj["id"]})
        self._walk(self.approach(obj, position if position in {"Left", "Right"} else NONE), steps, gait)
        self._face(steps, point=obj["position"])
        side = f"the {position.lower()} side of " if position in {"Left", "Right"} else ""
        return f"{gait.lower()}ing to {side}the {obj['name']}"

    def _walk_to(self, obj, position, steps, notes, mentioned):
        return self._locomote(obj, position, steps, "Walk")

    def _run_to(self, obj, position, steps, notes, mentioned):
        return self._locomote(obj, position, steps, "Run")

    def _idle(self, obj, position, steps, notes, mentioned):
        steps.append({"op": "idle"})
        return "relaxing"

    def _stand_up(self, obj, position, steps, notes, mentioned):
        if self.scene["agent"]["posture"] == "stand":
            notes.append("already standing")
            steps.append({"op": "idle"})
            return "already standing"
        self._ensure_standing(steps)
        return "standing up"

    def _sit_on(self, obj, position, steps, notes, mentioned):
        agent = self.scene["agent"]
        if agent["posture"] == "sit" and agent.get("seat") == obj["id"]:
            notes.append("already sitting there")
            return f"already sitting on the {obj['name']}"
        self._ensure_standing(steps)
        if obj.get("virtual"):
            steps.append({"op": "sit", "seat_height": 0.0, "position": agent["position"], "facing": agent["facing"]})
        else:
            fx, fz = self.front(obj)
            steps.append({"op": "look", "object": obj["id"]})
            self._walk(self.approach(obj), steps)
            edge = max(0.0, self.extent(obj, (fx, fz)) - 0.28)
            seat = _round([obj["position"][0] + fx * edge, obj["position"][1] + fz * edge])
            facing = float(obj.get("facing", 0.0))
            steps.append({"op": "face", "yaw": facing})
            steps.append({"op": "sit", "seat_height": float(obj.get("seat_height", self.top(obj))),
                          "position": seat, "facing": facing, "object": obj["id"]})
            agent["position"], agent["facing"] = seat, facing
        agent["posture"], agent["seat"] = "sit", obj["id"]
        return f"sitting {position.lower() if position != NONE else 'on'} the {obj['name']}"

    def _lie_on(self, obj, position, steps, notes, mentioned):
        agent = self.scene["agent"]
        self._ensure_standing(steps)
        if obj.get("virtual"):
            steps.append({"op": "lie", "surface_height": 0.0, "position": agent["position"], "head_direction": [-1, 0]})
        else:
            steps.append({"op": "look", "object": obj["id"]})
            self._walk(self.approach(obj), steps)
            self._face(steps, point=obj["position"])
            steps.append({"op": "lie", "surface_height": self.top(obj), "position": _round(obj["position"]),
                          "head_direction": obj.get("head_direction", [-1, 0]), "object": obj["id"]})
            agent["position"] = _round(obj["position"])
        agent["posture"], agent["seat"] = "lie", obj["id"]
        return f"lying on the {obj['name']}"

    def _toggle(self, obj, position, steps, notes, key, value, verb):
        if obj["state"].get(key) == value:
            notes.append(f"the {obj['name']} is already {verb}")
            steps.append({"op": "look", "object": obj["id"]})
            return f"the {obj['name']} is already {verb}"
        self._go_to(obj, NONE, steps)
        side = self._hand(position)
        steps.append({"op": "reach", "side": side, "object": obj["id"], "point": self.handle(obj)})
        steps.append({"op": "state", "object": obj["id"], "state": {key: value}})
        steps.append({"op": "release"})
        obj["state"][key] = value
        hand = f" with the {position.lower()} hand" if side != "auto" else ""
        return f"{verb} the {obj['name']}{hand}"

    def _open(self, obj, position, steps, notes, mentioned):
        return self._toggle(obj, position, steps, notes, "open", True, "open")

    def _close(self, obj, position, steps, notes, mentioned):
        return self._toggle(obj, position, steps, notes, "open", False, "closed")

    def _turn_on(self, obj, position, steps, notes, mentioned):
        return self._toggle(obj, position, steps, notes, "on", True, "on")

    def _turn_off(self, obj, position, steps, notes, mentioned):
        return self._toggle(obj, position, steps, notes, "on", False, "off")

    def _pick_up(self, obj, position, steps, notes):
        agent = self.scene["agent"]
        if agent.get("holding") == obj["id"]:
            notes.append(f"already holding the {obj['name']}")
            return
        if agent.get("holding"):
            raise Rejected(f"hands are full (holding the {agent['holding']}); ask me to put it down first")
        location = obj["state"].get("location", [obj["position"][0], 0.3, obj["position"][1]])
        holder = obj["state"].get("inside")
        if holder:
            container = next(o for o in self.scene["objects"] if o["id"] == holder)
            if not container["state"].get("open", True):
                self._toggle(container, NONE, steps, notes, "open", True, "open")
        self._ensure_standing(steps)
        steps.append({"op": "look", "point": location})
        support = next((o for o in self.scene["objects"] if o["id"] in {holder, obj["state"].get("resting_on")}), None)
        if obj.get("approach") and not obj["state"].get("moved"):
            stand_at = _round(obj["approach"])
        elif support is not None:
            stand_at = self.approach_near(support, [location[0], location[2]])
        else:
            stand_at = self.approach({**obj, "position": [location[0], location[2]], "approach": None})
        self._walk(stand_at, steps)
        self._face(steps, point=[location[0], location[2]])
        side = self._hand(position) if position in {"Left", "Right"} else "right"
        steps.append({"op": "reach", "side": side, "object": obj["id"], "point": _round(location)})
        steps.append({"op": "attach", "object": obj["id"], "side": side})
        steps.append({"op": "release"})
        agent["holding"] = obj["id"]
        obj["state"].update(held=True, inside=None, resting_on=None, moved=True)

    def _hold(self, obj, position, steps, notes, mentioned):
        self._pick_up(obj, position, steps, notes)
        return f"holding the {obj['name']}"

    def _bring(self, obj, position, steps, notes, mentioned):
        self._pick_up(obj, position, steps, notes)
        user = self.scene["user"]["position"]
        self._walk([user[0], user[1] - 0.9], steps)
        self._face(steps, point=user)
        agent = self.scene["agent"]["position"]
        steps.append({"op": "offer", "object": obj["id"], "point": _round([agent[0] + 0.1, 1.05, agent[1] + 0.45])})
        return f"bringing the {obj['name']} to you"

    def _put(self, obj, position, steps, notes, mentioned):
        agent = self.scene["agent"]
        carryables = {o["id"]: o for o in self.scene["objects"] if o.get("carryable")}
        if obj is not None and obj.get("carryable"):
            carried, destination = obj, None  # "put the pillow down"
        else:
            carried = carryables.get(agent.get("holding"))
            destination = obj
        if carried is None or agent.get("holding") != carried["id"]:
            hint = carried or next((o for cls in mentioned for o in carryables.values() if o["class"] == cls), None)
            if hint is None:
                raise Rejected("nothing in hand to put; ask me to bring or hold an object first")
            notes.append(f"picking up the {hint['name']} first")
            self._pick_up(hint, NONE, steps, notes)
            carried = hint
        if destination is None or destination.get("virtual"):
            if position in {"In"}:
                raise Rejected("'Put In' needs a container such as the drawer")
            fx, fz = math.sin(math.radians(agent["facing"])), math.cos(math.radians(agent["facing"]))
            dx = {"Left": -0.45, "Right": 0.45}.get(position, 0.0)
            point = _round([agent["position"][0] + fx * 0.45 + dx, carried["size"][1] / 2, agent["position"][1] + fz * 0.45])
            steps.append({"op": "reach", "side": "auto", "point": point})
            label = "on the floor"
        else:
            if position == "In" and not destination["state"].get("open", True):
                self._toggle(destination, NONE, steps, notes, "open", True, "open")
            side = position if position in {"Left", "Right"} else NONE
            self._go_to(destination, side, steps)
            x, z = destination["position"]
            if position == "In":
                fx, fz = self.front(destination)
                point = _round([x + fx * 0.1, destination.get("elevation", 0) + destination["size"][1] * 0.7, z + fz * 0.1])
            elif position in {"Left", "Right"}:
                offset = self.extent(destination, (1, 0)) + 0.3
                point = _round([x + (-offset if position == "Left" else offset), carried["size"][1] / 2, z])
            else:
                point = _round([x, self.top(destination) + carried["size"][1] / 2, z])
            steps.append({"op": "reach", "side": "auto", "point": point})
            label = f"{position.lower()} {'of ' if position in {'Left', 'Right'} else ''}the {destination['name']}"
        steps.append({"op": "place", "object": carried["id"], "point": point,
                      "inside": destination["id"] if destination is not None and position == "In" else None})
        steps.append({"op": "release"})
        agent["holding"] = None
        on_top = destination is not None and not destination.get("virtual") and position in {"On", NONE}
        carried["state"].update(held=False, location=point,
                                inside=destination["id"] if destination is not None and position == "In" else None,
                                resting_on=destination["id"] if on_top else None)
        return f"putting the {carried['name']} {label}"
