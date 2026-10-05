import json
from pathlib import Path

import pytest

from context_behavior.demo import parse_rules
from context_behavior.rules import RuleInterpreter
from context_behavior.scene import BehaviorPlanner

SCENE = json.loads((Path(__file__).resolve().parents[1] / "demo" / "scene.json").read_text())
RULES = RuleInterpreter()


def classes(text):
    p = RULES.predict(text)
    return p["subject"], p["action"], p["position"], p["target"]


@pytest.mark.parametrize("text, expected", [
    ("Please open the drawer", ("Virtual Human", "Open", "None", "Drawer")),
    ("Don't open the window", ("None", "None", "None", "None")),
    ("Turn the lamp off, it is on", ("Virtual Human", "Turn off", "None", "Lamp")),
    ("Switch on the light", ("Virtual Human", "Turn on", "None", "Lamp")),
    ("Put the pillow on the bed", ("Virtual Human", "Put", "On", "Bed")),
    ("Lay on the bed", ("Virtual Human", "Lay", "On", "Bed")),
    ("Go sit on the chair", ("Virtual Human", "Sit", "On", "Chair")),
    ("Walk to the window", ("Virtual Human", "Walk", "To", "Window")),
    ("Is the lamp on?", ("None", "None", "None", "None")),
    ("I'm cold, close the window", ("Virtual Human", "Close", "None", "Window")),
    ("I need something from the drawer, open it", ("Virtual Human", "Open", "None", "Drawer")),
    ("Stand up", ("Virtual Human", "Stand up", "None", "None")),
])
def test_rule_fallback_audit_inputs(text, expected):
    assert classes(text) == expected


def test_audit_failures_now_ground_correctly():
    planner = BehaviorPlanner(SCENE)
    assert RULES.predict("Don't open the window")["negated"]
    assert planner.plan(RULES.predict("Don't open the window")).route == "conversation"
    lie = planner.plan(RULES.predict("Lay on the bed"))
    assert lie.accepted and lie.command["behavior"] == "lie_on"
    put = BehaviorPlanner(SCENE).plan(RULES.predict("Put the pillow on the bed"))
    assert put.accepted and next(s for s in put.steps if s["op"] == "place")["object"] == "pillow"


def test_parse_rules_compatibility_helper():
    assert parse_rules("How are you?") == ("conversation", {})
    intent, entities = parse_rules("Please close the missing cabinet")
    assert intent == "action" and entities == {"action": "Close"}
