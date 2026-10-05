import json
from pathlib import Path

import pytest

from context_behavior.dispatcher import ActionDispatcher
from context_behavior.scene import BehaviorPlanner

SCENE = json.loads((Path(__file__).resolve().parents[1] / "demo" / "scene.json").read_text())


def act(action, position="None", target="None", **extra):
    return {"subject": "Virtual Human", "action": action, "position": position, "target": target, **extra}


def ops(decision):
    return [step["op"] for step in decision.steps]


def test_sit_on_chair_maps_to_seated_behaviour_regression():
    planner = BehaviorPlanner(SCENE)
    decision = planner.plan(act("Sit", "On", "Chair"))
    assert decision.accepted and decision.command["behavior"] == "sit_on"
    assert decision.command["position"] == "On" and decision.command["object"] == "chair"
    sit = next(s for s in decision.steps if s["op"] == "sit")
    chair = next(o for o in SCENE["objects"] if o["id"] == "chair")
    assert sit["seat_height"] == chair["seat_height"] and sit["facing"] == chair["facing"]
    walk = next(s for s in decision.steps if s["op"] == "walk")
    assert walk["to"][0] > chair["position"][0]  # chair faces +x, so it is approached from its front
    assert planner.states()["agent"]["posture"] == "sit"
    standing = planner.plan(act("Stand up"))
    assert standing.accepted and ops(standing)[0] == "stand"


def test_position_defaults_and_lie_on_bed():
    decision = BehaviorPlanner(SCENE).plan(act("Lay", "None", "Bed"))
    assert decision.accepted and decision.command["position"] == "On"
    lie = next(s for s in decision.steps if s["op"] == "lie")
    assert lie["surface_height"] == 0.48 and lie["head_direction"] == [0, -1]


@pytest.mark.parametrize("prediction, reason", [
    (act("Sit", "In", "Drawer"), "does not support 'Sit'"),
    (act("Open", "None", "Lamp"), "does not support 'Open'"),
    (act("Sit", "Left", "Chair"), "accepts On, In"),
    (act("Open", "In", "Window"), "is not supported"),
    (act("Open"), "needs a Target"),
    (act("Turn on", "None", "Curtain"), "does not support"),
    (act("Hold", "None", "Bed"), "does not support 'Hold'"),
    (act("Put", "On", "Bed"), "nothing in hand"),
    (act("Walk", "To"), "to where"),
])
def test_unsupported_combinations_are_rejected_with_reason(prediction, reason):
    planner = BehaviorPlanner(SCENE)
    before = planner.states()
    decision = planner.plan(prediction)
    assert not decision.accepted and reason in decision.reason
    assert planner.states() == before


def test_target_less_actions_and_side_positions():
    planner = BehaviorPlanner(SCENE)
    left = planner.plan(act("Walk", "Left"))
    assert left.accepted and left.steps[-1]["to"][0] < SCENE["agent"]["position"][0]
    assert planner.plan(act("Run", "To", "Window")).steps[-2]["gait"] == "run"
    side = BehaviorPlanner(SCENE).plan(act("Walk", "Right", "Bed"))
    assert side.steps[-2]["to"][0] > 1.85
    assert BehaviorPlanner(SCENE).plan(act("Idle")).accepted


def test_open_close_toggle_state_and_hand_side():
    planner = BehaviorPlanner(SCENE)
    opened = planner.plan(act("Open", "Right", "Drawer"))
    assert opened.accepted and {"op": "state", "object": "drawer", "state": {"open": True}} in opened.steps
    assert next(s for s in opened.steps if s["op"] == "reach")["side"] == "right"
    again = planner.plan(act("Open", "None", "Drawer"))
    assert again.accepted and "already open" in again.notes[0]
    assert planner.plan(act("Turn off", "None", "Switch")).accepted
    assert planner.states()["objects"]["switch"]["on"] is False


def test_bring_hold_put_move_the_prop():
    planner = BehaviorPlanner(SCENE)
    bring = planner.plan(act("Bring", "None", "Pillow"))
    assert ["attach", "release", "walk", "face", "offer"] == ops(bring)[-5:]
    assert planner.states()["agent"]["holding"] == "pillow"
    put = planner.plan(act("Put", "In", "Drawer"))
    assert put.accepted and "state" in ops(put)  # the closed drawer is opened first
    place = next(s for s in put.steps if s["op"] == "place")
    assert place["object"] == "pillow" and place["inside"] == "drawer"
    assert planner.states()["agent"]["holding"] is None
    # Put without a held object picks up the object named in the sentence.
    put_book = planner.plan(act("Put", "On", "Chair", mentioned=["Object", "Chair"]))
    assert put_book.accepted and ops(put_book).count("attach") == 1
    assert planner.states()["objects"]["book"]["resting_on"] == "chair"


def test_conversation_route_and_legacy_dispatcher_api():
    planner = BehaviorPlanner(SCENE)
    assert planner.plan({"subject": "None"}).route == "conversation"
    dispatcher = ActionDispatcher({"objects": [{"id": "lamp", "affordances": ["turn_on", "turn_off"]}]})
    accepted = dispatcher.dispatch("action", {"action": "turn_on", "target": "lamp"})
    assert accepted.accepted and accepted.command["object"] == "lamp" and accepted.command["action"] == "Turn on"
    assert not dispatcher.dispatch("action", {"action": "open", "target": "lamp"}).accepted
    assert not dispatcher.dispatch("action", {"action": "turn_on", "target": "window"}).accepted
    assert not dispatcher.dispatch("action", {"action": "turn_on"}).accepted
    chair = ActionDispatcher({"objects": [{"id": "chair", "affordances": ["sit_on"], "x": 36, "z": 65}]})
    assert chair.dispatch("action", {"action": "sit", "position": "on", "target": "chair"}).command["behavior"] == "sit_on"
