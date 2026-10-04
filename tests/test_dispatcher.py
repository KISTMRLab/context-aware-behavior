from context_behavior.data import EntitySpan, Record, validate_record
from context_behavior.dispatcher import ActionDispatcher


def test_dispatcher_checks_scene_affordances():
    dispatcher = ActionDispatcher({"objects": [{"id": "lamp", "affordances": ["turn_on", "turn_off"]}]})
    accepted = dispatcher.dispatch("action", {"action": "turn_on", "target": "lamp"})
    assert accepted.accepted and accepted.command["target"] == "lamp"
    assert not dispatcher.dispatch("action", {"action": "open", "target": "lamp"}).accepted
    assert not dispatcher.dispatch("action", {"action": "turn_on", "target": "window"}).accepted
    assert not dispatcher.dispatch("action", {"action": "turn_on"}).accepted


def test_dataset_contract_requires_groundable_action():
    record = Record("open the drawer", "action", (EntitySpan(0, 4, "action", "open"), EntitySpan(9, 15, "target", "drawer")))
    validate_record(record)

