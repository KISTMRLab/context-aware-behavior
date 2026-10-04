from context_behavior.demo import parse_rules
from context_behavior.dispatcher import ActionDispatcher


SCENE = {"objects": [{"id": "drawer", "affordances": ["open", "close"]},
                     {"id": "lamp", "affordances": ["turn_on", "turn_off"]}]}


def test_explicit_rules_are_grounded_before_dispatch():
    intent, entities = parse_rules("Please open the drawer", SCENE)
    result = ActionDispatcher(SCENE).dispatch(intent, entities)
    assert result.accepted and result.command["target"] == "drawer"
    intent, entities = parse_rules("Please open the lamp", SCENE)
    assert not ActionDispatcher(SCENE).dispatch(intent, entities).accepted


def test_conversation_and_missing_target():
    assert parse_rules("How are you?", SCENE) == ("conversation", {})
    intent, entities = parse_rules("Please close the missing cabinet", SCENE)
    assert intent == "action" and not ActionDispatcher(SCENE).dispatch(intent, entities).accepted
