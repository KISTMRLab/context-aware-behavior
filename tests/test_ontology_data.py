import json
from pathlib import Path

import pytest

from context_behavior.data import Record, convert_legacy, load_records, make_record, validate_record
from context_behavior.ontology import Ontology, check_table1

STARTER = Path(__file__).resolve().parents[1] / "src" / "context_behavior" / "resources" / "starter"
ONTOLOGY = Ontology.load()


def test_table1_classes_exactly():
    check_table1(ONTOLOGY)
    assert ONTOLOGY.classes["subject"] == ("None", "Virtual Human")
    assert ONTOLOGY.classes["action"] == ("None", "Walk", "Open", "Close", "Sit", "Stand up", "Turn on", "Turn off",
                                          "Lay", "Run", "Idle", "Bring", "Hold", "Put")
    assert ONTOLOGY.classes["position"] == ("None", "Left", "Right", "In", "On", "To")
    assert ONTOLOGY.classes["target"] == ("None", "Floor", "Chair", "Drawer", "Bed", "Lamp", "Window", "Curtain",
                                          "Object", "Pillow", "Switch")


def test_paraphrases_map_to_one_class():
    assert ONTOLOGY.from_surface("action", "switch on") == ONTOLOGY.from_surface("action", "turn on") == "Turn on"
    assert ONTOLOGY.from_surface("action", "turn_on") == "Turn on"
    assert ONTOLOGY.from_surface("target", "light") == ONTOLOGY.from_surface("target", "lamp") == "Lamp"
    assert ONTOLOGY.from_surface("action", "sit_on") == "Sit"
    assert ONTOLOGY.canonical("subject", "conversation") == "None"
    with pytest.raises(ValueError):
        ONTOLOGY.from_surface("target", "spaceship")


@pytest.mark.parametrize("action", ["Walk", "Run", "Idle", "Stand up"])
def test_target_less_actions_are_valid(action):
    validate_record(Record("do it", "Virtual Human", action), ONTOLOGY)


def test_invalid_rows_are_rejected():
    with pytest.raises(ValueError, match="needs a Target"):
        validate_record(Record("open", "Virtual Human", "Open"), ONTOLOGY)
    with pytest.raises(ValueError, match="conversation"):
        validate_record(Record("hello lamp", "None", "None", "None", "Lamp"), ONTOLOGY)
    with pytest.raises(ValueError, match="Action other than None"):
        validate_record(Record("the bed", "Virtual Human", "None", "None", "Bed"), ONTOLOGY)
    with pytest.raises(ValueError, match="not an ontology action class"):
        make_record({"text": "jump", "subject": "Virtual Human", "action": "Jump"}, ONTOLOGY)


def test_legacy_span_rows_convert_through_synonyms(tmp_path):
    text = "Please switch on the light"
    legacy = {"text": text, "intent": "action", "entities": [
        {"start": 7, "end": 16, "type": "action", "value": "switch on"},
        {"start": 21, "end": 26, "type": "target", "value": "light"}]}
    assert convert_legacy(legacy, ONTOLOGY) == {"text": text, "subject": "Virtual Human", "action": "Turn on",
                                                "position": "None", "target": "Lamp"}
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(legacy) + "\n" + json.dumps({"text": "Hi", "intent": "conversation", "entities": []}) + "\n")
    records = load_records(path, ONTOLOGY)
    assert [r.subject for r in records] == ["Virtual Human", "None"]


def test_starter_dataset_covers_every_class_without_leakage():
    train = load_records(STARTER / "train.jsonl", ONTOLOGY)
    val = load_records(STARTER / "val.jsonl", ONTOLOGY)
    assert len(train) + len(val) >= 300 and len(val) >= 40
    assert not {r.text.casefold() for r in train} & {r.text.casefold() for r in val}
    for head, classes in ONTOLOGY.classes.items():
        assert set(classes) == {getattr(r, head) for r in train + val}, head
    texts = {r.text: r for r in train + val}
    assert texts["Don't open the window"].subject == "None"
    assert texts["Switch on the lamp"].action == texts["Turn on the lamp"].action == "Turn on"
    provenance = json.loads((STARTER / "provenance.json").read_text())
    assert "CC0" in provenance["license"] and provenance["train_examples"] == len(train)
