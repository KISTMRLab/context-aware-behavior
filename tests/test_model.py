import json

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from context_behavior.checkpoint import load_checkpoint
from context_behavior.model import ContextClassifier
from context_behavior.ontology import HEADS, Ontology


def test_forward_has_one_head_per_table1_category(tiny_backbone):
    ontology = Ontology.load()
    tokenizer = AutoTokenizer.from_pretrained(tiny_backbone)
    model = ContextClassifier(AutoModel.from_pretrained(tiny_backbone), ontology)
    batch = tokenizer(["open the window", "hello"], padding=True, return_tensors="pt")
    labels = {h: torch.zeros(2, dtype=torch.long) for h in HEADS}
    output = model(**batch, labels=labels)
    assert {h: output["logits"][h].shape[-1] for h in HEADS} == {"subject": 2, "action": 14, "position": 6, "target": 11}
    expected = sum(torch.nn.functional.cross_entropy(output["logits"][h], labels[h]) for h in HEADS)
    assert torch.allclose(output["loss"], expected)
    output["loss"].backward()
    assert all(p.grad is None for p in model.backbone.parameters())  # frozen BERT
    assert all(p.grad is not None for p in model.heads.parameters() if p.requires_grad)


def test_training_saves_a_self_contained_checkpoint(tiny_checkpoint):
    folder, metrics = tiny_checkpoint
    for name in ("backbone/config.json", "tokenizer/vocab.txt", "ontology.json", "heads.pt", "config.json", "metrics.json"):
        assert (folder / name).is_file(), name
    assert metrics["val_examples"] > 0 and set(metrics["val"]["accuracy"]) == set(HEADS)
    model, tokenizer, ontology, config = load_checkpoint(folder)
    assert config["backbone_source"] != str(folder) and ontology.classes["action"][3] == "Close"
    assert all(torch.isfinite(v).all() for v in model.heads.state_dict().values())


def test_interpreter_prediction_is_consistent(tiny_checkpoint):
    from context_behavior.infer import ModelInterpreter

    prediction = ModelInterpreter(tiny_checkpoint[0]).predict("Please open the drawer")
    ontology = Ontology.load()
    for head in HEADS:
        assert prediction[head] in ontology.classes[head]
    if prediction["subject"] == "None":
        assert prediction["action"] == prediction["position"] == prediction["target"] == "None"


def test_legacy_checkpoint_is_reported(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"x")
    (tmp_path / "config.json").write_text(json.dumps({"backbone": "x", "label_maps": {}}))
    with pytest.raises(ValueError, match="legacy"):
        load_checkpoint(tmp_path)
