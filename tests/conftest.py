from pathlib import Path

import pytest

from context_behavior.data import load_records
from context_behavior.tiny import make_tiny_backbone

STARTER = Path(__file__).resolve().parents[1] / "src" / "context_behavior" / "resources" / "starter"


@pytest.fixture(scope="session")
def starter_records():
    return load_records(STARTER / "train.jsonl"), load_records(STARTER / "val.jsonl")


@pytest.fixture(scope="session")
def tiny_backbone(tmp_path_factory, starter_records):
    train, val = starter_records
    return make_tiny_backbone(tmp_path_factory.mktemp("tiny-bert"), [r.text for r in train + val])


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, tiny_backbone, starter_records):
    from context_behavior.train import train_model

    train, val = starter_records
    output = tmp_path_factory.mktemp("checkpoint") / "model"
    metrics = train_model(train, val, str(tiny_backbone), output, epochs=3, log=lambda *_: None)
    return output, metrics
