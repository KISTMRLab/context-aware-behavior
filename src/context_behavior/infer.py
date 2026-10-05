"""Classify text with a trained checkpoint and plan the resulting behaviour in a scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Lock

import torch

from .checkpoint import load_checkpoint
from .ontology import CONVERSATION, ENTITY_HEADS, HEADS, NONE
from .rules import RuleInterpreter, mentioned_targets
from .scene import BehaviorPlanner


class ModelInterpreter:
    """Loads one checkpoint once and serves predictions thread-safely."""

    backend = "trained BERT checkpoint (frozen encoder + SC/EC heads)"

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model, self.tokenizer, self.ontology, self.config = load_checkpoint(self.model_dir)
        self.lock = Lock()
        self.rules = RuleInterpreter(self.ontology)

    def predict(self, text: str) -> dict:
        encoded = self.tokenizer(text, truncation=True, max_length=self.config.get("max_length", 64), return_tensors="pt")
        with self.lock, torch.no_grad():
            logits = self.model(**encoded)["logits"]
        result, confidence = {}, {}
        for head in HEADS:
            probabilities = logits[head][0].softmax(-1)
            index = int(probabilities.argmax())
            result[head] = self.ontology.classes[head][index]
            confidence[head] = round(float(probabilities[index]), 3)
        negated = False
        if result["subject"] == CONVERSATION:
            # The paper applies the entity classifiers only to action sentences.
            result.update({head: NONE for head in ENTITY_HEADS})
            negated = self.rules.predict(text)["negated"]  # only shapes the fallback reply wording
        return {**result, "confidence": confidence, "negated": negated,
                "mentioned": mentioned_targets(text, self.ontology), "backend": self.backend}


_CACHE: dict[Path, ModelInterpreter] = {}
_CACHE_LOCK = Lock()


def cached_interpreter(model_dir: str | Path) -> ModelInterpreter:
    key = Path(model_dir).resolve()
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = ModelInterpreter(key)
        return _CACHE[key]


def predict(model_dir: Path, text: str) -> dict:
    return cached_interpreter(model_dir).predict(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("text")
    args = parser.parse_args()
    prediction = predict(Path(args.model), args.text)
    decision = BehaviorPlanner.load(args.scene).plan(prediction)
    print(json.dumps({"prediction": prediction, "dispatch": decision.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
