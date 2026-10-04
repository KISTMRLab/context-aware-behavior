from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import ENTITY_TYPES, INTENTS
from .dispatcher import ActionDispatcher
from .model import JointContextModel


def predict(model_dir: Path, text: str) -> tuple[str, dict[str, str]]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", use_fast=True)
    model = JointContextModel(config["backbone"], config["label_maps"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        output = model(**encoded)
    intent = INTENTS[int(output["intent_logits"][0].argmax())]
    entities: dict[str, str] = {}
    for kind in ENTITY_TYPES:
        probabilities = output["entity_logits"][kind][0].softmax(-1)
        token_scores, token_labels = probabilities.max(-1)
        candidates = [(float(token_scores[i]), int(token_labels[i]), offsets[i]) for i in range(len(offsets)) if int(token_labels[i]) != 0 and offsets[i][0] != offsets[i][1]]
        if candidates:
            _, label_id, _ = max(candidates)
            entities[kind] = config["label_maps"][kind][label_id]
    return intent, entities


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify text and validate an action against a scene")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("text")
    args = parser.parse_args()
    intent, entities = predict(Path(args.model), args.text)
    result = ActionDispatcher.load(args.scene).dispatch(intent, entities)
    print(json.dumps({"intent": intent, "entities": entities, "dispatch": vars(result)}, indent=2))


if __name__ == "__main__":
    main()

