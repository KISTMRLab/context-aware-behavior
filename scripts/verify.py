"""Train, reload, infer, and dispatch through the production CLI offline."""
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from context_behavior.dispatcher import ActionDispatcher


def run(*arguments: str) -> str:
    environment = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run([sys.executable, *arguments], check=True, text=True,
                               capture_output=True, env=environment)
    print(completed.stdout, end="")
    return completed.stdout


def entity(text: str, value: str, kind: str) -> dict:
    start = text.index(value)
    return {"start": start, "end": start + len(value), "type": kind, "value": value}


def main():
    root = Path("outputs/verify").resolve(); root.mkdir(parents=True, exist_ok=True)
    backbone = root / "tiny-backbone"; backbone.mkdir(exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "please", "open", "close",
             "the", "drawer", "hello", "there", "how", "are", "you"]
    (backbone / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(backbone / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(backbone)
    BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=4, intermediate_size=64)).save_pretrained(backbone)

    texts = ["please open the drawer", "please close the drawer", "hello there", "how are you"]
    records = []
    for text in texts[:2]:
        action = "open" if "open" in text else "close"
        records.append({"text": text, "intent": "action",
                        "entities": [entity(text, action, "action"), entity(text, "drawer", "target")]})
    records.extend({"text": text, "intent": "conversation", "entities": []} for text in texts[2:])
    dataset = root / "train.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    model_dir = root / "model"
    run("-m", "context_behavior.train", "--train", str(dataset), "--output", str(model_dir),
        "--backbone", str(backbone), "--epochs", "2", "--batch-size", "2")

    scene = root / "scene.json"
    scene.write_text(json.dumps({"objects": [{"id": "drawer", "affordances": ["open", "close"]}]}), encoding="utf-8")
    raw = run("-m", "context_behavior.infer", "--model", str(model_dir), "--scene", str(scene),
              "please open the drawer")
    prediction = json.loads(raw)
    assert prediction["intent"] in {"action", "conversation"}
    assert isinstance(prediction["entities"], dict)
    assert set(prediction["dispatch"]) == {"accepted", "route", "reason", "command"}
    assert all(torch.isfinite(value).all() for value in torch.load(model_dir / "model.pt", weights_only=True).values())
    (root / "prediction.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")

    # Random tiny weights need not predict correctly; verify grounding independently with a known-valid parse.
    known = ActionDispatcher.load(scene).dispatch("action", {"action": "open", "target": "drawer"})
    assert known.accepted and known.command == {"actor": "virtual_human", "action": "open", "target": "drawer"}
    (root / "known-valid-dispatch.json").write_text(json.dumps(asdict(known), indent=2), encoding="utf-8")
    print(f"verification passed: two training epochs, checkpoint reload, prediction schema, and grounded dispatch -> {root}")


if __name__ == "__main__":
    main()
