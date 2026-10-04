from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from .data import ENTITY_TYPES, build_label_maps, encode_record, load_records
from .model import JointContextModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the joint context model")
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone", default="google-bert/bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--fine-tune-backbone", action="store_true")
    args = parser.parse_args()

    records = load_records(args.train)
    labels = build_label_maps(records)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    encoded = [encode_record(item, tokenizer, labels, args.max_length) for item in records]
    padder = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def collate(batch):
        # DataLoader returns references to the encoded records; copy before padding/pop
        # so subsequent epochs see the complete example again.
        batch = copy.deepcopy(batch)
        intent = torch.tensor([item.pop("intent_label") for item in batch])
        entity = {kind: [item["entity_labels"].pop(kind) for item in batch] for kind in ENTITY_TYPES}
        for item in batch:
            item.pop("entity_labels")
        padded = padder(batch)
        width = padded["input_ids"].shape[1]
        padded["intent_labels"] = intent
        padded["entity_labels"] = {
            kind: torch.tensor([row + [-100] * (width - len(row)) for row in rows]) for kind, rows in entity.items()
        }
        return padded

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointContextModel(args.backbone, labels, freeze_backbone=not args.fine_tune_backbone).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            batch = {key: ({k: v.to(device) for k, v in value.items()} if isinstance(value, dict) else value.to(device)) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output["loss"].backward()
            optimizer.step()
            total += float(output["loss"].detach())
        print(f"epoch={epoch + 1} loss={total / len(loader):.4f}")

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "model.pt")
    tokenizer.save_pretrained(destination / "tokenizer")
    (destination / "config.json").write_text(json.dumps({"backbone": args.backbone, "label_maps": labels}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
