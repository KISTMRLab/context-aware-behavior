"""Train the joint SC/EC heads on Table 1 class labels and save a self-contained checkpoint."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from .checkpoint import save_checkpoint
from .data import Record, load_records
from .model import ContextClassifier
from .ontology import HEADS, Ontology

DEFAULT_BACKBONE = "google-bert/bert-base-uncased"
RESOURCES = Path(__file__).resolve().parent / "resources"
STARTER_TRAIN = RESOURCES / "starter" / "train.jsonl"
STARTER_VAL = RESOURCES / "starter" / "val.jsonl"


def _labels(records: list[Record], ontology: Ontology) -> dict[str, torch.Tensor]:
    return {head: torch.tensor([ontology.index(head, getattr(r, head)) for r in records]) for head in HEADS}


def _encode(tokenizer, texts: list[str], max_length: int):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


@torch.no_grad()
def _features(model: ContextClassifier, tokenizer, texts: list[str], max_length: int, batch_size: int, device) -> torch.Tensor:
    model.eval()
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = {k: v.to(device) for k, v in _encode(tokenizer, texts[start:start + batch_size], max_length).items()}
        rows.append(model.features(**batch).cpu())
    return torch.cat(rows)


@torch.no_grad()
def evaluate(model: ContextClassifier, features: torch.Tensor, labels: dict[str, torch.Tensor]) -> dict:
    model.eval()
    logits = model.classify(features)
    loss = float(model.joint_loss(logits, labels))
    correct = {head: logits[head].argmax(-1) == labels[head] for head in HEADS}
    exact = torch.stack(list(correct.values())).all(0)
    return {"loss": round(loss, 4), "exact_match": round(float(exact.float().mean()), 4),
            "accuracy": {head: round(float(value.float().mean()), 4) for head, value in correct.items()},
            "examples": int(features.shape[0])}


def train_model(train_records: list[Record], val_records: list[Record] | None, backbone: str, output: str | Path, *,
                ontology: Ontology | None = None, epochs: int = 60, batch_size: int = 32, learning_rate: float = 1e-3,
                max_length: int = 64, head_hidden: int = 256, dropout: float = 0.1, fine_tune_backbone: bool = False,
                seed: int = 13, log=print) -> dict:
    ontology = ontology or Ontology.load()
    random.seed(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    encoder = AutoModel.from_pretrained(backbone)
    model = ContextClassifier(encoder, ontology, head_hidden, dropout, freeze_backbone=not fine_tune_backbone).to(device)
    train_labels = _labels(train_records, ontology)
    val_labels = _labels(val_records, ontology) if val_records else None
    train_texts = [r.text for r in train_records]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=0.01)

    cached = None
    if not fine_tune_backbone:
        # The backbone is frozen, so its sentence features are computed once and the
        # four heads train on them; this is numerically the same as re-encoding each epoch.
        cached = _features(model, tokenizer, train_texts, max_length, 64, device)
    val_features = (_features(model, tokenizer, [r.text for r in val_records], max_length, 64, device)
                    if val_records else None)
    best, best_state, history = None, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_records))
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            labels = {head: value[index].to(device) for head, value in train_labels.items()}
            if cached is not None:
                logits = model.classify(cached[index].to(device))
                loss = model.joint_loss(logits, labels)
            else:
                batch = {k: v.to(device) for k, v in _encode(tokenizer, [train_texts[i] for i in index], max_length).items()}
                loss = model(**batch, labels=labels)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        row = {"epoch": epoch, "train_loss": round(total / len(order), 4)}
        if val_features is not None:
            if fine_tune_backbone:
                val_features = _features(model, tokenizer, [r.text for r in val_records], max_length, 64, device)
            row["val"] = evaluate(model, val_features.to(device), {k: v.to(device) for k, v in val_labels.items()})
            if best is None or row["val"]["loss"] < best["val"]["loss"]:
                best, best_state = row, copy.deepcopy(model.state_dict())
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 6) == 0:
            log(json.dumps(row))
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    metrics = {"backbone": str(backbone), "fine_tuned_backbone": fine_tune_backbone, "epochs": epochs,
               "train_examples": len(train_records), "val_examples": len(val_records or []),
               "best_epoch": best["epoch"] if best else epochs, "val": best["val"] if best else None,
               "training_seconds": round(elapsed, 1), "device": str(device),
               "note": "Informational only: accuracy on the authored starter validation split, not a paper benchmark."}
    save_checkpoint(model, tokenizer, ontology, output, {"backbone_source": str(backbone), "max_length": max_length,
                                                         "head_hidden": head_hidden, "dropout": dropout,
                                                         "fine_tuned_backbone": fine_tune_backbone}, metrics)
    log(json.dumps({"saved": str(output), **{k: metrics[k] for k in ("training_seconds", "best_epoch", "val")}}))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", help="Training JSONL (default: bundled authored starter train split)")
    parser.add_argument("--val", help="Validation JSONL (default: starter val split when --train is omitted)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help="Hugging Face id or local folder of a BERT-family encoder")
    parser.add_argument("--ontology", help="Ontology JSON (default: bundled Table 1 ontology)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--head-hidden", type=int, default=256, help="0 makes each classifier a single linear layer")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--fine-tune-backbone", action="store_true", help="Extension: update BERT instead of freezing it")
    args = parser.parse_args()
    ontology = Ontology.load(args.ontology)
    train_path = Path(args.train) if args.train else STARTER_TRAIN
    val_path = Path(args.val) if args.val else (None if args.train else STARTER_VAL)
    train_model(load_records(train_path, ontology), load_records(val_path, ontology) if val_path else None,
                args.backbone, args.output, ontology=ontology, epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, max_length=args.max_length, head_hidden=args.head_hidden,
                dropout=args.dropout, fine_tune_backbone=args.fine_tune_backbone, seed=args.seed)


if __name__ == "__main__":
    main()
