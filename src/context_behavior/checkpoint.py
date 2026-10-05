"""Self-contained checkpoints: backbone (config + weights), tokenizer, ontology and head weights."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from .model import ContextClassifier
from .ontology import Ontology

FORMAT = "context-behavior/table1-v1"


def save_checkpoint(model: ContextClassifier, tokenizer, ontology: Ontology, output: str | Path,
                    config: dict, metrics: dict | None = None) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(destination / "backbone")
    tokenizer.save_pretrained(destination / "tokenizer")
    (destination / "ontology.json").write_text(json.dumps(ontology.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(model.heads.state_dict(), destination / "heads.pt")
    (destination / "config.json").write_text(json.dumps({"format": FORMAT, **config}, indent=2), encoding="utf-8")
    if metrics is not None:
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return destination


def is_checkpoint(path: str | Path) -> bool:
    folder = Path(path)
    return (folder / "heads.pt").is_file() and (folder / "config.json").is_file()


def load_checkpoint(path: str | Path):
    folder = Path(path)
    if not is_checkpoint(folder):
        if (folder / "model.pt").is_file():
            raise ValueError(f"{folder} is a legacy token-tagging checkpoint; retrain it with context-behavior-train "
                             "on Table 1 class labels (legacy span JSONL is converted automatically)")
        raise ValueError(f"{folder} does not contain heads.pt/config.json from context-behavior-train")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format {config.get('format')!r}")
    ontology = Ontology.load(folder / "ontology.json")
    tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer", use_fast=True, local_files_only=True)
    backbone = AutoModel.from_pretrained(folder / "backbone", local_files_only=True)
    model = ContextClassifier(backbone, ontology, config.get("head_hidden", 256), config.get("dropout", 0.1),
                              freeze_backbone=True)
    model.heads.load_state_dict(torch.load(folder / "heads.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, tokenizer, ontology, config
