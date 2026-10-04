from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ENTITY_TYPES = ("action", "position", "target")
INTENTS = ("conversation", "action")


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    type: str
    value: str


@dataclass(frozen=True)
class Record:
    text: str
    intent: str
    entities: tuple[EntitySpan, ...]


def load_records(path: str | Path) -> list[Record]:
    records: list[Record] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            record = Record(str(raw["text"]), str(raw["intent"]), tuple(EntitySpan(**item) for item in raw.get("entities", [])))
            validate_record(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        records.append(record)
    if not records:
        raise ValueError("Dataset is empty")
    return records


def validate_record(record: Record) -> None:
    if record.intent not in INTENTS:
        raise ValueError(f"intent must be one of {INTENTS}")
    occupied: list[tuple[int, int]] = []
    for entity in record.entities:
        if entity.type not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type: {entity.type}")
        if not 0 <= entity.start < entity.end <= len(record.text):
            raise ValueError(f"invalid span {entity.start}:{entity.end}")
        if record.text[entity.start:entity.end].casefold() != entity.value.casefold():
            raise ValueError(f"span text does not match value '{entity.value}'")
        if any(entity.start < end and start < entity.end for start, end in occupied):
            raise ValueError("entity spans overlap")
        occupied.append((entity.start, entity.end))
    if record.intent == "conversation" and record.entities:
        raise ValueError("conversation records must have no action entities")
    if record.intent == "action":
        present = {entity.type for entity in record.entities}
        if not {"action", "target"}.issubset(present):
            raise ValueError("action records require action and target spans")


def build_label_maps(records: list[Record]) -> dict[str, list[str]]:
    maps = {kind: {"O"} for kind in ENTITY_TYPES}
    for record in records:
        for entity in record.entities:
            maps[entity.type].add(entity.value.casefold().replace(" ", "_"))
    return {kind: ["O", *sorted(values - {"O"})] for kind, values in maps.items()}


def encode_record(record: Record, tokenizer, label_maps: dict[str, list[str]], max_length: int):
    encoded = tokenizer(record.text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    labels = {kind: [-100 if start == end else 0 for start, end in offsets] for kind in ENTITY_TYPES}
    for entity in record.entities:
        label = entity.value.casefold().replace(" ", "_")
        label_id = label_maps[entity.type].index(label)
        matched = False
        for token_index, (start, end) in enumerate(offsets):
            if start < entity.end and entity.start < end:
                labels[entity.type][token_index] = label_id
                matched = True
        if not matched:
            raise ValueError(f"Entity '{entity.value}' was truncated; raise --max-length")
    encoded["intent_label"] = INTENTS.index(record.intent)
    encoded["entity_labels"] = labels
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the context behavior JSONL contract")
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("path")
    args = parser.parse_args()
    records = load_records(args.path)
    print(json.dumps({"records": len(records), "labels": build_label_maps(records)}, indent=2))


if __name__ == "__main__":
    main()

