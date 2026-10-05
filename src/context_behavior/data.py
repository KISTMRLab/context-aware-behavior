"""Sentence-level class labels over the Table 1 ontology.

Each JSONL row is ``{"text", "subject", "action", "position", "target"}`` with exact
ontology class names; ``None`` (or an omitted key) is the paper's None class.
Rows in the earlier span format (``intent`` + character-offset ``entities``) are
converted on load: each span's text is mapped onto a class through the ontology
synonym lists, and ``context-behavior-data convert`` rewrites such files.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .ontology import CONVERSATION, ENTITY_HEADS, HEADS, NONE, VIRTUAL_HUMAN, Ontology


@dataclass(frozen=True)
class Record:
    text: str
    subject: str
    action: str = NONE
    position: str = NONE
    target: str = NONE

    def labels(self) -> dict[str, str]:
        return {head: getattr(self, head) for head in HEADS}


def is_legacy(raw: dict) -> bool:
    return "intent" in raw and "subject" not in raw


def convert_legacy(raw: dict, ontology: Ontology) -> dict:
    """Old span rows -> class rows. Span values become classes via the synonym map."""
    text = str(raw["text"])
    intent = str(raw["intent"]).casefold()
    if intent not in {"conversation", "action"}:
        raise ValueError("legacy intent must be 'conversation' or 'action'")
    row = {"text": text, "subject": CONVERSATION if intent == "conversation" else VIRTUAL_HUMAN,
           "action": NONE, "position": NONE, "target": NONE}
    for entity in raw.get("entities", []):
        kind = entity["type"]
        if kind not in ENTITY_HEADS:
            raise ValueError(f"unknown entity type: {kind}")
        start, end = int(entity["start"]), int(entity["end"])
        if not 0 <= start < end <= len(text):
            raise ValueError(f"invalid span {start}:{end}")
        value = str(entity.get("value", text[start:end]))
        if text[start:end].casefold() != value.casefold():
            raise ValueError(f"span text does not match value '{value}'")
        row[kind] = ontology.from_surface(kind, value)
    return row


def make_record(raw: dict, ontology: Ontology) -> Record:
    if is_legacy(raw):
        raw = convert_legacy(raw, ontology)
    text = str(raw["text"]).strip()
    if not text:
        raise ValueError("text is empty")
    labels = {head: ontology.canonical(head, raw.get(head)) for head in HEADS}
    record = Record(text, **labels)
    validate_record(record, ontology)
    return record


def validate_record(record: Record, ontology: Ontology) -> None:
    for head in HEADS:
        if getattr(record, head) not in ontology.classes[head]:
            raise ValueError(f"{head} '{getattr(record, head)}' is not in the ontology")
    if record.subject == CONVERSATION:
        if any(getattr(record, head) != NONE for head in ENTITY_HEADS):
            raise ValueError("conversation (Subject None) rows must have Action/Position/Target None")
        return
    if record.action == NONE:
        raise ValueError("Virtual Human rows need an Action other than None")
    if record.target == NONE and record.action not in ontology.target_optional_actions:
        raise ValueError(f"Action '{record.action}' needs a Target; target-less actions are "
                         f"{sorted(ontology.target_optional_actions)}")


def load_records(path: str | Path, ontology: Ontology | None = None) -> list[Record]:
    ontology = ontology or Ontology.load()
    records: list[Record] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(make_record(json.loads(line), ontology))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records


def write_records(records: list[Record], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def label_counts(records: list[Record]) -> dict[str, dict[str, int]]:
    return {head: dict(sorted(Counter(getattr(r, head) for r in records).items())) for head in HEADS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or convert context-behavior JSONL (Table 1 class labels)")
    parser.add_argument("command", choices=["validate", "convert"])
    parser.add_argument("path")
    parser.add_argument("output", nargs="?", help="convert: destination JSONL in the class-label format")
    parser.add_argument("--ontology", help="Ontology JSON (default: bundled Table 1 ontology)")
    args = parser.parse_args()
    ontology = Ontology.load(args.ontology)
    records = load_records(args.path, ontology)
    if args.command == "convert":
        if not args.output:
            parser.error("convert needs an output path")
        write_records(records, args.output)
    print(json.dumps({"records": len(records), "labels": label_counts(records)}, indent=2))


if __name__ == "__main__":
    main()
