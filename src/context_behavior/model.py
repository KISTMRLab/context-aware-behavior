from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .data import ENTITY_TYPES


class JointContextModel(nn.Module):
    """One shared encoder with a sentence head and three token heads."""

    def __init__(self, backbone_name: str, label_maps: dict[str, list[str]], freeze_backbone: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.label_maps = label_maps
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.intent_head = nn.Linear(hidden, 2)
        self.entity_heads = nn.ModuleDict({kind: nn.Linear(hidden, len(label_maps[kind])) for kind in ENTITY_TYPES})
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, intent_labels=None, entity_labels=None):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence = output.last_hidden_state
        intent_logits = self.intent_head(sequence[:, 0])
        entity_logits = {kind: head(sequence) for kind, head in self.entity_heads.items()}
        loss = None
        if intent_labels is not None and entity_labels is not None:
            loss = nn.functional.cross_entropy(intent_logits, intent_labels)
            for kind in ENTITY_TYPES:
                loss = loss + nn.functional.cross_entropy(
                    entity_logits[kind].reshape(-1, entity_logits[kind].shape[-1]),
                    entity_labels[kind].reshape(-1),
                    ignore_index=-100,
                )
        return {"loss": loss, "intent_logits": intent_logits, "entity_logits": entity_logits}
