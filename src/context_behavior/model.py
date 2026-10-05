"""Frozen BERT with a sentence classifier (SC) and three entity classifiers (EC).

The SC reads the [CLS] vector and predicts Subject (None/small talk vs Virtual
Human). Each EC is a sentence-level multi-class head over the Table 1 classes for
Action, Position and Target. The paper does not state the EC input; here they read
[CLS] concatenated with the attention-masked mean of the token vectors. All four
heads are trained jointly with the sum of their cross-entropy losses.
"""
from __future__ import annotations

import torch
from torch import nn

from .ontology import ENTITY_HEADS, HEADS, Ontology


def _head(inputs: int, hidden: int, outputs: int, dropout: float) -> nn.Module:
    if hidden <= 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(inputs, outputs))
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(inputs, hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden, outputs))


class ContextClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, ontology: Ontology, head_hidden: int = 256,
                 dropout: float = 0.1, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.ontology = ontology
        self.freeze_backbone = freeze_backbone
        hidden = backbone.config.hidden_size
        self.hidden_size = hidden
        sizes = {head: len(ontology.classes[head]) for head in HEADS}
        self.heads = nn.ModuleDict({"subject": _head(hidden, head_hidden, sizes["subject"], dropout)})
        for head in ENTITY_HEADS:
            self.heads[head] = _head(2 * hidden, head_hidden, sizes[head], dropout)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # frozen weights: no dropout noise in the shared features
        return self

    def features(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(sequence.dtype)
        mean = (sequence * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return torch.cat([sequence[:, 0], mean], dim=-1)

    def classify(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = {"subject": self.heads["subject"](features[:, : self.hidden_size])}
        for head in ENTITY_HEADS:
            logits[head] = self.heads[head](features)
        return logits

    @staticmethod
    def joint_loss(logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(nn.functional.cross_entropy(logits[head], labels[head]) for head in HEADS)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        logits = self.classify(self.features(input_ids, attention_mask, token_type_ids))
        return {"logits": logits, "loss": self.joint_loss(logits, labels) if labels is not None else None}
