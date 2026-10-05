"""A tiny, randomly initialised local BERT for offline pipeline checks (no downloads)."""
from __future__ import annotations

import re
from pathlib import Path


def make_tiny_backbone(folder: str | Path, texts: list[str], hidden: int = 32, seed: int = 0) -> Path:
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    words = sorted({w for text in texts for w in re.findall(r"[a-z]+|[^\sa-z]", text.casefold())})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    (folder / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(folder / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(folder)
    torch.manual_seed(seed)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=hidden, num_hidden_layers=1, num_attention_heads=4,
                         intermediate_size=hidden * 2, max_position_embeddings=128)).save_pretrained(folder)
    return folder
