from pypinyin import pinyin, Style
import torch.nn as nn
import torch

import models.vocabulary


class TextEncoder(nn.Module):
    def __init__(self, max_tokens: int = 512):
        super().__init__()
        self.max_tokens = max_tokens

        self.vocab = models.vocabulary.generate_pinyin_vocab()
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        self.pinyin2id = {p: i for i, p in enumerate(self.vocab)}
        self.id2pinyin = {i: p for p, i in self.pinyin2id.items()}

        self.pad_id = self.pinyin2id[self.pad_token]
        self.unk_id = self.pinyin2id[self.unk_token]

    @torch.inference_mode()
    def encode(self, text: str, device: torch.device | None = None) -> torch.Tensor:
        phones = self._chinese_to_pinyin(text)
        ids = [self.pinyin2id.get(p, self.unk_id) for p in phones]
        ids = ids[:self.max_tokens]
        ids += [self.pad_id] * (self.max_tokens - len(ids))
        out = torch.tensor([ids], dtype=torch.long)
        return out.to(device) if device else out

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """
        Convert tensor of token IDs back to pinyin string list.

        Removes <PAD> tokens.
        """
        if token_ids.ndim == 2:
            token_ids = token_ids[0]  # assume [1, T]
        return [
            self.id2pinyin.get(i.item(), self.unk_token)
            for i in token_ids
            if i.item() != self.pad_id
        ]

    def tokenize(self, text: str) -> torch.Tensor:
        return self.encode(text)

    def _chinese_to_pinyin(self, text: str) -> list[str]:
        py = pinyin(list(text), style=Style.TONE3, errors="default", strict=False)
        return [item[0] for item in py if item]

        py = pinyin(text, style=Style.TONE3, errors="ignore", strict=False)
        return [item[0] for item in py if item]
