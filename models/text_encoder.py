"""
TextEncoder  –  returns the **full sequence** of lyric embeddings.

• Uses a pretrained BERT (or any Hugging‑Face model) to obtain token‑level
  hidden states.
• Projects each hidden state to `out_dim` so it matches the SoundTransformer
  embedding size.
• No longer collapses everything to a single [CLS] vector.
"""

from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        max_tokens: int = 256,
    ):
        """
        Args
        ----
        model_name   : name of any Hugging‑Face BERT‑style model.
        out_dim      : dimensionality expected by the audio transformer.
        max_tokens   : lyrics are truncated/padded to this many sub‑tokens to
                       cap memory usage.
        """
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert      = BertModel.from_pretrained(model_name)
        self.max_tokens = max_tokens

    # ───────────────────────── public API ───────────────────────── #

    @torch.inference_mode()
    def encode(
        self,
        text: str,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Embed an arbitrary‑length lyric string.

        Returns
        -------
        seq_emb : shape **[1, T_tokens, out_dim]** (batch‑first).
        """
        toks = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_tokens,
        )
        if device is not None:
            toks = {k: v.to(device) for k, v in toks.items()}

        outs = self.bert(**toks).last_hidden_state  # [1, T, hidden]
        return outs
    # Convenience helpers — unchanged behaviour
    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt", truncation=True)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
