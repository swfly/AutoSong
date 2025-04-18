from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, 512)  # match token embedding size
        self.model_name = model_name

    def encode(self, text: str) -> torch.Tensor:
        """Encode a text string into a fixed-size embedding."""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.proj(cls_embedding)  # [1, 512]

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize a string and return token IDs."""
        return self.tokenizer.encode(text, return_tensors="pt", truncation=True)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back into a text string."""
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
