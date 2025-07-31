"""Light wrapper around 🤗 AutoModelForSequenceClassification."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification


@dataclass
class SentimentModel:
    model: AutoModelForSequenceClassification

    # construction

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_labels: int = 3,
        tokenizer=None,
        **hf_kwargs,
    ) -> "SentimentModel":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, **hf_kwargs
        )
        if tokenizer is not None and tokenizer.added_tokens_encoder:
            model.resize_token_embeddings(len(tokenizer))
        return cls(model)

    # sugar

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # saving / loading

    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path):
        return cls(AutoModelForSequenceClassification.from_pretrained(path))