"""Data loading & preprocessing utilities for RuSentNE-2023.

  • чтение *.tsv
  • маппинг меток {-1,0,1} → {0,1,2}
  • вариант токенизации: "cls" или "entity" (добавляем [TAG] перед сущностью)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding

SPECIAL_TOKENS = ["[PERSON]", "[ORG]", "[PROFESSION]", "[COUNTRY]", "[NATIONALITY]"]
LABEL_MAP_ORIG2TRAIN = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_TRAIN2ORIG = {v: k for k, v in LABEL_MAP_ORIG2TRAIN.items()}


def _read_tsv(path: str | Path, has_label: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False)

    # relative positions присутствуют в kaggle-версии датасета
    for col in ("entity_pos_start_rel", "entity_pos_end_rel"):
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(-1)
                .astype("int")
            )

    if has_label and "label" in df.columns:
        df["label"] = df["label"].map(LABEL_MAP_ORIG2TRAIN).astype("int")

    return df


def get_dataset(data_dir: str | Path) -> DatasetDict:
    data_dir = Path(data_dir)
    train_df = _read_tsv(data_dir / "train_data.csv", has_label=True)
    val_df = _read_tsv(data_dir / "validation_data_labeled.csv", has_label=True)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
        }
    )


def build_tokenizer(model_name: str, add_special: bool = True):
    tok = AutoTokenizer.from_pretrained(model_name)
    if add_special:
        tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


def _mark_entity(example: dict) -> str:
    tag_tok = f"[{example['entity_tag']}]"
    return example["sentence"].replace(example["entity"], f"{tag_tok} {example['entity']}", 1)


def tokenize_function(
    example: dict,
    tokenizer,
    variant: Literal["cls", "entity"] = "cls",
    max_len: int = 128,
):
    text = _mark_entity(example) if variant == "entity" else example["sentence"]
    enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length")
    if "label" in example:
        enc["labels"] = example["label"]
    return enc


def tokenised_dataset(
    ds: DatasetDict,
    tokenizer,
    variant: Literal["cls", "entity"] = "cls",
    max_len: int = 128,
):
    return ds.map(
        lambda ex: tokenize_function(ex, tokenizer, variant, max_len),
        batched=False,
        remove_columns=ds["train"].column_names,
    )


def collator(tokenizer):
    return DataCollatorWithPadding(tokenizer)