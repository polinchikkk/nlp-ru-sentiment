"""Быстрый оффлайн-скрипт для оценки сохранённого чекпойнта."""

from __future__ import annotations

import argparse, pathlib, csv
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .modeling import SentimentModel
from .data import tokenize_function, LABEL_MAP_ORIG2TRAIN
from .metrics import macro_f1


def _load_samples(tsv_path: str | pathlib.Path):
    df = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False)
    if "label" in df.columns:
        df["label"] = df["label"].map(LABEL_MAP_ORIG2TRAIN).astype("int")
    return df.to_dict(orient="records")


def main(model_dir: str, test_tsv: str, variant: str = "cls"):
    model_dir = pathlib.Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = SentimentModel.load(model_dir).model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = _load_samples(test_tsv)
    enc = [tokenize_function(s, tokenizer, variant) for s in samples]

    input_ids = torch.tensor([e["input_ids"] for e in enc]).to(device)
    attn_mask = torch.tensor([e["attention_mask"] for e in enc]).to(device)

    with torch.no_grad():
        preds = model(input_ids, attention_mask=attn_mask).logits.argmax(-1).cpu().numpy()

    if "label" in samples[0]:
        y_true = np.array([s["label"] for s in samples])
        print("Macro-F1:", macro_f1(y_true, preds))
    else:
        print("Predictions:", preds.tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--test_tsv", required=True)
    p.add_argument("--variant", choices=["cls", "entity"], default="cls")
    args = p.parse_args()
    main(args.model_dir, args.test_tsv, args.variant)
