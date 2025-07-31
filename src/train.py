"""CLI-скрипт обучения."""

from __future__ import annotations

import argparse, random, yaml, json, math, pathlib
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from .data import get_dataset, build_tokenizer, tokenised_dataset, collator
from .modeling import SentimentModel
from .metrics import macro_f1


def _seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    return {"macro_f1_np": macro_f1(y_true, y_pred)}


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    _seed_everything(cfg.get("seed", 42))

    # data
    ds = get_dataset(cfg["dataset"]["path"])
    tokenizer = build_tokenizer(cfg["model_name"], cfg.get("add_special_tokens", True))
    ds_tok = tokenised_dataset(ds, tokenizer, cfg.get("variant", "cls"))
    data_coll = collator(tokenizer)

    # model
    model = SentimentModel.from_pretrained(cfg["model_name"], tokenizer=tokenizer).model

    # training args
    out_dir = pathlib.Path(cfg.get("output_dir", "runs")) / pathlib.Path(cfg["model_name"]).name
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = cfg.get("batch_size", 8)
    steps_per_epoch = math.ceil(len(ds_tok["train"]) / batch)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=cfg.get("lr", 2e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_dir=str(out_dir / "logs"),
        logging_steps=cfg.get("logging_steps", 200),
        evaluation_strategy="steps",
        eval_steps=cfg.get("eval_steps", steps_per_epoch),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", steps_per_epoch),
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1_np",
        greater_is_better=True,
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        seed=cfg.get("seed", 42),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=data_coll,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stop", 2))],
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    with (out_dir / "config_used.json").open("w") as fp:
        json.dump(cfg, fp, indent=2)
    print(f"✓ Training done. Best checkpoint: {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
