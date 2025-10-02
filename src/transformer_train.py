import os
import random
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ----------------------------
# Reproducibility & device
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP16 = torch.cuda.is_available()


# ----------------------------
# Paths and settings
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent  # .../src
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
BASE_MODEL_DIR = ROOT_DIR / "models" / "distilbert_news"
LOAD_MODEL_DIR = BASE_MODEL_DIR / "final"
OUTPUT_DIR = BASE_MODEL_DIR / "trained_model"
FINAL_SAVE_DIR = BASE_MODEL_DIR / "final_trained"
CACHE_DIR = ROOT_DIR / ".cache"

NUM_LABELS = 4  # AG News has 4 classes
LABEL2ID = {"world": 0, "sports": 1, "business": 2, "sci/tech": 3}
ID2LABEL = {v: k.title() if k != "sci/tech" else "Sci/Tech" for k, v in LABEL2ID.items()}


# ----------------------------
# Data loading
# ----------------------------
def load_ag_news_dataset() -> Dict[str, Any]:
    csv_files = {
        "train": DATA_DIR / "train.csv",
        "test": DATA_DIR / "test.csv",
    }
    for split, path in csv_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for split '{split}': {path}")

    dataset = load_dataset(
        "csv",
        data_files={"train": str(csv_files["train"]), "test": str(csv_files["test"])},
        cache_dir=str(CACHE_DIR),
    )

    # Normalize and map columns
    def build_example(example: Dict[str, Any]) -> Dict[str, Any]:
        title = str(example.get("Title", "")).strip()
        desc = str(example.get("Description", "")).strip()
        text = (title + ". " + desc).strip(". ")
        class_index = int(example.get("Class Index", 1))  # 1..4
        label_zero_based = class_index - 1  # 0..3
        return {"text": text, "label": label_zero_based}

    dataset = dataset.map(build_example, desc="Preparing text and labels")
    return dataset


# ----------------------------
# Tokenization
# ----------------------------
def get_tokenizer():
    # Prefer local; if missing/corrupt, fallback to hub
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(LOAD_MODEL_DIR), local_files_only=True)
    except Exception:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", cache_dir=str(CACHE_DIR))
    return tokenizer


def tokenize_function(tokenizer):
    def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=128,
        )

    return _tokenize


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    set_seed(42)

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    dataset = load_ag_news_dataset()

    # Tokenizer & tokenization with multiprocessing
    tokenizer = get_tokenizer()
    num_proc = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
    tokenized = dataset.map(
        tokenize_function(tokenizer),
        batched=True,
        num_proc=num_proc,
        # Keep 'label' for training; drop only raw 'text'
        remove_columns=[col for col in dataset["train"].column_names if col != "label"],
        desc="Tokenizing",
    )

    # Set format for torch
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Model: try local directory else hub. Ensure label maps are set.
    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            str(LOAD_MODEL_DIR),
            num_labels=NUM_LABELS,
            local_files_only=True,
            ignore_mismatched_sizes=True,
            id2label=ID2LABEL,
            label2id={k.title(): v for v, k in ID2LABEL.items()},
        )
    except Exception:
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=NUM_LABELS,
            cache_dir=str(CACHE_DIR),
            id2label=ID2LABEL,
            label2id={k.title(): v for v, k in ID2LABEL.items()},
        )

    model.to(DEVICE)

    # Data collator (dynamic padding for speed/memory)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if FP16 else None)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_dir=str(ROOT_DIR / "logs"),
        logging_steps=100,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=FP16,
        dataloader_num_workers=max(0, (os.cpu_count() or 1) - 1),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to=[],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save final model and tokenizer so Streamlit app can load locally
    trainer.save_model(str(FINAL_SAVE_DIR))
    tokenizer.save_pretrained(str(FINAL_SAVE_DIR))
    print(f"Training complete! Model saved to: {FINAL_SAVE_DIR}")


if __name__ == "__main__":
    main()
