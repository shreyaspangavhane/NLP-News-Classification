
import os
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ----------------------------
# Paths
# ----------------------------
MODEL_DIR = r"D:/ML Project/models/distilbert_news/final"
DATA_PATH = r"D:/ML Project/data/"

# ----------------------------
# Load test dataset
# ----------------------------
dataset = load_dataset("csv", data_files={"test": os.path.join(DATA_PATH, "test.csv")})
dataset = dataset["test"]

# ----------------------------
# Preprocess dataset: combine Title + Description
# ----------------------------
def rename_columns(example):
    text = example['Title'] + ". " + example['Description']
    return {"text": text, "label": example["Class Index"]}

dataset = dataset.map(rename_columns)

# ----------------------------
# Load tokenizer and model from local files
# ----------------------------
tokenizer_file = os.path.join(MODEL_DIR, "tokenizer.json")  # local tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ----------------------------
# Class names (AG News)
# ----------------------------
class_names = ["World", "Sports", "Business", "Sci/Tech"]

# ----------------------------
# Evaluation
# ----------------------------
all_preds = []
all_labels = []

for example in dataset:
    text = example["text"]
    label = example["label"]

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()

    all_preds.append(pred_idx)
    all_labels.append(label)

# ----------------------------
# Compute metrics
# ----------------------------
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
