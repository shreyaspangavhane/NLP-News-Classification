from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_DIR = "D:/ML Project/models/distilbert_news/final"  # use forward slashes

# Load model locally
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

label_map = {0:'World', 1:'Sports', 2:'Business', 3:'Sci/Tech'}

def predict(text):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        return label_map[pred_idx], probs[0][pred_idx].item()

# Example
text = "NASA launches new satellite to study climate change"
category, prob = predict(text)
print(f"Predicted: {category}, Confidence: {prob:.2f}")
