# src/baseline_model.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "train_final.csv")
VAL_PATH   = os.path.join(DATA_DIR, "val_final.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test_processed.csv")

# --- Load data ---
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("Train size:", train_df.shape)
print("Val size  :", val_df.shape)
print("Test size :", test_df.shape)

# --- TF-IDF Vectorizer ---
vectorizer = TfidfVectorizer(
    max_features=20000,   # limit vocab size for speed
    ngram_range=(1,2),    # unigrams + bigrams
    stop_words='english'  # remove common stopwords
)

X_train = vectorizer.fit_transform(train_df['text_clean'])
X_val   = vectorizer.transform(val_df['text_clean'])
X_test  = vectorizer.transform(test_df['text_clean'])

y_train = train_df['label_name']
y_val   = val_df['label_name']
y_test  = test_df['label_name'] if 'label_name' in test_df.columns else None

# --- Logistic Regression ---
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg.fit(X_train, y_train)

val_preds_lr = log_reg.predict(X_val)
print("\nLogistic Regression Validation Results:")
print("Accuracy:", accuracy_score(y_val, val_preds_lr))
print(classification_report(y_val, val_preds_lr))

# --- Naive Bayes ---
print("\nTraining Multinomial Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train, y_train)

val_preds_nb = nb.predict(X_val)
print("\nNaive Bayes Validation Results:")
print("Accuracy:", accuracy_score(y_val, val_preds_nb))
print(classification_report(y_val, val_preds_nb))

# --- Optional: Test set evaluation if labels available ---
if y_test is not None and not test_df['label_name'].isna().all():
    print("\nEvaluating on Test set...")
    test_preds = log_reg.predict(X_test)
    print("Test Accuracy (LogReg):", accuracy_score(y_test, test_preds))
