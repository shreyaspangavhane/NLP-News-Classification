# src/step2_eda.py
import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import re

DATA_DIR = "data"
OUT_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# --- robust loader for common AG News CSV variants ---
def load_ag_csv(path):
    # Try headerless format (label,title,description) used by many AG News CSVs
    try:
        df = pd.read_csv(path, header=None, encoding='utf-8', names=['label','title','description'])
        # If first label looks like a header string -> fallback to header=0
        first_label = str(df.loc[0, 'label']).lower()
        if any(x in first_label for x in ('label','class','topic','category','title')):
            raise ValueError("Header detected in first row")
        return df
    except Exception:
        # fallback: read with header row
        df = pd.read_csv(path, header=0, encoding='utf-8')
        cols = list(df.columns)
        # Attempt to normalize columns
        if len(cols) >= 3:
            # pick first 3 columns as label, title, description if names unknown
            df = df.iloc[:, :3]
            df.columns = ['label','title','description']
        elif len(cols) == 2:
            df.columns = ['title','description']
            df['label'] = np.nan
        else:
            raise ValueError(f"Can't interpret columns in {path}. Columns: {cols}")
        return df

def basic_clean(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+", "", text)                 # remove urls
    text = re.sub(r"<.*?>", "", text)                   # strip html tags
    text = re.sub(r"\s+", " ", text).strip()            # normalize whitespace
    return text

def prepare_df(df):
    # ensure columns exist
    for c in ['title','description']:
        if c not in df.columns:
            df[c] = ""
    df['title'] = df['title'].fillna("").astype(str)
    df['description'] = df['description'].fillna("").astype(str)
    # combine
    df['text'] = (df['title'] + ". " + df['description']).str.strip()
    # cleaning
    df['text_clean'] = df['text'].apply(basic_clean)
    # token counts (simple whitespace tokenizer)
    df['tokens_ws'] = df['text_clean'].apply(lambda t: len(t.split()))
    df['chars'] = df['text_clean'].apply(len)
    return df

# --- load train & test ---
train_path = os.path.join(DATA_DIR, "train.csv")
test_path  = os.path.join(DATA_DIR, "test.csv")

print("Loading files:")
print("  -", train_path, "->", os.path.exists(train_path))
print("  -", test_path,  "->", os.path.exists(test_path))

train = load_ag_csv(train_path)
test  = load_ag_csv(test_path)

print("\nTRAIN shape:", train.shape)
print("TEST  shape:", test.shape)
print("\nTRAIN columns:", train.columns.tolist())

# prepare dataframes
train = prepare_df(train)
test = prepare_df(test)

# label mapping detection (AG News sometimes uses labels 1-4 or 0-3)
unique_labels = sorted([int(x) for x in pd.Series(train['label'].dropna().unique()).astype(int)]) if train['label'].notna().any() else []
print("\nFound labels (train):", unique_labels)

if set(unique_labels) <= {0,1,2,3}:
    label_map = {0:'World', 1:'Sports', 2:'Business', 3:'Sci/Tech'}
elif set(unique_labels) <= {1,2,3,4}:
    label_map = {1:'World', 2:'Sports', 3:'Business', 4:'Sci/Tech'}
else:
    # fallback: use string representation
    label_map = {l: str(l) for l in unique_labels}

if train['label'].notna().any():
    train['label'] = train['label'].astype(int)
    train['label_name'] = train['label'].map(label_map)
    test['label'] = test['label'].astype(int)
    test['label_name'] = test['label'].map(label_map)
else:
    train['label_name'] = np.nan
    test['label_name'] = np.nan

# --- prints & EDA ---
print("\n--- Sample rows (train) ---")
print(train[['label','label_name','title','description']].head(6).to_string(index=False))

print("\n--- Class distribution (train) ---")
print(train['label_name'].value_counts(dropna=False))
print("\nPercentages:")
print((train['label_name'].value_counts(normalize=True, dropna=False)*100).round(2))

# show 3 examples per class
print("\n--- 3 examples per class (train) ---")
for lbl in train['label_name'].dropna().unique():
    print(f"\nClass: {lbl}")
    sample = train[train['label_name']==lbl].sample(min(3, len(train[train['label_name']==lbl])), random_state=1)
    for i, row in sample.iterrows():
        print(" -", row['title'][:120], " /", (row['description'][:120] if row['description'] else ""))

# token / length stats
def length_stats(s):
    return {
        'count': int(s.count()),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'min': int(s.min()),
        'max': int(s.max())
    }

print("\n--- Token/char stats (train) ---")
print("Tokens (whitespace) stats:", length_stats(train['tokens_ws']))
print("Chars stats:", length_stats(train['chars']))

# simple histogram plots (saved to outputs)
plt.figure(figsize=(8,4))
plt.hist(train['tokens_ws'], bins=80)
plt.title("Histogram of token counts (train) — whitespace tokens")
plt.xlabel("tokens")
plt.ylabel("examples")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tokens_hist_train.png"))
plt.close()

plt.figure(figsize=(8,4))
plt.hist(train['chars'], bins=80)
plt.title("Histogram of char counts (train)")
plt.xlabel("chars")
plt.ylabel("examples")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chars_hist_train.png"))
plt.close()

print("\nSaved histograms to:", OUT_DIR)

# top words per class (very simple, no stopword removal)
basic_stop = set("""
the a an and of to in for on with at from by about as is are was were that this it
""".split())

def top_words(series, n=20):
    c = Counter()
    for t in series:
        for w in re.findall(r"\b\w+\b", t.lower()):
            if w in basic_stop: continue
            c[w] += 1
    return c.most_common(n)

print("\n--- Top words overall (train) ---")
print(top_words(train['text_clean'], 20))

if train['label_name'].notna().any():
    for lbl in train['label_name'].dropna().unique():
        print(f"\nTop words for class {lbl}:")
        print(top_words(train[train['label_name']==lbl]['text_clean'], 15))

# Save processed CSVs
train_out = os.path.join(OUT_DIR, "train_processed.csv")
test_out  = os.path.join(OUT_DIR, "test_processed.csv")
train.to_csv(train_out, index=False)
test.to_csv(test_out, index=False)
print("\nSaved processed CSVs:")
print(" -", train_out)
print(" -", test_out)

print("\nEDA done. Inspect the saved CSVs and the histograms in data/processed/.")
