# src/preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "train_processed.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test_processed.csv")

# --- Load ---
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("Loaded shapes:")
print("Train:", train_df.shape)
print("Test :", test_df.shape)

# --- Train/Validation split (80/20) ---
train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df['label_name']  # ensures balanced classes
)

print("\nSplit sizes:")
print("Train:", train_split.shape)
print("Validation:", val_split.shape)

# --- Save ---
train_out = os.path.join(DATA_DIR, "train_final.csv")
val_out   = os.path.join(DATA_DIR, "val_final.csv")

train_split.to_csv(train_out, index=False)
val_split.to_csv(val_out, index=False)

print("\nSaved:")
print(" -", train_out)
print(" -", val_out)

print("\nStep 3 done ✅ Train/Validation split completed.")
