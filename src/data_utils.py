
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_sample_csv(path, text_col="text", label_col="label", test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    # basic cleaning
    df = df.dropna(subset=[text_col])
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_col] if label_col in df else None)
    return train.reset_index(drop=True), test.reset_index(drop=True)

def ensure_dirs():
    for d in ["outputs", "mlruns"]:
        if not os.path.exists(d):
            os.makedirs(d)
