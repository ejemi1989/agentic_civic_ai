# src/preprocess.py
from transformers import AutoTokenizer
import pandas as pd

def get_tokenizer(model_name, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.model_max_length = max_length
    return tokenizer

def tokenize_dataframe(df, tokenizer, text_col="text", label_col="label", max_length=128):
    texts = df[text_col].astype(str).tolist()
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    labels = df[label_col].astype(int).tolist() if label_col in df else None
    return encodings, labels
