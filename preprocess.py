import json
import pandas as pd
import re

def load_data(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess(df):
    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["input_description"].fillna("") + " " +
        df["output_description"].fillna("")
    )
    df["combined_text"] = df["combined_text"].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_data("data/problems_data.jsonl")
    df = preprocess(df)
    print(df[["combined_text", "problem_class", "problem_score"]].head())
